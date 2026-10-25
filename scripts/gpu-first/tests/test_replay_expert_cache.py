import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "replay-expert-cache.py"
SPEC = importlib.util.spec_from_file_location("replay_expert_cache", MODULE)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def route(sequence, layer, experts, size=1, tokens=1, k=None, model=0):
    return {"kind": "route", "seq": sequence, "model": model, "eval": sequence,
            "layer": layer, "tokens": tokens, "k": k or len(experts) // tokens,
            "bytes": size, "experts": experts}


class CacheReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "trace.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def trace(self, records, end=True, truncated=False):
        values = [{"kind": "header", "version": 1, "unit": "expert_bundle", "scope": "runner_shared"}, *records]
        if end:
            values.append({"kind": "end", "routes": len(records), "truncated": truncated})
        self.path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf8")

    def test_read_rejects_missing_or_truncated_end_unless_explicit(self):
        records = [route(1, 0, [3])]
        self.trace(records, end=False)
        with self.assertRaisesRegex(ValueError, "no end record"):
            list(replay.read_trace(self.path))
        self.assertEqual(list(replay.read_trace(self.path, True)), records)
        self.trace(records, truncated=True)
        with self.assertRaisesRegex(ValueError, "truncated"):
            list(replay.read_trace(self.path))
        self.assertEqual(list(replay.read_trace(self.path, True)), records)

    def test_malformed_record_is_not_treated_as_partial_capture(self):
        self.trace([route(1, 0, [3])], end=False)
        with self.path.open("a", encoding="utf8") as stream:
            stream.write('{"kind":"route"')
        with self.assertRaisesRegex(ValueError, "Malformed"):
            list(replay.read_trace(self.path, True))

    def test_routes_are_deduplicated_only_within_the_same_transfer_batch(self):
        simulation = replay.Replay(4, 0, "lru")
        simulation.route(route(1, 0, [0, 1, 0, 1], size=2, tokens=2))
        self.assertEqual(simulation.h2d_bytes, 4)
        self.assertEqual(simulation.ssd_bytes, 4)
        simulation.route(route(2, 0, [0, 1], size=2))
        self.assertEqual(simulation.h2d_bytes, 4)
        self.assertEqual(simulation.gpu.hits, 2)

    def test_ram_hits_avoid_ssd_but_still_require_upload(self):
        simulation = replay.Replay(2, 6, "lru")
        for sequence, expert in enumerate([0, 1, 2, 0], 1):
            simulation.route(route(sequence, 0, [expert], size=2))
        self.assertEqual(simulation.h2d_bytes, 8)
        self.assertEqual(simulation.ssd_bytes, 6)
        self.assertEqual(simulation.ram.hits, 1)

    def test_lru_cycle_and_frequency_admission_have_different_traffic(self):
        lru = replay.Replay(2, 0, "lru")
        admitted = replay.Replay(2, 0, "admission")
        # A recurring pair with one-use scanning traffic between repetitions.
        accesses = [0, 1, 0, 1, 2, 3, 4, 5, 0, 1]
        for sequence, expert in enumerate(accesses, 1):
            record = route(sequence, 0, [expert])
            lru.route(record)
            admitted.route(record)
        self.assertEqual(lru.gpu.hits, 2)
        self.assertEqual(admitted.gpu.hits, 4)
        self.assertLess(admitted.h2d_bytes, lru.h2d_bytes)
        self.assertEqual(admitted.gpu.used, 2)

    def test_byte_budgets_handle_oversized_experts_and_protected_batches(self):
        cache = replay.ExpertCache(3, "slru")
        cache.access_batch({"a": 1}, 2)
        cache.access_batch({"a": 1, "b": 1}, 2)
        self.assertIn("a", cache)
        self.assertNotIn("b", cache)
        self.assertEqual(cache.used, 2)
        cache.access_batch({"oversized": 1}, 4)
        self.assertEqual(cache.used, 2)
        self.assertEqual(cache.bypasses, 2)

    def test_layers_and_model_identities_do_not_alias(self):
        simulation = replay.Replay(6, 0, "lru")
        for record in [route(1, 0, [3]), route(2, 1, [3]), route(3, 0, [3], model=1)]:
            simulation.route(record)
        self.assertEqual(simulation.gpu.misses, 3)
        simulation.route(route(4, 0, [3]))
        self.assertEqual(simulation.gpu.hits, 1)
        with self.assertRaisesRegex(ValueError, "byte size changed"):
            simulation.route(route(5, 0, [3], size=2))

    def test_zero_capacity_never_allocates_cache_bytes(self):
        simulation = replay.Replay(0, 0, "slru")
        for sequence in range(1, 4):
            simulation.route(route(sequence, 0, [7], size=9))
        self.assertEqual((simulation.gpu.used, simulation.ram.used), (0, 0))
        self.assertEqual((simulation.ssd_bytes, simulation.h2d_bytes), (27, 27))

    def test_complete_trace_detects_count_schema_and_sequence_errors(self):
        for invalid in [route(1, 0, [-1]), route(1, -1, [0]), route(1, 0, [0], tokens=2),
                        {**route(1, 0, [0]), "bytes": True}]:
            self.trace([invalid])
            with self.assertRaises(ValueError):
                list(replay.read_trace(self.path))
        self.trace([route(2, 0, [0]), route(1, 0, [0])])
        with self.assertRaisesRegex(ValueError, "sequence"):
            list(replay.read_trace(self.path))
        self.trace([route(2, 0, [0])])
        with self.assertRaisesRegex(ValueError, "missing sequence"):
            list(replay.read_trace(self.path))
        self.trace([route(1, 0, [0])])
        content = self.path.read_text(encoding="utf8").replace('"routes": 1', '"routes": 2')
        self.path.write_text(content, encoding="utf8")
        with self.assertRaisesRegex(ValueError, "route count"):
            list(replay.read_trace(self.path))

    def test_replay_grid_uses_mib_and_reports_no_throughput_prediction(self):
        self.trace([route(1, 0, [0, 1], size=1024**2), route(2, 0, [0, 1], size=1024**2)])
        result = replay.replay(self.path, [1, 2], 1, ["lru", "slru"])
        self.assertEqual(len(result["results"]), 4)
        large = result["results"][2]
        self.assertEqual(large["gpu"]["capacity_bytes"], 2 * 1024**2)
        self.assertEqual(large["gpu"]["byte_hit_ratio"], .5)
        self.assertIn("tokens/s", result["limitations"][0])


if __name__ == "__main__":
    unittest.main()
