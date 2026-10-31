"""Small-model paired compact-mask benchmark, explicit 0/1/default controls.

Two isolated same-binary servers remain loaded; requests are serial. Warmups,
path proof and hashing are excluded from measured windows. This is synthetic
prefill latency evidence, not trained-model throughput or quality evidence.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

if not __debug__:
    raise RuntimeError('Do not optimize away benchmark checks')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mask_native_launcher', ROOT / 'integration/native_execution_features_smoke.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def set_arg(command, name, value):
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend((name, str(value)))


def summary(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
                mean=statistics.mean(values), cv_percent=100 * statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_server(args, label, setting, verbose=False):
    server = launcher.NativeServer(args.executable, args.model, args.output_dir / label, args.gpu)
    if setting == 'auto':
        server.env.pop('WILDLLAMA_DEVICE_MASK', None)
    else:
        server.env['WILDLLAMA_DEVICE_MASK'] = setting
    for name, value in (('--ctx-size', args.context), ('--batch-size', args.batch),
                        ('--ubatch-size', args.batch), ('--parallel', 1),
                        ('--threads', args.threads), ('--threads-batch', args.threads),
                        ('--cache-ram', 0), ('--flash-attn', 'on'),
                        ('--log-verbosity', 5 if verbose else 4)):
        set_arg(server.command, name, value)
    if args.context_metadata_key:
        # Only diagnostic metadata in the isolated process is overridden.
        # This permits long-prefill tests with tiny generated hybrid fixtures.
        set_arg(server.command, '--override-kv', args.context_metadata_key + '=int:' + str(args.context))
    if args.cpu_mask:
        for name, value in (('--cpu-mask', args.cpu_mask), ('--cpu-mask-batch', args.cpu_mask),
                            ('--cpu-strict', 1), ('--cpu-strict-batch', 1)):
            set_arg(server.command, name, value)
    return server


def request_body(n, probs=0):
    return dict(prompt=[3 + (i * 13 + 11) % 256 for i in range(n)], n_predict=1,
                seed=424242, temperature=0, ignore_eos=True, cache_prompt=False,
                return_tokens=True, stream=False, n_probs=probs, post_sampling_probs=True)


def stable(response):
    return {k: response.get(k) for k in ('content', 'tokens', 'stop', 'stop_type',
            'stopping_word', 'truncated', 'completion_probabilities')}


def run(args):
    expected = {}
    raw, warmups, windows, cases, proof = [], [], [], [], {}
    # DEBUG is used only in a distinct untimed proof process, with the same
    # model/placement/context as the timed process. CPU must remain fallback.
    with make_server(args, 'proof', args.after_mask, True) as server:
        response = server.request('/completion', request_body(max(args.prompts)))
        assert len(response['tokens']) == 1
        proof['command'] = server.command
    proof['log_sha256'] = launcher.sha(server.logfile)
    proof['actual_compact_input_count'] = server.logfile.read_text(encoding='utf-8', errors='replace').count('device-mask inputs:')
    if args.gpu and args.after_mask in ('1', 'auto'):
        assert proof['actual_compact_input_count'] > 0, 'No compact CUDA masks executed in proof process'
    else:
        assert proof['actual_compact_input_count'] == 0, 'Disabled/CPU control admitted CUDA mask'
    with ExitStack() as resources:
        startup = [('before', args.before_mask), ('after', args.after_mask)]
        if args.reverse_start_order:
            startup.reverse()
        servers = {label: resources.enter_context(make_server(args, label, value))
                   for label, value in startup}
        commands = {label: server.command for label, server in servers.items()}
        for n in args.prompts:
            for label, server in servers.items():
                response = server.request('/completion', request_body(n, 5))
                assert len(response.get('tokens', [])) == 1 and response.get('stop_type') == 'limit'
                assert response.get('timings', {}).get('prompt_n') == n, 'Prompt cache/reuse or truncation invalidates benchmark'
                view = stable(response)
                if n not in expected:
                    expected[n] = view
                else:
                    assert view == expected[n], 'Token/content/probability output changed'
                raw.append(dict(phase='correctness', label=label, prompt_tokens=n, response=response))
            expected_tokens = expected[n]['tokens']

            def window(label, measured, cycle=-1, slot=-1):
                assert all(s.process.poll() is None for s in servers.values())
                started = time.perf_counter_ns()
                response = servers[label].request('/completion', request_body(n))
                wall_ms = (time.perf_counter_ns() - started) / 1e6
                assert response.get('tokens') == expected_tokens and response.get('stop_type') == 'limit'
                timings = response['timings']
                assert timings['prompt_n'] == n and response['tokens_evaluated'] == n
                prompt_ms = timings['prompt_ms']
                assert math.isfinite(prompt_ms) and prompt_ms > 0 and wall_ms > 0
                item = dict(label=label, prompt_tokens=n, cycle=cycle, slot=slot,
                            prompt_ms=prompt_ms, wall_ms=wall_ms, prompt_tps=n * 1000 / prompt_ms,
                            wall_prompt_tps=n * 1000 / wall_ms, native_timings=timings)
                (windows if measured else warmups).append(item)
                raw.append(dict(phase='measurement' if measured else 'warmup', **item, response=response))

            for _ in range(args.warmups):
                window('before', False)
                window('after', False)
            for cycle in range(args.cycles):
                # Reverse whole ABBA order on odd cycles for an additional
                # scheduling/thermal balance while keeping paired neighbours.
                order = ('before', 'after', 'after', 'before') if cycle % 2 == 0 else ('after', 'before', 'before', 'after')
                for slot, label in enumerate(order):
                    window(label, True, cycle, slot)
            metrics = {}
            for key in ('prompt_ms', 'wall_ms', 'prompt_tps', 'wall_prompt_tps'):
                values = {label: summary([w[key] for w in windows if w['label'] == label and w['prompt_tokens'] == n])
                          for label in servers}
                values['median_percent'] = (values['after']['median'] / values['before']['median'] - 1) * 100
                values['paired_percent'] = []
                for cycle in range(args.cycles):
                    pair = {label: statistics.mean([w[key] for w in windows if w['label'] == label and w['prompt_tokens'] == n and w['cycle'] == cycle]) for label in servers}
                    values['paired_percent'].append((pair['after'] / pair['before'] - 1) * 100)
                metrics[key] = values
            cases.append(dict(prompt_tokens=n, metrics=metrics, outputs_exact=True,
                              semantic_sha256=digest(expected[n])))
            print(json.dumps(dict(prompt_tokens=n, prompt_ms_percent=metrics['prompt_ms']['median_percent'],
                                 wall_ms_percent=metrics['wall_ms']['median_percent'])), flush=True)
    dump = args.output_dir / 'responses.json'
    dump.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return dict(status='passed', gpu=args.gpu, before_mask=args.before_mask, after_mask=args.after_mask,
                context=args.context, batch=args.batch, threads=args.threads, cycles=args.cycles,
                requested_strict_native_cpu_mask=args.cpu_mask,
                synthetic_context_metadata_key=args.context_metadata_key,
                warmups_per_payload_per_case=args.warmups, fixtures_small_only=True,
                model_bytes=args.model.stat().st_size, model_sha256=launcher.sha(args.model),
                executable_sha256=launcher.sha(args.executable), backend_sha256=launcher.sha(servers['after'].backend),
                harness_sha256=launcher.sha(Path(__file__)), launcher_sha256=launcher.sha(Path(launcher.__file__)),
                commands=commands, reverse_start_order=args.reverse_start_order,
                proof=proof, cases=cases, windows=windows, warmups=warmups,
                response_dump_sha256=launcher.sha(dump), exact_tokens_and_probabilities=True,
                scope='Small random-weight dense synthetic prefill latency; does not establish trained-model TPS or language quality.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--reverse-start-order', action='store_true')
    p.add_argument('--before-mask', choices=('0', '1', 'auto'), default='0')
    p.add_argument('--after-mask', choices=('0', '1', 'auto'), default='1')
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--context-metadata-key', choices=('llama.context_length', 'qwen35.context_length'))
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--prompts', nargs='+', type=int, default=[512, 2048, 4096])
    p.add_argument('--cycles', type=int, default=8)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--cpu-mask', help='Optional native strict worker CPU mask in hexadecimal; identical in both controls')
    args = p.parse_args()
    for name in ('executable', 'model', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    assert args.output_dir.drive.upper() != 'C:'
    assert args.executable.is_file() and args.model.is_file()
    assert 0 < args.model.stat().st_size <= 64 * 1024 * 1024
    assert args.cycles >= 2 and args.warmups >= 1 and 0 < max(args.prompts) < args.context
    if args.cpu_mask:
        assert 0 < int(args.cpu_mask, 16) < (1 << 256), 'Invalid bounded CPU mask'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / 'report.json'
    try:
        result = run(args)
    except BaseException as e:
        report.write_text(json.dumps(dict(status='failed', error=str(e)), indent=2) + '\n')
        raise
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='passed', cases=len(result['cases']), outputs_exact=True)))
