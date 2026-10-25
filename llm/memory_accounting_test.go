package llm

import (
	"context"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestMemoryAccountingFlashCachesAndAutoGrowth(t *testing.T) {
	const mib = uint64(1024 * 1024)
	for _, step := range []int{1, 37, 32768} {
		t.Run(fmt.Sprintf("pipe_chunk_%d", step), func(t *testing.T) {
			runner := &llamaServerRunner{gpus: []ml.DeviceInfo{{Name: "CUDA0"}}}
			w := &memoryParsingWriter{runner: runner, inner: io.Discard}
			transcript := "load_tensors: CUDA0 model buffer size = 2000.00 MiB\n" +
				"llama_context: constructing llama_context\n" +
				"llama_kv_cache: CUDA0 KV buffer size = 1632.00 MiB\n" +
				"llama_memory_recurrent: CUDA0 RS buffer size = 112.00 MiB\n" +
				"operator (): creating indexer KV cache, size = 131072 cells\n" +
				"llama_kv_cache: CUDA0 KV buffer size = 408.00 MiB\n" +
				"gpu-exclusive: 1021.00 MiB bounded expert cache, 3 layouts, 144 output-row tiles\n" +
				"sched_reserve: CUDA0 compute buffer size = 278.00 MiB\n" +
				"alloc_compute_meta: CUDA0 compute buffer size = 10.00 MiB\n" +
				"sched_reserve: CUDA_Host compute buffer size = 5.00 MiB\n" +
				"gpu-exclusive: 6688.00 MiB bounded expert cache, 3 layouts, 144 output-row tiles\n" +
				"sched_reserve: CUDA0 compute buffer size = 278.00 MiB\n"
			// Re-reading a complete startup transcript must not count another
			// allocation; each cache and graph component replaces its snapshot.
			for range 2 {
				for at := 0; at < len(transcript); at += step {
					if _, err := w.Write([]byte(transcript[at:min(at+step, len(transcript))])); err != nil {
						t.Fatal(err)
					}
				}
			}
			wantGPU := uint64(2000+1632+112+408+6688+278+10) * mib
			if total, gpu := runner.MemorySize(); total != wantGPU+5*mib || gpu != wantGPU {
				t.Fatalf("memory = %d/%d, want %d/%d", total, gpu, wantGPU+5*mib, wantGPU)
			}
			if runner.vramByDevice["CUDA0"] != wantGPU || runner.memModelFileBacked != 2000*mib {
				t.Fatal("expert cache lost from device accounting or counted as mirrored model weights")
			}
		})
	}
}

func TestMemoryAccountingEqualKVAndContextReplacement(t *testing.T) {
	const mib = uint64(1024 * 1024)
	runner := &llamaServerRunner{}
	w := &memoryParsingWriter{runner: runner, inner: io.Discard}
	_, _ = w.Write([]byte("llama_context: constructing llama_context\n" +
		"llama_kv_cache: CUDA0 KV buffer size = 408.00 MiB\n" +
		"operator (): creating indexer KV cache, size = 131072 cells\n" +
		"llama_kv_cache: CUDA0 KV buffer size = 408.00 MiB\n" +
		"llama_kv_cache: CUDA0 KV buffer size = 408.00 MiB\n" +
		"gpu-exclusive: 1021.00 MiB bounded expert cache, 3 layouts, 144 output-row tiles\n" +
		"sched_reserve: CUDA0 compute buffer size = 278.00 MiB\n"))
	if runner.memGPU != (408+408+1021+278)*mib {
		t.Fatal("distinct equal-sized KV caches were merged or repeated snapshot was added twice")
	}
	_, _ = w.Write([]byte("llama_context: constructing llama_context\n" +
		"llama_kv_cache: CUDA0 KV buffer size = 100.00 MiB\n"))
	if runner.memGPU != (100+278)*mib {
		t.Fatal("new context retained stale KV/expert banks or lost the graph buffer")
	}
}

func TestMemoryAccountingNativeManagedVRAM(t *testing.T) {
	const mib = uint64(1024 * 1024)
	id := ml.DeviceID{ID: "0", Library: "CUDA"}
	status := NewStatusWriter(io.Discard)
	runner := &llamaServerRunner{
		status: status,
		gpus:   []ml.DeviceInfo{{DeviceID: id, Name: "CUDA0", TotalMemory: 16384 * mib}},
	}
	w := &memoryParsingWriter{runner: runner, inner: status}
	_, _ = w.Write([]byte("using device CUDA0 (NVIDIA GPU) (0000:01:00.0) - 15000 MiB free\n" +
		"load_tensors: CUDA0 model buffer size = 2000.00 MiB\n" +
		"load_tensors: CPU model buffer size = 128.00 MiB\n" +
		"wildllama_activity {\"kind\":\"memory\",\"stage\":\"ram\",\"memory\":{\"ram_bytes\":4096}}\n"))
	if _, gpu := runner.MemorySize(); gpu != 2000*mib {
		t.Fatal("RAM-only telemetry suppressed the legacy GPU accounting")
	}
	for _, sample := range []struct{ used, limit uint64 }{
		{12616745344, 12884901888}, // first Flash snapshot after auto growth
		{12658688384, 12884901888}, // graph pool grows during generation
		{11000 * mib, 0},           // physical-only policy still reports real bytes
		{0, 0},                     // an observed zero is not a missing snapshot
	} {
		_, _ = w.Write([]byte(fmt.Sprintf("wildllama_activity {\"kind\":\"memory\",\"stage\":\"cuda\",\"memory\":{\"vram_bytes\":%d,\"vram_limit_bytes\":%d}}\n", sample.used, sample.limit)))
		// A later RAM event must preserve the observed CUDA bytes.
		_, _ = w.Write([]byte("wildllama_activity {\"kind\":\"memory\",\"stage\":\"ram\",\"memory\":{\"ram_bytes\":8192}}\n"))
		if total, gpu := runner.MemorySize(); gpu != sample.used || total != sample.used+128*mib {
			t.Fatalf("managed snapshot double-counted/missed memory: %d/%d", total, gpu)
		}
		if got := runner.VRAMByGPU(id); got != sample.used {
			t.Fatalf("scheduler device usage = %d, want observed %d", got, sample.used)
		}
		infos := runner.GetDeviceInfos(context.Background())
		if infos[0].FreeMemory != 15000*mib-sample.used {
			t.Fatalf("scheduler free memory = %d, want %d", infos[0].FreeMemory, 15000*mib-sample.used)
		}
	}
	if runner.VRAMByGPU(ml.DeviceID{ID: "unknown", Library: "CUDA"}) != 0 {
		t.Fatal("aggregate snapshot attributed to an unrelated GPU")
	}
}

func TestMemoryAccountingManagedAggregateDoesNotInventDeviceSplit(t *testing.T) {
	status := NewStatusWriter(io.Discard)
	runner := &llamaServerRunner{
		status:   status,
		memTotal: 350, memGPU: 300,
		vramByDevice: map[string]uint64{"CUDA0": 100, "CUDA1": 200},
		gpus: []ml.DeviceInfo{
			{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Name: "CUDA0", TotalMemory: 1000},
			{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, Name: "CUDA1", TotalMemory: 1000},
		},
	}
	_, _ = status.Write([]byte("wildllama_activity {\"kind\":\"memory\",\"stage\":\"cuda\",\"memory\":{\"vram_bytes\":600,\"vram_limit_bytes\":800}}\n"))
	if total, gpu := runner.MemorySize(); total != 650 || gpu != 600 {
		t.Fatalf("aggregate usage = %d/%d, want 650/600", total, gpu)
	}
	for _, gpu := range runner.gpus {
		if runner.VRAMByGPU(gpu.DeviceID) != runner.vramByDevice[gpu.Name] {
			t.Fatal("aggregate native value was duplicated across devices")
		}
	}
}

func TestMemoryAccountingIgnoresOversizedPartialLine(t *testing.T) {
	runner := &llamaServerRunner{}
	w := &memoryParsingWriter{runner: runner, inner: io.Discard}
	_, _ = w.Write([]byte(strings.Repeat("x", 40000)))
	_, _ = w.Write([]byte("load_tensors: CUDA0 model buffer size = 9999.00 MiB\n"))
	_, _ = w.Write([]byte("load_tensors: CUDA0 model buffer size = 10.00 MiB\n"))
	if len(w.tail) > 32768 || runner.memGPU != 10*1024*1024 {
		t.Fatal("oversized line consumed memory or its suffix was parsed as a fresh record")
	}
}
