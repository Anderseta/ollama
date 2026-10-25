package llm

import (
	"errors"
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

func gpuExclusiveOptions(t *testing.T) api.Options {
	t.Helper()
	for _, key := range []string{"OLLAMA_GPU_EXCLUSIVE", "OLLAMA_GPU_DIAGNOSTICS", "OLLAMA_GPU_VRAM_LIMIT_MIB", "LLAMA_ARG_GPU_EXCLUSIVE", "LLAMA_ARG_GPU_DIAGNOSTICS", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB", "LLAMA_ARG_N_GPU_LAYERS", "LLAMA_ARG_CPU_MOE", "LLAMA_ARG_MOE_CACHE_MIB", "LLAMA_ARG_FLASH_ATTN", "OLLAMA_FLASH_ATTENTION", "LLAMA_ARG_NO_WARMUP"} {
		t.Setenv(key, "")
	}
	opts := api.DefaultOptions()
	enabled := true
	opts.GPUExclusive = &enabled
	return opts
}

func TestGPUExclusiveBackendAndPlacementValidation(t *testing.T) {
	cuda := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "CUDA"}, ComputeMajor: 8}
	for _, name := range []string{"no CUDA", "other backend", "CPU request", "partial layers", "partial environment", "unsupported FA", "uncached CPU experts", "cached CPU experts", "normal", "all"} {
		t.Run(name, func(t *testing.T) {
			opts := gpuExclusiveOptions(t)
			gpus := []ml.DeviceInfo{cuda}
			failure := true
			switch name {
			case "no CUDA":
				gpus = nil
			case "other backend":
				gpus[0].Library = "Vulkan"
			case "CPU request":
				opts.NumGPU = 0
			case "partial layers":
				opts.NumGPU = 1
			case "partial environment":
				opts.RunnerEnv = map[string]string{"LLAMA_ARG_N_GPU_LAYERS": "1"}
			case "unsupported FA":
				gpus[0].ComputeMajor = 5
				opts.RunnerEnv = map[string]string{"LLAMA_ARG_FLASH_ATTN": "on"}
			case "uncached CPU experts":
				opts.RunnerEnv = map[string]string{"LLAMA_ARG_CPU_MOE": "1", "LLAMA_ARG_MOE_CACHE_MIB": "0"}
			case "cached CPU experts":
				opts.RunnerEnv = map[string]string{"LLAMA_ARG_CPU_MOE": "1", "LLAMA_ARG_MOE_CACHE_MIB": "1024"}
				failure = false
			case "normal":
				opts.GPUExclusive = nil
				gpus = nil
				failure = false
			case "all":
				failure = false
			}
			if err := ValidateGPUExecution(opts, gpus, 4); (err != nil) != failure {
				t.Fatalf("error %v; expected failure %v", err, failure)
			}
		})
	}
}

func TestGPUExclusiveNativeArgumentsAndWarmup(t *testing.T) {
	opts := gpuExclusiveOptions(t)
	limit, enabled := 12288, true
	opts.GPUVRAMLimitMiB, opts.GPUDiagnostics = &limit, &enabled
	got, err := appendGPUExecutionArgs(nil, opts)
	want := []string{"--gpu-exclusive", "-ngl", "all", "--fit", "off", "--gpu-vram-limit-mib", "12288", "--gpu-diagnostics"}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("args %v %v", got, err)
	}
	for _, raw := range []string{"", "0", "1"} {
		opts.RunnerEnv = map[string]string{"LLAMA_ARG_NO_WARMUP": raw}
		got, err := appendRunnerWarmupArgs(nil, opts)
		if err != nil {
			t.Fatal(err)
		}
		if raw == "" && len(got) != 0 {
			t.Fatal("implicit warmup defeats lazy GPU mode")
		}
		if raw == "0" && !reflect.DeepEqual(got, []string{"--warmup"}) {
			t.Fatal("explicit warmup not preserved")
		}
		if raw == "1" && !reflect.DeepEqual(got, []string{"--no-warmup"}) {
			t.Fatal("explicit no-warmup not preserved")
		}
	}
	runner := &llamaServerRunner{launch: llamaServerLaunchConfig{opts: opts, projectors: []string{"projector"}}}
	if runner.shouldRetryMMProjCPUOffload(errors.New("out of memory")) {
		t.Fatal("exclusive mode retried with projector on CPU")
	}
	if !runner.launch.requiresMMProjGPUOffload() {
		t.Fatal("exclusive projector not kept on GPU")
	}
}

func TestGPUVRAMLimitWithoutExclusiveMode(t *testing.T) {
	opts := gpuExclusiveOptions(t)
	enabled, limit := false, 12288
	opts.GPUExclusive, opts.GPUVRAMLimitMiB = &enabled, &limit
	got, err := appendGPUExecutionArgs(nil, opts)
	if err != nil || !reflect.DeepEqual(got, []string{"--gpu-vram-limit-mib", "12288"}) {
		t.Fatalf("VRAM limit silently ignored: %v %v", got, err)
	}
}
