package llm

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/ml"
)

// ValidateGPUExecution rejects conflicts before native loading can select a CPU
// fallback. Native per-operator enforcement remains authoritative for computation.
func ValidateGPUExecution(opts api.Options, gpus []ml.DeviceInfo, modelLayers uint64) error {
	cfg, err := api.ResolveGPUExecution(opts)
	if err != nil || !cfg.Exclusive {
		return err
	}
	if len(gpus) == 0 {
		return fmt.Errorf("gpu_exclusive requires an available CUDA device; CPU fallback is disabled")
	}
	for _, gpu := range gpus {
		if !strings.EqualFold(gpu.Library, "CUDA") {
			return fmt.Errorf("gpu_exclusive requires CUDA devices, got %s", gpu.Library)
		}
	}
	if opts.NumGPU == 0 || (opts.NumGPU > 0 && modelLayers > 0 && uint64(opts.NumGPU) < modelLayers) {
		return fmt.Errorf("gpu_exclusive conflicts with num_gpu=%d; use -1 or all model layers", opts.NumGPU)
	}
	if opts.MainGPU != nil && (*opts.MainGPU < 0 || *opts.MainGPU >= len(gpus)) {
		return fmt.Errorf("gpu_exclusive main_gpu is outside the selected CUDA device group")
	}
	if raw := gpuNativeEnvironment(opts, "LLAMA_ARG_N_GPU_LAYERS"); raw != "" && raw != "auto" && raw != "all" && raw != "-1" {
		n, err := strconv.ParseUint(raw, 10, 64)
		if err != nil || n == 0 || (modelLayers > 0 && n < modelLayers) {
			return fmt.Errorf("gpu_exclusive conflicts with LLAMA_ARG_N_GPU_LAYERS=%q; use all", raw)
		}
	}
	if LlamaServerFlashAttentionForOptions(gpus, opts) == ml.FlashAttentionEnabled && !ml.FlashAttentionSupported(gpus) {
		return fmt.Errorf("gpu_exclusive requested Flash Attention on a CUDA device that does not support it; select a supported device or explicitly disable Flash Attention")
	}
	if raw := gpuNativeEnvironment(opts, "LLAMA_ARG_CPU_MOE"); raw != "" {
		cpuStorage, err := strconv.ParseBool(raw)
		if err != nil {
			return fmt.Errorf("invalid LLAMA_ARG_CPU_MOE: %w", err)
		}
		if cpuStorage {
			cache, err := strconv.ParseUint(gpuNativeEnvironment(opts, "LLAMA_ARG_MOE_CACHE_MIB"), 10, 64)
			if err != nil || cache == 0 {
				return fmt.Errorf("gpu_exclusive with CPU-stored experts requires a positive LLAMA_ARG_MOE_CACHE_MIB for CUDA execution")
			}
		}
	}
	return nil
}

func gpuNativeEnvironment(opts api.Options, key string) string {
	if value, exists := opts.RunnerEnv[key]; exists {
		return value
	}
	return envconfig.Var(key)
}

func setMOERuntimeEnvironment(environment map[string]string, enabled bool) {
	environment["WILDLLAMA_MOE_RUNTIME"] = "0"
	if enabled {
		environment["WILDLLAMA_MOE_RUNTIME"] = "1"
	}
}

func appendGPUExecutionArgs(params []string, opts api.Options) ([]string, error) {
	cfg, err := api.ResolveGPUExecution(opts)
	if err != nil {
		return nil, err
	}
	if cfg.Exclusive {
		params = append(params, "--gpu-exclusive", "-ngl", "all", "--fit", "off")
	}
	if cfg.Exclusive || cfg.VRAMLimitMiB > 0 {
		params = append(params, "--gpu-vram-limit-mib", strconv.Itoa(cfg.VRAMLimitMiB))
	}
	if cfg.Diagnostics {
		params = append(params, "--gpu-diagnostics")
	}
	return params, nil
}

func appendRunnerWarmupArgs(params []string, opts api.Options) ([]string, error) {
	raw, exists := opts.RunnerEnv["LLAMA_ARG_NO_WARMUP"]
	if !exists {
		raw, exists = os.LookupEnv("LLAMA_ARG_NO_WARMUP")
	}
	if !exists || raw == "" {
		return params, nil
	}
	disabled, err := strconv.ParseBool(raw)
	if err != nil {
		return nil, fmt.Errorf("LLAMA_ARG_NO_WARMUP must be a boolean: %w", err)
	}
	if disabled {
		return append(params, "--no-warmup"), nil
	}
	return append(params, "--warmup"), nil
}
