package api

import (
	"fmt"
	"strconv"

	"github.com/ollama/ollama/envconfig"
)

// GPUExecutionConfig is the effective opt-in native execution policy.
// VRAMLimitMiB limits this runner's native allocations, not other GPU processes.
type GPUExecutionConfig struct {
	KVCacheInitialTokens int
	Runtime              bool
	Exclusive            bool
	Diagnostics          bool
	VRAMLimitMiB         int
}

// ResolveGPUExecution retains explicit false and zero overrides. A cleared
// runner environment value removes the inherited native setting.
func ResolveGPUExecution(opts Options) (GPUExecutionConfig, error) {
	var cfg GPUExecutionConfig
	var err error
	cfg.KVCacheInitialTokens, err = ResolveKVCacheInitialTokens(opts)
	if err != nil {
		return cfg, err
	}
	if opts.MOERuntime != nil {
		cfg.Runtime = *opts.MOERuntime
	} else {
		raw, exists := opts.RunnerEnv["WILDLLAMA_MOE_RUNTIME"]
		if !exists {
			raw = envconfig.Var("OLLAMA_MOE_RUNTIME")
		}
		if raw != "" {
			enabled, err := strconv.ParseBool(raw)
			if err != nil {
				return cfg, fmt.Errorf("moe_runtime must be a boolean: %w", err)
			}
			cfg.Runtime = enabled
		}
	}
	if opts.GPUExclusive != nil {
		cfg.Exclusive = *opts.GPUExclusive
	} else {
		raw := gpuExecutionEnv(opts, "OLLAMA_GPU_EXCLUSIVE", "LLAMA_ARG_GPU_EXCLUSIVE")
		if raw != "" {
			enabled, err := strconv.ParseBool(raw)
			if err != nil {
				return cfg, fmt.Errorf("gpu_exclusive must be a boolean: %w", err)
			}
			cfg.Exclusive = enabled
		}
	}
	if opts.GPUVRAMLimitMiB != nil {
		cfg.VRAMLimitMiB = *opts.GPUVRAMLimitMiB
	} else {
		raw := gpuExecutionEnv(opts, "OLLAMA_GPU_VRAM_LIMIT_MIB", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB")
		if raw != "" {
			value, err := strconv.ParseInt(raw, 10, 64)
			if err != nil || value < 0 || value > 2097151 {
				return cfg, fmt.Errorf("gpu_vram_limit_mib must be an integer between 0 and 2097151 MiB")
			}
			cfg.VRAMLimitMiB = int(value)
		}
	}
	if opts.GPUDiagnostics != nil {
		cfg.Diagnostics = *opts.GPUDiagnostics
	} else if raw := gpuExecutionEnv(opts, "OLLAMA_GPU_DIAGNOSTICS", "LLAMA_ARG_GPU_DIAGNOSTICS"); raw != "" {
		enabled, err := strconv.ParseBool(raw)
		if err != nil {
			return cfg, fmt.Errorf("gpu_diagnostics must be a boolean: %w", err)
		}
		cfg.Diagnostics = enabled
	}
	if cfg.VRAMLimitMiB < 0 || cfg.VRAMLimitMiB > 2097151 {
		return cfg, fmt.Errorf("gpu_vram_limit_mib must be an integer between 0 and 2097151 MiB")
	}
	return cfg, nil
}

func gpuExecutionEnv(opts Options, server, native string) string {
	if value, exists := opts.RunnerEnv[native]; exists {
		return value
	}
	if value := envconfig.Var(server); value != "" {
		return value
	}
	return envconfig.Var(native)
}

// NormalizeGPUExecution makes effective enabled policy part of runner identity.
// Ordinary requests keep their nil defaults for compatibility with old options.
func NormalizeGPUExecution(opts Options) (Options, GPUExecutionConfig, error) {
	cfg, err := ResolveGPUExecution(opts)
	if err == nil && cfg.KVCacheInitialTokens > 0 {
		opts.KVCacheInitialTokens = &cfg.KVCacheInitialTokens
	}
	if err == nil && cfg.Runtime {
		opts.MOERuntime = &cfg.Runtime
	}
	if err == nil && cfg.Exclusive {
		opts.GPUExclusive = &cfg.Exclusive
	}
	if err == nil && (cfg.Exclusive || cfg.VRAMLimitMiB > 0) {
		opts.GPUVRAMLimitMiB = &cfg.VRAMLimitMiB
	}
	if err == nil && cfg.Diagnostics {
		opts.GPUDiagnostics = &cfg.Diagnostics
	}
	return opts, cfg, err
}
