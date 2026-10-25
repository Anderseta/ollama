package api

import (
	"encoding/json"
	"math"
	"testing"
)

func clearGPUExecutionEnvironment(t *testing.T) {
	t.Helper()
	for _, key := range []string{"OLLAMA_GPU_EXCLUSIVE", "LLAMA_ARG_GPU_EXCLUSIVE", "OLLAMA_GPU_VRAM_LIMIT_MIB", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB", "OLLAMA_GPU_DIAGNOSTICS", "LLAMA_ARG_GPU_DIAGNOSTICS"} {
		t.Setenv(key, "")
	}
}

func TestGPUExecutionPrecedenceAndRunnerIdentity(t *testing.T) {
	clearGPUExecutionEnvironment(t)
	opts := DefaultOptions()
	resolved, cfg, err := NormalizeGPUExecution(opts)
	if err != nil || cfg.Exclusive || cfg.Diagnostics || resolved.GPUExclusive != nil || resolved.GPUVRAMLimitMiB != nil || resolved.GPUDiagnostics != nil {
		t.Fatalf("normal defaults changed: %+v, %v", cfg, err)
	}
	t.Setenv("OLLAMA_GPU_EXCLUSIVE", "true")
	t.Setenv("OLLAMA_GPU_VRAM_LIMIT_MIB", "12288")
	t.Setenv("OLLAMA_GPU_DIAGNOSTICS", "1")
	resolved, cfg, err = NormalizeGPUExecution(opts)
	if err != nil || !cfg.Exclusive || !cfg.Diagnostics || cfg.VRAMLimitMiB != 12288 || resolved.GPUExclusive == nil || resolved.GPUDiagnostics == nil {
		t.Fatalf("server policy missing from runner identity: %+v, %v", cfg, err)
	}
	opts.RunnerEnv = map[string]string{"LLAMA_ARG_GPU_EXCLUSIVE": "0", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB": "4096", "LLAMA_ARG_GPU_DIAGNOSTICS": "0"}
	_, cfg, err = NormalizeGPUExecution(opts)
	if err != nil || cfg.Exclusive || cfg.Diagnostics || cfg.VRAMLimitMiB != 4096 {
		t.Fatalf("runner overrides lost: %+v, %v", cfg, err)
	}
	enabled, disabled, zero := true, false, 0
	opts.GPUExclusive, opts.GPUDiagnostics, opts.GPUVRAMLimitMiB = &enabled, &disabled, &zero
	_, cfg, err = NormalizeGPUExecution(opts)
	if err != nil || !cfg.Exclusive || cfg.Diagnostics || cfg.VRAMLimitMiB != 0 {
		t.Fatalf("explicit API override lost: %+v, %v", cfg, err)
	}
	opts.GPUExclusive = &disabled
	_, cfg, err = NormalizeGPUExecution(opts)
	if err != nil || cfg.Exclusive {
		t.Fatal("explicit false did not disable server mode")
	}
	data, err := json.Marshal(opts)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Options
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.GPUExclusive == nil || *decoded.GPUExclusive || decoded.GPUVRAMLimitMiB == nil || *decoded.GPUVRAMLimitMiB != 0 {
		t.Fatal("false/zero lost during JSON round trip")
	}
}

func TestGPUExecutionValidation(t *testing.T) {
	clearGPUExecutionEnvironment(t)
	for _, value := range []any{float64(-1), 0.5, math.NaN(), math.Inf(1), float64(2097152), "12GiB"} {
		opts := DefaultOptions()
		if err := opts.FromMap(map[string]any{"gpu_vram_limit_mib": value}); err == nil {
			t.Errorf("accepted VRAM budget %v", value)
		}
	}
	for _, key := range []string{"gpu_exclusive", "gpu_diagnostics"} {
		opts := DefaultOptions()
		if err := opts.FromMap(map[string]any{key: "true"}); err == nil {
			t.Errorf("accepted nonboolean %s", key)
		}
	}
	t.Setenv("OLLAMA_GPU_EXCLUSIVE", "invalid")
	if _, err := ResolveGPUExecution(DefaultOptions()); err == nil {
		t.Fatal("invalid mode silently disabled")
	}
	opts := DefaultOptions()
	opts.RunnerEnv = map[string]string{"LLAMA_ARG_GPU_EXCLUSIVE": "", "LLAMA_ARG_GPU_DIAGNOSTICS": "", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB": ""}
	if cfg, err := ResolveGPUExecution(opts); err != nil || cfg.Exclusive || cfg.Diagnostics || cfg.VRAMLimitMiB != 0 {
		t.Fatalf("cleared runner values did not remove inherited policy: %+v %v", cfg, err)
	}
	if _, err := ParseRunnerEnvironment(map[string]string{"LLAMA_ARG_GPU_EXCLUSIVE": "1", "LLAMA_ARG_GPU_DIAGNOSTICS": "0", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB": "12288"}); err != nil {
		t.Fatal(err)
	}
}
