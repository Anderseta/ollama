package api

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestMOERuntimePrecedenceAndIdentity(t *testing.T) {
	clearGPUExecutionEnvironment(t)
	t.Setenv("WILDLLAMA_MOE_RUNTIME", "1")
	baseline, cfg, err := NormalizeGPUExecution(DefaultOptions())
	if err != nil || cfg.Runtime || baseline.MOERuntime != nil {
		t.Fatalf("native inheritance enabled default-off runtime: %+v %v", cfg, err)
	}
	t.Setenv("OLLAMA_MOE_RUNTIME", "true")
	enabled, cfg, err := NormalizeGPUExecution(DefaultOptions())
	if err != nil || !cfg.Runtime || enabled.MOERuntime == nil || !*enabled.MOERuntime || reflect.DeepEqual(baseline.Runner, enabled.Runner) {
		t.Fatalf("enabled runtime missing from runner identity: %+v %v", cfg, err)
	}
	for _, raw := range []string{"0", ""} {
		opts := DefaultOptions()
		opts.RunnerEnv = map[string]string{"WILDLLAMA_MOE_RUNTIME": raw}
		_, cfg, err = NormalizeGPUExecution(opts)
		if err != nil || cfg.Runtime {
			t.Fatalf("per-runner disable %q lost: %+v %v", raw, cfg, err)
		}
	}
	disabled := false
	enabled.MOERuntime = &disabled
	enabled.RunnerEnv = map[string]string{"WILDLLAMA_MOE_RUNTIME": "1"}
	normalized, cfg, err := NormalizeGPUExecution(enabled)
	if err != nil || cfg.Runtime || normalized.MOERuntime == nil || *normalized.MOERuntime {
		t.Fatalf("explicit API false lost: %+v %v", cfg, err)
	}
	t.Setenv("OLLAMA_MOE_RUNTIME", "false")
	opts := DefaultOptions()
	opts.RunnerEnv = map[string]string{"WILDLLAMA_MOE_RUNTIME": "1"}
	if cfg, err := ResolveGPUExecution(opts); err != nil || !cfg.Runtime {
		t.Fatalf("runner enable lost to server false: %+v %v", cfg, err)
	}
	value := true
	opts.MOERuntime = &value
	opts.RunnerEnv["WILDLLAMA_MOE_RUNTIME"] = "0"
	if cfg, err := ResolveGPUExecution(opts); err != nil || !cfg.Runtime {
		t.Fatalf("explicit API true lost to runner false: %+v %v", cfg, err)
	}
}

func TestMOERuntimeJSONAndValidation(t *testing.T) {
	clearGPUExecutionEnvironment(t)
	opts := DefaultOptions()
	encoded, err := json.Marshal(opts)
	if err != nil || strings.Contains(string(encoded), `"moe_runtime"`) {
		t.Fatalf("nil runtime changed ordinary JSON: %s %v", encoded, err)
	}
	if err := opts.FromMap(map[string]any{"moe_runtime": false}); err != nil {
		t.Fatal(err)
	}
	encoded, err = json.Marshal(opts)
	var decoded Options
	if err != nil || json.Unmarshal(encoded, &decoded) != nil || decoded.MOERuntime == nil || *decoded.MOERuntime {
		t.Fatalf("explicit false lost in JSON round trip: %s %v", encoded, err)
	}
	for _, invalid := range []any{"true", float64(1), map[string]any{}} {
		value := DefaultOptions()
		if err := value.FromMap(map[string]any{"moe_runtime": invalid}); err == nil {
			t.Fatalf("accepted nonboolean runtime %v", invalid)
		}
	}
	for _, raw := range []string{"", "0", "1"} {
		if _, err := ParseRunnerEnvironment(map[string]string{"WILDLLAMA_MOE_RUNTIME": raw}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := ParseRunnerEnvironment(map[string]string{"WILDLLAMA_MOE_RUNTIME": "true"}); err == nil {
		t.Fatal("runner runtime accepted an unsupported native value")
	}
	t.Setenv("OLLAMA_MOE_RUNTIME", "invalid")
	if _, err := ResolveGPUExecution(DefaultOptions()); err == nil {
		t.Fatal("invalid server runtime silently disabled")
	}
}
