package api

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestKVCacheInitialTokensPrecedenceAndIdentity(t *testing.T) {
	clearGPUExecutionEnvironment(t)
	t.Setenv("OLLAMA_KV_CACHE_INITIAL_TOKENS", "")
	t.Setenv("WILDLLAMA_KV_CACHE_INITIAL_TOKENS", "8192")
	baseline, cfg, err := NormalizeGPUExecution(DefaultOptions())
	if err != nil || cfg.KVCacheInitialTokens != 0 || baseline.KVCacheInitialTokens != nil {
		t.Fatalf("native inheritance enabled experimental KV commitment: %+v %v", cfg, err)
	}
	t.Setenv("OLLAMA_KV_CACHE_INITIAL_TOKENS", "4096")
	enabled, cfg, err := NormalizeGPUExecution(DefaultOptions())
	if err != nil || cfg.KVCacheInitialTokens != 4096 || enabled.KVCacheInitialTokens == nil || reflect.DeepEqual(baseline.Runner, enabled.Runner) {
		t.Fatalf("server option missing from runner identity: %+v %v", cfg, err)
	}
	zero := 0
	enabled.KVCacheInitialTokens = &zero
	disabled, cfg, err := NormalizeGPUExecution(enabled)
	if err != nil || cfg.KVCacheInitialTokens != 0 || disabled.KVCacheInitialTokens == nil || *disabled.KVCacheInitialTokens != 0 {
		t.Fatalf("explicit API zero lost to inherited enable: %+v %v", cfg, err)
	}
	t.Setenv("OLLAMA_KV_CACHE_INITIAL_TOKENS", "invalid")
	if _, err := ResolveKVCacheInitialTokens(DefaultOptions()); err == nil {
		t.Fatal("invalid server setting silently disabled")
	}
	if got, err := ResolveKVCacheInitialTokens(disabled); err != nil || got != 0 {
		t.Fatal("invalid server setting overrode explicit API zero")
	}
}

func TestKVCacheInitialTokensOptionValidation(t *testing.T) {
	for _, n := range []int64{0, 256, 4096, MaxKVCacheInitialTokens} {
		opts := DefaultOptions()
		if err := opts.FromMap(map[string]any{"kv_cache_initial_tokens": n}); err != nil || opts.KVCacheInitialTokens == nil || *opts.KVCacheInitialTokens != int(n) {
			t.Fatalf("valid value %d rejected: %v", n, err)
		}
		encoded, err := json.Marshal(opts)
		var decoded Options
		if err != nil || json.Unmarshal(encoded, &decoded) != nil || decoded.KVCacheInitialTokens == nil || *decoded.KVCacheInitialTokens != int(n) {
			t.Fatalf("value %d lost in JSON round-trip: %s", n, encoded)
		}
	}
	for _, invalid := range []any{int64(-1), int64(1), int64(255), int64(MaxKVCacheInitialTokens + 1), 4096.5, math.NaN(), math.Inf(1), "4096", true} {
		opts := DefaultOptions()
		if err := opts.FromMap(map[string]any{"kv_cache_initial_tokens": invalid}); err == nil {
			t.Fatalf("invalid commitment %v accepted", invalid)
		}
	}
	for _, n := range []int{-1, 255, MaxKVCacheInitialTokens + 1} {
		opts := DefaultOptions()
		opts.KVCacheInitialTokens = &n
		if _, err := ResolveKVCacheInitialTokens(opts); err == nil {
			t.Fatalf("direct API pointer bypassed range checks: %d", n)
		}
	}
	encoded, _ := json.Marshal(DefaultOptions())
	if strings.Contains(string(encoded), "kv_cache_initial_tokens") {
		t.Fatal("nil setting changed legacy JSON")
	}
}
