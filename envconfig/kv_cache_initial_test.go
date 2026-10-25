package envconfig

import "testing"

func TestKVCacheInitialTokensEnvironmentDefaultsAndRegistration(t *testing.T) {
	t.Setenv("OLLAMA_KV_CACHE_INITIAL_TOKENS", "")
	if KVCacheInitialTokens() != 0 {
		t.Fatal("experimental physical KV commitment is not default-off")
	}
	t.Setenv("OLLAMA_KV_CACHE_INITIAL_TOKENS", "4096")
	if KVCacheInitialTokens() != 4096 {
		t.Fatal("server environment did not enable requested initial commitment")
	}
	if setting, ok := AsMap()["OLLAMA_KV_CACHE_INITIAL_TOKENS"]; !ok || setting.Value != "4096" {
		t.Fatal("server environment report omitted KV commitment")
	}
}
