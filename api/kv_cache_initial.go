package api

import (
	"fmt"
	"math"
	"strconv"

	"github.com/ollama/ollama/envconfig"
)

const MaxKVCacheInitialTokens = 2097152

func validKVCacheInitialTokens(n int64) bool {
	return n == 0 || (n >= 256 && n <= MaxKVCacheInitialTokens)
}

// ResolveKVCacheInitialTokens is an opt-in physical commitment target, never a
// smaller logical context. Explicit API zero overrides the server default.
// Native-only inherited settings are ignored so they cannot contradict policy.
func ResolveKVCacheInitialTokens(opts Options) (int, error) {
	var n int64
	if opts.KVCacheInitialTokens != nil {
		n = int64(*opts.KVCacheInitialTokens)
	} else if raw := envconfig.Var("OLLAMA_KV_CACHE_INITIAL_TOKENS"); raw != "" {
		value, err := strconv.ParseInt(raw, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("kv_cache_initial_tokens must be 0 or an integer between 256 and %d", MaxKVCacheInitialTokens)
		}
		n = value
	}
	if !validKVCacheInitialTokens(n) {
		return 0, fmt.Errorf("kv_cache_initial_tokens must be 0 or an integer between 256 and %d", MaxKVCacheInitialTokens)
	}
	return int(n), nil
}

func parseKVCacheInitialTokensOption(value any) (int, error) {
	var n float64
	switch v := value.(type) {
	case int64:
		n = float64(v)
	case float64:
		n = v
	default:
		return 0, fmt.Errorf("kv_cache_initial_tokens must be an integer")
	}
	if math.IsNaN(n) || math.IsInf(n, 0) || n != math.Trunc(n) || n < 0 || n > MaxKVCacheInitialTokens || !validKVCacheInitialTokens(int64(n)) {
		return 0, fmt.Errorf("kv_cache_initial_tokens must be 0 or an integer between 256 and %d", MaxKVCacheInitialTokens)
	}
	return int(n), nil
}
