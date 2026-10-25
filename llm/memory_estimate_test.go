package llm

import (
	"math"
	"testing"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
)

func qwen35MemoryMetadata() gguftest.KV {
	return gguftest.KV{
		"general.architecture": "qwen35", "qwen35.block_count": uint32(32),
		"qwen35.context_length": uint32(262144), "qwen35.embedding_length": uint32(4096),
		"qwen35.attention.head_count": uint32(16), "qwen35.attention.head_count_kv": uint32(4),
		"qwen35.attention.key_length": uint32(256), "qwen35.attention.value_length": uint32(256),
		"qwen35.full_attention_interval": uint32(4), "qwen35.ssm.conv_kernel": uint32(4),
		"qwen35.ssm.inner_size": uint32(6144), "qwen35.ssm.state_size": uint32(128),
		"qwen35.ssm.group_count": uint32(16),
	}
}

func clearEstimateCacheEnvironment(t *testing.T) {
	t.Helper()
	for _, key := range []string{"OLLAMA_KV_CACHE_TYPE", "LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V", "OLLAMA_KV_CACHE_INITIAL_TOKENS"} {
		t.Setenv(key, "")
	}
}

func TestEstimateServerVRAMQwen35Hybrid(t *testing.T) {
	clearEstimateCacheEnvironment(t)
	f := loadTestGGUF(t, qwen35MemoryMetadata())
	opts := api.DefaultOptions()
	opts.NumCtx = 131072
	for _, item := range []struct {
		kind string
		kv   uint64
	}{{"q8_0", 2176 << 20}, {"f16", 4 << 30}, {"bf16", 4 << 30}} {
		t.Run(item.kind, func(t *testing.T) {
			opts.RunnerEnv = map[string]string{"OLLAMA_KV_CACHE_TYPE": item.kind}
			e := EstimateServerVRAM("", f, opts, 1)
			if !e.Valid || e.KV != item.kv || e.WorstCaseKV != item.kv || e.InitialKV != item.kv {
				t.Fatalf("KV estimate = %+v, want %d", e, item.kv)
			}
			// Matches llama_hparams::n_embd_r/s, F32 states, 24 GDN layers.
			wantRS := uint64(4 * (3*(6144+2*16*128) + 128*6144) * 24)
			if e.Recurrent != wantRS || e.Auxiliary < 256<<20 || e.Total() < e.Weights+e.KV+e.Recurrent+e.Auxiliary {
				t.Fatalf("recurrent/auxiliary buffers omitted: %+v", e)
			}
			two := EstimateServerVRAM("", f, opts, 2)
			if two.KV != 2*e.KV || two.Recurrent != 2*e.Recurrent || two.Weights != e.Weights {
				t.Fatalf("parallel cache sizing wrong: one=%+v two=%+v", e, two)
			}
		})
	}
	flags := make([]bool, 32)
	for i := range flags {
		flags[i] = (i+1)%4 != 0
	}
	metadata := qwen35MemoryMetadata()
	metadata["qwen35.attention.recurrent_layers"] = flags
	metadata["qwen35.full_attention_interval"] = uint32(1) // explicit flags win
	flagged := EstimateServerVRAM("", loadTestGGUF(t, metadata), opts, 1)
	if !flagged.Valid || flagged.KV != 4<<30 {
		t.Fatal("per-layer recurrence flags did not override the default interval")
	}
}

func TestEstimateServerVRAMTransformerAndHeadDimensions(t *testing.T) {
	clearEstimateCacheEnvironment(t)
	base := gguftest.KV{
		"general.architecture": "llama", "llama.block_count": uint32(2),
		"llama.context_length": uint32(8192), "llama.embedding_length": uint32(512),
		"llama.attention.head_count": []uint32{8, 4}, "llama.attention.head_count_kv": []uint32{2, 1},
	}
	opts := api.DefaultOptions()
	opts.NumCtx = 4096
	e := EstimateServerVRAM("", loadTestGGUF(t, base), opts, 1)
	if !e.Valid || e.KV != 4<<20 || e.Recurrent != 0 {
		t.Fatalf("per-layer heads were collapsed to a min/max approximation: %+v", e)
	}
	base["llama.block_count"] = uint32(1)
	base["llama.attention.head_count"] = uint32(8)
	base["llama.attention.head_count_kv"] = uint32(2)
	base["llama.attention.key_length"] = uint32(128)
	base["llama.attention.value_length"] = uint32(64)
	opts.RunnerEnv = map[string]string{"LLAMA_ARG_CACHE_TYPE_K": "q8_0", "LLAMA_ARG_CACHE_TYPE_V": "q4_0"}
	e = EstimateServerVRAM("", loadTestGGUF(t, base), opts, 1)
	if !e.Valid || e.KV != (272+72)*4096 {
		t.Fatalf("explicit K/V dimensions, formats or block metadata ignored: %+v", e)
	}
	delete(base, "llama.attention.head_count_kv") // native default is MHA, not one KV head
	opts.RunnerEnv = nil
	e = EstimateServerVRAM("", loadTestGGUF(t, base), opts, 1)
	if !e.Valid || e.KV != (8*128*2+8*64*2)*4096 {
		t.Fatalf("missing GQA did not use attention head count: %+v", e)
	}
}

func TestEstimateServerVRAMCachePrecedenceAndContext(t *testing.T) {
	clearEstimateCacheEnvironment(t)
	f := loadTestGGUF(t, qwen35MemoryMetadata())
	opts := api.DefaultOptions()
	opts.NumCtx = 131072
	t.Setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
	t.Setenv("LLAMA_ARG_CACHE_TYPE_K", "q4_0")
	t.Setenv("LLAMA_ARG_CACHE_TYPE_V", "q4_0")
	if e := EstimateServerVRAM("", f, opts, 1); e.KV != 2176<<20 {
		t.Fatal("common CLI cache type lost to separate inherited native environment")
	}
	opts.RunnerEnv = map[string]string{"OLLAMA_KV_CACHE_TYPE": "", "LLAMA_ARG_CACHE_TYPE_K": "f16", "LLAMA_ARG_CACHE_TYPE_V": ""}
	if e := EstimateServerVRAM("", f, opts, 1); e.KV != 4<<30 {
		t.Fatalf("cleared runner overrides did not remove inheritance: %+v", e)
	}
	opts.RunnerEnv = map[string]string{"OLLAMA_KV_CACHE_TYPE": "q8_0"}
	opts.NumCtx = 524288
	if e := EstimateServerVRAM("", f, opts, 1); e.KV != 4352<<20 {
		t.Fatal("estimate did not match the training-context clamp")
	}
	opts.NumCtx = 0
	if e := EstimateServerVRAM("", f, opts, 1); e.KV != 4352<<20 {
		t.Fatal("native -c 0 training-context default was not estimated")
	}
}

func TestEstimateServerVRAMInvalidAndOverflow(t *testing.T) {
	clearEstimateCacheEnvironment(t)
	opts := api.DefaultOptions()
	opts.NumCtx = 131072
	for _, change := range []func(gguftest.KV){
		func(k gguftest.KV) { k["qwen35.block_count"] = uint64(math.MaxUint64) },
		func(k gguftest.KV) { k["qwen35.full_attention_interval"] = uint32(0) },
		func(k gguftest.KV) { k["qwen35.attention.head_count"] = int32(-1) },
		func(k gguftest.KV) { k["qwen35.attention.head_count_kv"] = []uint32{4} },
		func(k gguftest.KV) { k["qwen35.attention.key_length"] = uint64(math.MaxUint64) },
		func(k gguftest.KV) { k["qwen35.ssm.inner_size"] = uint64(math.MaxUint64) },
		func(k gguftest.KV) { k["qwen35.attention.recurrent_layers"] = []bool{true} },
	} {
		metadata := qwen35MemoryMetadata()
		change(metadata)
		e := EstimateServerVRAM("", loadTestGGUF(t, metadata), opts, 1)
		if e.Valid || e.Total() != math.MaxUint64 {
			t.Fatalf("invalid/overflowed estimate could be admitted as a fit: %+v", e)
		}
	}
	f := loadTestGGUF(t, qwen35MemoryMetadata())
	for _, dimensions := range [][2]int{{-1, 1}, {131072, 0}, {math.MaxInt, math.MaxInt}} {
		opts.NumCtx = dimensions[0]
		if e := EstimateServerVRAM("", f, opts, dimensions[1]); e.Valid || e.Total() != math.MaxUint64 {
			t.Fatalf("invalid context/parallel %v wrapped into a fit: %+v", dimensions, e)
		}
	}
	opts.NumCtx = 4096
	opts.RunnerEnv = map[string]string{"OLLAMA_KV_CACHE_TYPE": "unknown"}
	if EstimateServerVRAM("", f, opts, 1).Total() != math.MaxUint64 {
		t.Fatal("unknown cache type was treated as a smaller supported precision")
	}
	if vramAdd(math.MaxUint64-5, 10) != math.MaxUint64 || vramMul(math.MaxUint64, 2) != math.MaxUint64 {
		t.Fatal("budget arithmetic did not saturate")
	}
}
