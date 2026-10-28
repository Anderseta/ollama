package llm

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestEstimateServerVRAMQwen35MTPHead(t *testing.T) {
	clearEstimateCacheEnvironment(t)
	t.Setenv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K", "")
	t.Setenv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V", "")
	metadata := qwen35MemoryMetadata()
	metadata["qwen35.block_count"] = uint32(33)
	metadata["qwen35.nextn_predict_layers"] = uint32(1)
	opts := api.DefaultOptions()
	opts.NumCtx = 131072
	opts.DraftNumPredict = 2
	opts.SpeculativeType = "draft-mtp"
	opts.RunnerEnv = map[string]string{"OLLAMA_KV_CACHE_TYPE": "q8_0"}
	model := loadTestGGUF(t, metadata)
	mtp := EstimateServerVRAM("", model, opts, 1)
	wantState := uint64(4 * (3*(6144+2*16*128) + 128*6144) * 24)
	if !mtp.Valid || mtp.KV != (2176+512)<<20 || mtp.Recurrent != 3*wantState {
		t.Fatalf("MTP head must add an F16 attention cache and only trunk rollback state: %+v", mtp)
	}
	for _, kind := range []string{"", "auto"} {
		opts.SpeculativeType = kind
		if got := EstimateServerVRAM("", model, opts, 1); got.KV != mtp.KV || got.Recurrent != mtp.Recurrent {
			t.Fatalf("embedded automatic MTP was not accounted: %+v", got)
		}
	}
	for _, kind := range []string{"none", "ngram-simple"} {
		opts.SpeculativeType = kind
		got := EstimateServerVRAM("", model, opts, 1)
		if !got.Valid || got.KV != 2176<<20 || got.Recurrent != wantState {
			t.Fatalf("disabled/non-MTP speculation incorrectly allocates the MTP head cache: %+v", got)
		}
	}
	opts.SpeculativeType = "draft-mtp"
	opts.KVCacheInitialTokens = new(int)
	*opts.KVCacheInitialTokens = 256
	growing := EstimateServerVRAM("", model, opts, 1)
	if !growing.Valid || growing.InitialKV != (512<<20)+(2176<<20)/512 || growing.WorstCaseKV != mtp.KV {
		t.Fatalf("draft context must retain full commitment while target grows: %+v", growing)
	}
	t.Setenv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K", "q8_0")
	t.Setenv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V", "q8_0")
	if got := EstimateServerVRAM("", model, opts, 1); !got.Valid || got.KV != (2176+272)<<20 {
		t.Fatalf("native draft cache overrides were ignored: %+v", got)
	}
	t.Setenv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V", "unknown")
	if EstimateServerVRAM("", model, opts, 1).Valid {
		t.Fatal("unknown draft cache precision must not be admitted")
	}
}

func TestEstimateServerVRAMRejectsInvalidMTPLayout(t *testing.T) {
	clearEstimateCacheEnvironment(t)
	metadata := qwen35MemoryMetadata()
	metadata["qwen35.nextn_predict_layers"] = uint32(32)
	opts := api.DefaultOptions()
	opts.NumCtx = 4096
	if EstimateServerVRAM("", loadTestGGUF(t, metadata), opts, 1).Valid {
		t.Fatal("a Qwen model cannot contain only MTP layers")
	}
}
