package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/gguf"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func TestSchedQwen35CacheTypePreflightCoexistence(t *testing.T) {
	for _, item := range []struct {
		kind  string
		evict bool
	}{{"q8_0", false}, {"f16", true}} {
		t.Run(item.kind, func(t *testing.T) {
			t.Setenv("OLLAMA_NUM_PARALLEL", "1")
			t.Setenv("OLLAMA_KV_CACHE_TYPE", "f16") // per-runner q8 must win
			ctx, cancel := context.WithTimeout(t.Context(), time.Second)
			defer cancel()
			p, _ := createBinFile(t, gguftest.KV{
				"general.architecture": "qwen35", "qwen35.block_count": uint32(32),
				"qwen35.context_length": uint32(262144), "qwen35.embedding_length": uint32(4096),
				"qwen35.attention.head_count": uint32(16), "qwen35.attention.head_count_kv": uint32(4),
				"qwen35.attention.key_length": uint32(256), "qwen35.attention.value_length": uint32(256),
				"qwen35.ssm.inner_size": uint32(6144), "qwen35.ssm.state_size": uint32(128),
				"qwen35.ssm.conv_kernel": uint32(4), "qwen35.ssm.group_count": uint32(16),
			}, nil)
			scenario := newScenarioRequest(t, ctx, "hybrid-preflight", 100*format.MebiByte, nil, nil)
			scenario.req.model.ModelPath = p
			scenario.req.model.Config.ModelFamily = "qwen35"
			scenario.req.opts.NumCtx = 131072
			scenario.req.opts.RunnerEnv = map[string]string{"OLLAMA_KV_CACHE_TYPE": item.kind}
			s := InitScheduler(ctx)
			s.loaded["existing-rag"] = &runnerRef{llama: &mockLlm{modelPath: "rag"}, modelKey: "existing-rag"}
			gpus := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Metal", ID: "fixture"}, TotalMemory: 8 << 30, FreeMemory: 4 << 30}}
			called := false
			s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, _ string, _ *gguf.Model, _, _ []string, _ api.Options, _ int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
				called = true
				return scenario.srv, nil
			}
			needEvict := s.load(scenario.req, getSystemInfoFn(), gpus, true)
			if needEvict != item.evict || called == item.evict {
				t.Fatalf("cache %s preflight eviction=%v spawned=%v; want eviction=%v", item.kind, needEvict, called, item.evict)
			}
		})
	}
}

func TestMemoryPredictionArithmeticDoesNotWrap(t *testing.T) {
	if saturatingMemorySum(^uint64(0)-5, 10) != ^uint64(0) {
		t.Fatal("overflowed memory budget was admitted as a tiny allocation")
	}
	if effectiveLlamaServerContext(int(^uint(0)>>1), nil, 64) != int(^uint(0)>>1) {
		t.Fatal("parallel context overflowed")
	}
}

func TestKVCacheInitialChangeRequiresRunnerReload(t *testing.T) {
	t.Setenv("OLLAMA_KV_CACHE_INITIAL_TOKENS", "4096")
	opts, _, err := api.NormalizeGPUExecution(api.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{}
	runner := &runnerRef{Options: &opts, llama: &mockLlm{}, numParallel: 1}
	runner.model.Store(m)
	req := &LlmRequest{model: m, opts: opts}
	if runner.needsReload(t.Context(), req) {
		t.Fatal("unchanged physical KV policy unnecessarily reloaded")
	}
	zero := 0
	req.opts.KVCacheInitialTokens = &zero
	if !runner.needsReload(t.Context(), req) {
		t.Fatal("disabled KV growth reused a runner initialized with partial commitment")
	}
}
