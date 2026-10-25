package llm

import (
	"os/exec"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

func TestKVCacheInitialChildEnvironmentOverridesInheritance(t *testing.T) {
	for _, tokens := range []int{0, 4096} {
		t.Setenv("WILDLLAMA_KV_CACHE_INITIAL_TOKENS", "8192")
		environment := map[string]string{"WILDLLAMA_KV_CACHE_INITIAL_TOKENS": "8192"}
		setKVCacheInitialEnvironment(environment, tokens)
		command := &exec.Cmd{}
		SetupLlamaServerCommandEnv(command, "llama-server", nil, environment)
		found := 0
		for _, entry := range command.Env {
			key, value, _ := strings.Cut(entry, "=")
			if strings.EqualFold(key, "WILDLLAMA_KV_CACHE_INITIAL_TOKENS") {
				found++
				want := "0"
				if tokens > 0 {
					want = "4096"
				}
				if value != want {
					t.Fatalf("child inherited contradictory KV commitment: %s", entry)
				}
			}
		}
		if found != 1 {
			t.Fatal("native KV policy override duplicated or missing")
		}
	}
}

func TestKVCacheInitialEstimateRetainsNominalAdmission(t *testing.T) {
	clearEstimateCacheEnvironment(t)
	t.Setenv("OLLAMA_KV_CACHE_INITIAL_TOKENS", "0")
	f := loadTestGGUF(t, qwen35MemoryMetadata())
	opts := api.DefaultOptions()
	opts.NumCtx = 131072
	opts.RunnerEnv = map[string]string{"OLLAMA_KV_CACHE_TYPE": "q8_0"}
	base := EstimateServerVRAM("", f, opts, 1)
	for _, setting := range []struct {
		tokens int
		mib    uint64
	}{{4096, 68}, {8192, 136}, {131072, 2176}, {262144, 2176}, {0, 2176}} {
		opts.KVCacheInitialTokens = &setting.tokens
		e := EstimateServerVRAM("", f, opts, 1)
		if !e.Valid || e.InitialKV != setting.mib<<20 || e.KV != base.KV || e.WorstCaseKV != base.WorstCaseKV || e.Total() != base.Total() {
			t.Fatalf("unsafe optimistic KV admission or wrong initial commitment: %+v", e)
		}
		if opts.NumCtx != 131072 {
			t.Fatal("physical KV control changed logical context")
		}
	}
}

func TestKVCacheInitialLoadObservationIsNotLifetimeBound(t *testing.T) {
	t.Setenv("OLLAMA_KV_CACHE_INITIAL_TOKENS", "0")
	model := loadTestGGUF(t, qwen35MemoryMetadata())
	opts := api.DefaultOptions()
	opts.NumCtx = 131072
	initial := 4096
	opts.KVCacheInitialTokens = &initial
	devices := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA", ID: "growing-kv"}, TotalMemory: 16 << 30}}
	runner := &llamaServerRunner{modelPath: "growing-kv-fixture", metadata: model, memGPU: 1 << 30, totalLayers: 33, gpuLayers: 33,
		launch: llamaServerLaunchConfig{numParallel: 1, opts: opts, gpus: devices}}
	runner.rememberPlacement()
	if _, ok := ObservedServerVRAM(runner.modelPath, model, opts, devices, 1, LlamaServerConfig{}); ok {
		t.Fatal("initial partial KV observation was reused as a lifetime allocation bound")
	}
}
