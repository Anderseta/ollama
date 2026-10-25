package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

func TestMOERuntimeChangeRequiresRunnerReload(t *testing.T) {
	for _, key := range []string{"OLLAMA_GPU_EXCLUSIVE", "LLAMA_ARG_GPU_EXCLUSIVE", "OLLAMA_GPU_DIAGNOSTICS", "LLAMA_ARG_GPU_DIAGNOSTICS", "OLLAMA_GPU_VRAM_LIMIT_MIB", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB", "OLLAMA_MOE_RUNTIME"} {
		t.Setenv(key, "")
	}
	t.Setenv("OLLAMA_MOE_RUNTIME", "true")
	options, _, err := api.NormalizeGPUExecution(api.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	model := &Model{}
	runner := &runnerRef{Options: &options, llama: &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}, numParallel: 1}
	runner.model.Store(model)
	request := &LlmRequest{model: model, opts: options}
	if runner.needsReload(t.Context(), request) {
		t.Fatal("unchanged runtime unnecessarily reloaded the runner")
	}
	disabled := false
	request.opts.MOERuntime = &disabled
	if !runner.needsReload(t.Context(), request) {
		t.Fatal("explicit runtime false reused enabled native runner")
	}
}
