package llm

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

func TestObservedPlacementUsesExactRunnerAndDeviceSignature(t *testing.T) {
	model := loadTestGGUF(t, map[string]any{"general.architecture": "llama"})
	opts := api.DefaultOptions()
	opts.NumCtx = 4096
	devices := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA", ID: "fixture"}, TotalMemory: 16 << 30}}
	runner := &llamaServerRunner{modelPath: "fixture-observed", metadata: model, memGPU: 4 << 30, totalLayers: 8, gpuLayers: 8,
		launch: llamaServerLaunchConfig{numParallel: 1, opts: opts, gpus: devices}}
	runner.rememberPlacement()
	bytes, ok := ObservedServerVRAM(runner.modelPath, model, opts, devices, 1, LlamaServerConfig{})
	if !ok || bytes <= 4<<30 {
		t.Fatal("observed scratch/allocator headroom was not retained")
	}
	opts.NumUbatch = 128
	if _, ok := ObservedServerVRAM(runner.modelPath, model, opts, devices, 1, LlamaServerConfig{}); ok {
		t.Fatal("microbatch changed without invalidating observation")
	}
	opts.NumUbatch = 0
	devices[0].DriverMinor++
	if _, ok := ObservedServerVRAM(runner.modelPath, model, opts, devices, 1, LlamaServerConfig{}); ok {
		t.Fatal("driver changed without invalidating observation")
	}
}
