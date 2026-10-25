package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

func TestGPUExclusiveSchedulingPreservesContextAndMmap(t *testing.T) {
	opts := api.DefaultOptions()
	enabled := true
	opts.GPUExclusive = &enabled
	opts.NumCtx = 131072
	req := &LlmRequest{opts: opts, numCtxAuto: true}
	if _, _, _, _, _, retry := req.reduceAutoNumCtxForLoadOOM(nil, 1, true, ml.SystemInfo{}, nil, opts); retry {
		t.Fatal("GPU-exclusive OOM reduced context")
	}
	if req.opts.NumCtx != 131072 {
		t.Fatal("context changed")
	}
	s := &Scheduler{}
	got := s.applyLlamaServerMmapDefaults(req, opts, ml.SystemInfo{}, nil, nil, 1)
	if got.UseMMap == nil || !*got.UseMMap {
		t.Fatal("strict GPU mode disabled demand-paged weights")
	}
	if req.opts.NumCtx != 131072 {
		t.Fatal("mmap setup changed context")
	}
}

func TestGPUExclusiveDiscoveryRejectsCPUWithoutLaunching(t *testing.T) {
	opts := api.DefaultOptions()
	enabled := true
	opts.GPUExclusive = &enabled
	request := &LlmRequest{model: &Model{}, opts: opts, errCh: make(chan error, 1)}
	if (&Scheduler{}).load(request, ml.SystemInfo{}, nil, false) {
		t.Fatal("requested retry instead of rejecting missing CUDA")
	}
	select {
	case err := <-request.errCh:
		if err == nil {
			t.Fatal("missing CUDA was not an error")
		}
	default:
		t.Fatal("no rejection was delivered")
	}
}
