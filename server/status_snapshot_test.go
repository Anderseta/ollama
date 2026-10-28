package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/gguf"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
	"github.com/stretchr/testify/require"
)

type statusWaitingLlama struct {
	mockLlm
	waiting     chan struct{}
	ready       chan struct{}
	loadWaiting chan struct{}
	loadReady   chan struct{}
	unblockOnce sync.Once
	closed      atomic.Bool
	context     atomic.Int64
	total       atomic.Uint64
	vram        atomic.Uint64
	cancel      context.CancelFunc
}

func (s *statusWaitingLlama) WaitUntilRunning(ctx context.Context) error {
	close(s.waiting)
	select {
	case <-s.ready:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (s *statusWaitingLlama) Load(ctx context.Context, info ml.SystemInfo, gpus []ml.DeviceInfo, full bool) ([]ml.DeviceID, error) {
	if s.loadReady != nil {
		close(s.loadWaiting)
		select {
		case <-s.loadReady:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return s.mockLlm.Load(ctx, info, gpus, full)
}
func (s *statusWaitingLlama) unblock() {
	s.unblockOnce.Do(func() {
		close(s.ready)
		if s.loadReady != nil {
			close(s.loadReady)
		}
	})
}
func (s *statusWaitingLlama) MemorySize() (uint64, uint64) { return s.total.Load(), s.vram.Load() }
func (s *statusWaitingLlama) ContextLength() int           { return int(s.context.Load()) }
func (s *statusWaitingLlama) Close() error                 { s.closed.Store(true); return nil }

func loadingStatusFixture(t *testing.T, duringLoad ...bool) (*Scheduler, *statusWaitingLlama, *LlmRequest) {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_NUM_PARALLEL", "1")
	path, _ := createBinFile(t, gguftest.KV{
		"general.architecture": "llama", "llama.context_length": uint32(32),
		"llama.embedding_length": uint32(32), "llama.block_count": uint32(1),
		"llama.attention.head_count": uint32(1), "llama.attention.head_count_kv": uint32(1),
		"tokenizer.ggml.tokens": []string{" "}, "tokenizer.ggml.scores": []float32{0}, "tokenizer.ggml.token_type": []int32{0},
	}, []*gguftest.Tensor{{Name: "output.weight", Type: gguf.TensorTypeF32, Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 4))}})
	ctx, cancel := context.WithCancel(t.Context())
	s := InitScheduler(ctx)
	fake := &statusWaitingLlama{waiting: make(chan struct{}), ready: make(chan struct{})}
	fake.cancel = cancel
	waiting := fake.waiting
	if len(duringLoad) > 0 && duringLoad[0] {
		fake.loadWaiting, fake.loadReady = make(chan struct{}), make(chan struct{})
		waiting = fake.loadWaiting
	}
	fake.context.Store(32)
	fake.total.Store(4096)
	fake.vram.Store(1024)
	s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, path string, _ *gguf.Model, _, _ []string, _ api.Options, _ int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		fake.modelPath = path
		return fake, nil
	}
	opts := api.DefaultOptions()
	opts.NumCtx, opts.NumGPU = 32, 0
	req := &LlmRequest{ctx: ctx, opts: opts, model: &Model{ModelPath: path, ShortName: "status-fixture:latest", Name: "status-fixture:latest", Config: model.ConfigV2{ModelFamily: "llama", Capabilities: []string{"completion"}}}, successCh: make(chan *runnerRef, 1), errCh: make(chan error, 1), sessionDuration: &api.Duration{Duration: time.Hour}}
	loadDone := make(chan struct{})
	go func() {
		s.load(req, ml.SystemInfo{TotalMemory: 1 << 40, FreeMemory: 1 << 40}, nil, false)
		close(loadDone)
	}()
	select {
	case <-waiting:
	case <-time.After(time.Second):
		t.Fatal("fixture did not enter runner readiness wait")
	}
	t.Cleanup(func() {
		cancel()
		fake.unblock()
		select {
		case <-loadDone:
		case <-time.After(time.Second):
			t.Error("readiness waiter did not finish")
		}
	})
	return s, fake, req
}

func TestStatusSnapshotActiveLoadPromotesWithoutDuplicates(t *testing.T) {
	s, fake, req := loadingStatusFixture(t, true)
	loading := s.loadedModels()
	require.Len(t, loading, 1)
	require.True(t, loading[0].loading)
	require.Equal(t, "status-fixture:latest", loading[0].model.ShortName)
	fake.context.Store(64)
	fake.total.Store(8192)
	fake.vram.Store(2048)
	updated := s.loadedModels()
	require.Equal(t, 64, updated[0].contextLength)
	require.Equal(t, int64(8192), updated[0].size)
	require.Equal(t, int64(2048), updated[0].sizeVRAM)
	fake.unblock()
	select {
	case <-req.successCh:
	case err := <-req.errCh:
		t.Fatal(err)
	case <-time.After(time.Second):
		t.Fatal("model did not become ready")
	}
	ready := s.loadedModels()
	require.Len(t, ready, 1)
	require.False(t, ready[0].loading)
	require.Nil(t, s.loadedSnapshot.Load().loading)
}

func TestStatusSnapshotCancelledLoadDisappears(t *testing.T) {
	s, fake, req := loadingStatusFixture(t, true)
	require.Len(t, s.loadedModels(), 1)
	fake.cancel()
	select {
	case err := <-req.errCh:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("cancelled model did not finish")
	}
	require.True(t, fake.closed.Load())
	require.Empty(t, s.loadedModels())
}

func TestStatusSnapshotReplacementRemovesOldRunner(t *testing.T) {
	s, fake, req := loadingStatusFixture(t)
	fake.unblock()
	var first *runnerRef
	select {
	case first = <-req.successCh:
	case <-time.After(time.Second):
		t.Fatal("first runner did not become ready")
	}
	oldRegistry := s.loadedSnapshot.Load()
	replacement := &statusWaitingLlama{mockLlm: mockLlm{modelPath: req.model.ModelPath}, waiting: make(chan struct{}), ready: make(chan struct{})}
	replacement.total.Store(16384)
	replacement.context.Store(32)
	replacement.unblock()
	s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, _ string, _ *gguf.Model, _, _ []string, _ api.Options, _ int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		return replacement, nil
	}
	m := *req.model
	m.ShortName = "replacement:latest"
	next := &LlmRequest{ctx: req.ctx, opts: req.opts, model: &m, successCh: make(chan *runnerRef, 1), errCh: make(chan error, 1)}
	require.False(t, s.load(next, ml.SystemInfo{TotalMemory: 1 << 40, FreeMemory: 1 << 40}, nil, false))
	select {
	case <-next.successCh:
	case <-time.After(time.Second):
		t.Fatal("replacement runner did not become ready")
	}
	status := s.loadedModels()
	require.Len(t, status, 1)
	require.Equal(t, "replacement:latest", status[0].model.ShortName)
	require.Equal(t, int64(16384), status[0].size)
	require.True(t, fake.closed.Load())
	require.Nil(t, first.status.Load())
	require.Nil(t, oldRegistry.runners[0].model.Load())
}

func TestStatusSnapshotExpirationAndUnload(t *testing.T) {
	s, fake, req := loadingStatusFixture(t)
	fake.unblock()
	select {
	case <-req.successCh:
	case <-time.After(time.Second):
		t.Fatal("model did not become ready")
	}
	ctx, cancel := context.WithCancel(t.Context())
	finished := make(chan struct{})
	go func() { defer close(finished); s.processCompleted(ctx) }()
	defer func() { cancel(); <-finished }()
	fake.cancel()
	require.Eventually(t, func() bool {
		registry := s.loadedSnapshot.Load()
		return len(registry.runners) == 1 && !registry.runners[0].status.Load().expiresAt.IsZero()
	}, time.Second, time.Millisecond)
	expiration := s.loadedModels()[0].expiresAt
	require.WithinDuration(t, time.Now().Add(time.Hour), expiration, time.Second)
	s.expireRunner(req.model)
	require.Eventually(t, func() bool { return len(s.loadedModels()) == 0 }, time.Second, time.Millisecond)
	require.True(t, fake.closed.Load())
}

func TestStatusSnapshotConcurrentPollsDoNotWaitForScheduler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s, fake, _ := loadingStatusFixture(t, true)
	router := gin.New()
	server := &Server{sched: s}
	router.GET("/api/ps", server.PsHandler)
	httpServer := httptest.NewServer(router)
	defer func() { fake.unblock(); httpServer.Close() }()
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	const count = 32
	results := make(chan error, count)
	for range count {
		go func() {
			response, err := (&http.Client{Timeout: time.Second}).Get(httpServer.URL + "/api/ps")
			if err == nil {
				var status api.ProcessResponse
				err = json.NewDecoder(response.Body).Decode(&status)
				response.Body.Close()
				if err == nil && (len(status.Models) != 1 || !status.Models[0].Loading) {
					err = fmt.Errorf("incorrect loading status: %+v", status)
				}
			}
			results <- err
		}()
	}
	for range count {
		require.NoError(t, <-results)
	}
}

func TestPsHandlerResponsiveDuringRunnerReadiness(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, blocked := range []string{"runner", "scheduler"} {
		t.Run(blocked, func(t *testing.T) {
			s, fake, _ := loadingStatusFixture(t)
			router := gin.New()
			server := &Server{sched: s}
			router.GET("/api/ps", server.PsHandler)
			httpServer := httptest.NewServer(router)
			defer func() { fake.unblock(); httpServer.Close() }()
			if blocked == "scheduler" {
				s.loadedMu.Lock()
				defer s.loadedMu.Unlock()
			}
			response, err := (&http.Client{Timeout: 200 * time.Millisecond}).Get(httpServer.URL + "/api/ps")
			if err != nil {
				t.Fatalf("status waited for %s work: %v", blocked, err)
			}
			defer response.Body.Close()
			var status api.ProcessResponse
			require.NoError(t, json.NewDecoder(response.Body).Decode(&status))
			require.Len(t, status.Models, 1, "a loading model must not disappear from status")
			require.Equal(t, "status-fixture:latest", status.Models[0].Name)
			require.Equal(t, int64(4096), status.Models[0].Size)
			require.True(t, status.Models[0].Loading)
		})
	}
}
