package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/require"
)

func idleWaitStarted(t *testing.T, runner *runnerRef) {
	t.Helper()
	require.Eventually(t, func() bool {
		runner.refMu.Lock()
		defer runner.refMu.Unlock()
		return runner.idleCh != nil
	}, time.Second, time.Millisecond)
}

func idleFixture(t *testing.T, references uint) (*Scheduler, *runnerRef) {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	s := InitScheduler(t.Context())
	opts := api.DefaultOptions()
	runner := &runnerRef{modelKey: "active-A", modelPath: "active-A", refCount: references,
		sessionDuration: time.Hour, Options: &opts, llama: &mockLlm{modelPath: "active-A"}}
	runner.model.Store(&Model{ModelPath: "active-A"})
	s.loaded[runner.modelKey] = runner
	return s, runner
}

func TestSchedulerCancelledEvictionDoesNotBlockNextLoad(t *testing.T) {
	testSchedulerCancelledEviction(t, false)
}

func TestSchedulerCancelledOOMEvictionDoesNotBlockNextLoad(t *testing.T) {
	testSchedulerCancelledEviction(t, true)
}

func testSchedulerCancelledEviction(t *testing.T, oom bool) {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "2")
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	s := InitScheduler(ctx)
	s.getSystemInfoFn = func() ml.SystemInfo { return ml.SystemInfo{} }
	opts := api.DefaultOptions()
	opts.NumGPU = 0
	active := &runnerRef{modelKey: "active-A", modelPath: "active-A", refCount: 1,
		sessionDuration: time.Hour, Options: &opts}
	s.loaded[active.modelKey] = active
	makeRequest := func(requestCtx context.Context, path string) *LlmRequest {
		return &LlmRequest{ctx: requestCtx, model: &Model{ModelPath: path}, opts: opts,
			successCh: make(chan *runnerRef, 1), errCh: make(chan error, 1)}
	}
	firstCtx, cancelFirst := context.WithCancel(ctx)
	defer cancelFirst()
	first := makeRequest(firstCtx, "pending-B")
	first.oomRetryAttempted = oom
	next := makeRequest(ctx, "pending-C")
	loadedNext := make(chan struct{}, 1)
	s.loadFn = func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
		if req == first {
			return true // B needs A's memory, which cannot unload while active.
		}
		if req == next {
			loadedNext <- struct{}{}
		}
		return false // C fits alongside A and does not need an unload.
	}
	done := make(chan struct{})
	go func() { defer close(done); s.processPending(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("scheduler did not stop")
		}
	})
	s.pendingReqCh <- first
	idleWaitStarted(t, active)
	cancelFirst()
	s.pendingReqCh <- next
	select {
	case <-loadedNext:
	case <-time.After(250 * time.Millisecond):
		t.Fatal("cancelled eviction request blocked a later model that fits without eviction")
	}
	active.refMu.Lock()
	require.Equal(t, time.Hour, active.sessionDuration)
	active.refMu.Unlock()
	require.Empty(t, s.expiredCh)
}

func TestSchedulerCancellationWinsBeforeIdleExpiration(t *testing.T) {
	s, runner := idleFixture(t, 1)
	requestCtx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan bool, 1)
	go func() { _, ok := s.expireIdleRunner(t.Context(), requestCtx, runner); done <- ok }()
	idleWaitStarted(t, runner)
	runner.refMu.Lock()
	runner.refCount = 0
	runner.notifyIdleLocked()
	cancel()
	runner.refMu.Unlock()
	select {
	case ok := <-done:
		require.False(t, ok)
	case <-time.After(time.Second):
		t.Fatal("idle/cancel race did not stop")
	}
	require.Equal(t, time.Hour, runner.sessionDuration)
	require.Empty(t, s.expiredCh)
}

func TestSchedulerCancelledLoaderCleanupPreservesOtherModel(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	s := InitScheduler(t.Context())
	loader := &mockLlm{modelPath: "other-A"}
	s.activeLoading = loader
	s.discardPendingLoading(&LlmRequest{model: &Model{ModelPath: "cancelled-B"}})
	require.Same(t, loader, s.activeLoading)
	require.False(t, loader.closeCalled.Load())
}

func TestSchedulerCancellationDuringDiscoveryDoesNotStartLoading(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "2")
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	requestCtx, cancelRequest := context.WithCancel(ctx)
	defer cancelRequest()
	s := InitScheduler(ctx)
	s.getSystemInfoFn = func() ml.SystemInfo { return ml.SystemInfo{} }
	discovered, release := make(chan struct{}), make(chan struct{})
	s.getGpuFn = func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		close(discovered)
		<-release
		return nil
	}
	loads := make(chan struct{}, 1)
	s.loadFn = func(*LlmRequest, ml.SystemInfo, []ml.DeviceInfo, bool) bool {
		loads <- struct{}{}
		return false
	}
	done := make(chan struct{})
	go func() { defer close(done); s.processPending(ctx) }()
	opts := api.DefaultOptions()
	s.pendingReqCh <- &LlmRequest{ctx: requestCtx, model: &Model{ModelPath: "cancelled-A"}, opts: opts}
	select {
	case <-discovered:
	case <-time.After(time.Second):
		t.Fatal("discovery did not start")
	}
	cancelRequest()
	close(release)
	// A marker request passes the loop only after the cancelled request was skipped.
	s.pendingReqCh <- &LlmRequest{ctx: ctx, model: &Model{ModelPath: "marker-B"}, opts: api.Options{Runner: api.Runner{NumGPU: 0}}}
	select {
	case <-loads:
	case <-time.After(time.Second):
		t.Fatal("next request did not proceed")
	}
	cancel()
	<-done
	require.Empty(t, loads, "cancelled discovery started a model load")
}

func TestSchedulerCancelledWaitPreservesUpdatedAndExplicitUnloadPolicy(t *testing.T) {
	for _, duration := range []time.Duration{3 * time.Hour, 0} {
		t.Run(duration.String(), func(t *testing.T) {
			s, runner := idleFixture(t, 1)
			requestCtx, cancel := context.WithCancel(t.Context())
			defer cancel()
			done := make(chan bool, 1)
			go func() { _, ok := s.expireIdleRunner(t.Context(), requestCtx, runner); done <- ok }()
			idleWaitStarted(t, runner)
			if duration == 0 {
				s.expireRunner(runner.model.Load())
			} else {
				runner.refMu.Lock()
				runner.sessionDuration = duration
				runner.refMu.Unlock()
			}
			cancel()
			select {
			case ok := <-done:
				require.False(t, ok)
			case <-time.After(time.Second):
				t.Fatal("cancelled wait did not stop")
			}
			runner.refMu.Lock()
			require.Equal(t, duration, runner.sessionDuration)
			runner.refMu.Unlock()
			require.Empty(t, s.expiredCh)
		})
	}
}

func TestSchedulerIdleWaitObservesBothRequestAndServerDeadline(t *testing.T) {
	for _, server := range []bool{false, true} {
		t.Run(map[bool]string{false: "request", true: "server"}[server], func(t *testing.T) {
			s, runner := idleFixture(t, 1)
			deadline, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
			defer cancel()
			ctx, requestCtx := context.Context(t.Context()), context.Context(t.Context())
			if server {
				ctx = deadline
			} else {
				requestCtx = deadline
			}
			enqueued, ok := s.expireIdleRunner(ctx, requestCtx, runner)
			require.False(t, enqueued)
			require.False(t, ok)
			require.Equal(t, time.Hour, runner.sessionDuration)
			require.Empty(t, s.expiredCh)
		})
	}
}

func TestSchedulerIdleWaitRechecksNewReferenceBeforeExpiration(t *testing.T) {
	s, runner := idleFixture(t, 1)
	done := make(chan bool, 1)
	go func() { enqueued, ok := s.expireIdleRunner(t.Context(), t.Context(), runner); done <- enqueued && ok }()
	idleWaitStarted(t, runner)
	runner.refMu.Lock()
	runner.refCount = 0
	runner.notifyIdleLocked()
	runner.refCount = 1 // A new request acquired A before the waiting request resumes.
	runner.refMu.Unlock()
	idleWaitStarted(t, runner)
	require.Empty(t, s.expiredCh)
	runner.refMu.Lock()
	require.Equal(t, time.Hour, runner.sessionDuration)
	runner.refCount = 0
	runner.notifyIdleLocked()
	runner.refMu.Unlock()
	select {
	case ok := <-done:
		require.True(t, ok)
	case <-time.After(time.Second):
		t.Fatal("idle model was never expired")
	}
	require.Len(t, s.expiredCh, 1)
}

func TestSchedulerCancelledFullExpirationQueueDoesNotChangePolicyOrHoldLock(t *testing.T) {
	s, runner := idleFixture(t, 0)
	s.expiredCh = make(chan *runnerRef, 1)
	s.expiredCh <- &runnerRef{}
	runner.expireTimer = time.NewTimer(time.Hour)
	defer runner.expireTimer.Stop()
	original := runner.expireTimer
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Millisecond)
	defer cancel()
	done := make(chan bool, 1)
	go func() { _, ok := s.expireIdleRunner(t.Context(), ctx, runner); done <- ok }()
	runner.refMu.Lock()
	require.Equal(t, time.Hour, runner.sessionDuration)
	runner.refMu.Unlock()
	select {
	case ok := <-done:
		require.False(t, ok)
	case <-time.After(time.Second):
		t.Fatal("full queue did not observe cancellation")
	}
	require.Equal(t, time.Hour, runner.sessionDuration)
	require.Same(t, original, runner.expireTimer)
}

func TestSchedulerCompletedReferencesNotifyIdleOnceAndUnloadNormally(t *testing.T) {
	s, runner := idleFixture(t, 2)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	completed := make(chan struct{})
	go func() { defer close(completed); s.processCompleted(ctx) }()
	t.Cleanup(func() { cancel(); <-completed })
	done := make(chan bool, 1)
	go func() { enqueued, ok := s.expireIdleRunner(ctx, t.Context(), runner); done <- enqueued && ok }()
	idleWaitStarted(t, runner)
	s.finishedReqCh <- &LlmRequest{model: runner.model.Load()}
	require.Eventually(t, func() bool { runner.refMu.Lock(); defer runner.refMu.Unlock(); return runner.refCount == 1 }, time.Second, time.Millisecond)
	require.Empty(t, done)
	s.finishedReqCh <- &LlmRequest{model: runner.model.Load()}
	select {
	case ok := <-done:
		require.True(t, ok)
	case <-time.After(time.Second):
		t.Fatal("idle notification was lost")
	}
	select {
	case <-s.unloadedCh:
	case <-time.After(time.Second):
		t.Fatal("idle model was not unloaded")
	}
	runner.refMu.Lock()
	require.True(t, runner.unloaded)
	require.Nil(t, runner.idleCh)
	runner.refMu.Unlock()
	require.True(t, runner.llama.(*mockLlm).closeCalled.Load())
}

func TestSchedulerBulkCancellationPreservesBusyAndIdleModels(t *testing.T) {
	s, busy := idleFixture(t, 1)
	opts := api.DefaultOptions()
	idle := &runnerRef{modelKey: "idle-B", Options: &opts, sessionDuration: 2 * time.Hour}
	s.loaded[idle.modelKey] = idle
	requestCtx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan bool, 1)
	go func() { done <- s.evictAllAndWait(t.Context(), requestCtx, "new-C") }()
	idleWaitStarted(t, busy)
	cancel()
	select {
	case ok := <-done:
		require.False(t, ok)
	case <-time.After(time.Second):
		t.Fatal("bulk wait ignored cancellation")
	}
	require.Equal(t, time.Hour, busy.sessionDuration)
	require.Equal(t, 2*time.Hour, idle.sessionDuration)
	require.Empty(t, s.expiredCh)
}

func TestSchedulerCancelledRetryClosesOnlyItsPendingLoader(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	s := InitScheduler(ctx)
	s.getSystemInfoFn = func() ml.SystemInfo { return ml.SystemInfo{} }
	requestCtx, cancelRequest := context.WithCancel(ctx)
	loader := &mockLlm{modelPath: "cancelled-A"}
	nextSeen := make(chan struct{}, 1)
	loads := 0
	s.loadFn = func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
		if req.model.ModelPath == "cancelled-A" {
			loads++
			s.loadedMu.Lock()
			s.activeLoading = loader
			s.loadedMu.Unlock()
			cancelRequest()
			return true
		}
		nextSeen <- struct{}{}
		return false
	}
	done := make(chan struct{})
	go func() { defer close(done); s.processPending(ctx) }()
	t.Cleanup(func() { cancel(); <-done })
	opts := api.DefaultOptions()
	opts.NumGPU = 0
	s.pendingReqCh <- &LlmRequest{ctx: requestCtx, model: &Model{ModelPath: "cancelled-A"}, opts: opts}
	s.pendingReqCh <- &LlmRequest{ctx: ctx, model: &Model{ModelPath: "next-B"}, opts: opts}
	select {
	case <-nextSeen:
	case <-time.After(time.Second):
		t.Fatal("cancelled load retry blocked next request")
	}
	require.Equal(t, 1, loads)
	require.True(t, loader.closeCalled.Load())
	s.loadedMu.Lock()
	require.Nil(t, s.activeLoading)
	s.loadedMu.Unlock()
}
