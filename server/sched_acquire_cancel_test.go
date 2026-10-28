package server

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func acquireTestRunner(t *testing.T) (*Scheduler, *runnerRef, *LlmRequest) {
	t.Helper()
	opts := api.DefaultOptions()
	opts.NumCtx = 32
	m := &Model{ModelPath: "cancel-test-model"}
	r := withModel(&runnerRef{llama: &mockLlm{}, Options: &opts, contextShift: true}, m)
	s := InitScheduler(t.Context())
	s.loaded[schedulerModelKey(m)] = r
	s.publishLoadedSnapshotLocked()
	req := &LlmRequest{ctx: t.Context(), model: m, opts: opts}
	return s, r, req
}

func TestRunnerReloadWaitHonorsCancellation(t *testing.T) {
	_, runner, req := acquireTestRunner(t)
	runner.refMu.Lock()
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan bool, 1)
	go func() { done <- runner.needsReload(ctx, req) }()
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		runner.refMu.Unlock()
		<-done
		t.Fatal("cancelled request blocked on a loading runner mutex")
	}
	runner.refMu.Unlock()
}

func TestLoadedRunnerRequestDeadlineDuringLongLoad(t *testing.T) {
	s, runner, req := acquireTestRunner(t)
	runner.refMu.Lock() // held across WaitUntilRunning by the real loader
	defer runner.refMu.Unlock()
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		success, failed := s.getRunner(ctx, req.model, req.opts, nil, false, false, nil)
		select {
		case <-success:
			done <- errors.New("deadline request acquired loading runner")
		case err := <-failed:
			done <- err
		}
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("deadline waited for model load to release its reference mutex")
	}
	if runner.refCount != 0 || len(s.pendingReqCh) != 0 {
		t.Fatal("timed out request left pending model work")
	}
}

func TestLoadedModelMetadataLookupDoesNotWaitForGlobalLoadMutex(t *testing.T) {
	s, _, req := acquireTestRunner(t)
	req.model.Name = "metadata-loaded-model"
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	done := make(chan []*Model, 1)
	go func() { done <- s.loadedModelsForName(model.ParseName(req.model.Name), "") }()
	select {
	case got := <-done:
		if len(got) != 1 || got[0] != req.model {
			t.Fatalf("models=%+v", got)
		}
	case <-time.After(time.Second):
		t.Fatal("metadata lookup blocked behind unrelated model construction")
	}
}

func TestLoadedRunnerLookupDoesNotWaitForOtherModelLoad(t *testing.T) {
	s, runner, req := acquireTestRunner(t)
	s.loadedMu.Lock()
	done := make(chan error, 1)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	go func() {
		ready, failed := s.getRunner(ctx, req.model, req.opts, nil, false, false, nil)
		select {
		case got := <-ready:
			if got != runner {
				done <- errors.New("wrong loaded runner")
				return
			}
			done <- nil
		case err := <-failed:
			done <- err
		case <-ctx.Done():
			done <- ctx.Err()
		}
	}()
	select {
	case err := <-done:
		s.loadedMu.Unlock()
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		s.loadedMu.Unlock()
		cancel()
		<-done
		t.Fatal("ready model request blocked behind another model's loadedMu")
	}
}

type acquireBlockingPing struct {
	mockLlm
	started chan struct{}
	finish  chan struct{}
	once    sync.Once
}

func (r *acquireBlockingPing) Ping(ctx context.Context) error {
	r.once.Do(func() { close(r.started) })
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-r.finish:
		return nil
	}
}

func TestRunnerPingDoesNotHoldReferenceMutex(t *testing.T) {
	_, runner, req := acquireTestRunner(t)
	ping := &acquireBlockingPing{started: make(chan struct{}), finish: make(chan struct{})}
	runner.llama = ping
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan bool, 1)
	go func() { done <- runner.needsReload(ctx, req) }()
	<-ping.started
	locked := runner.refMu.TryLock()
	if locked {
		runner.refMu.Unlock()
	}
	cancel()
	<-done
	if !locked {
		t.Fatal("runner Ping holds refMu and blocks cancellation/unload/other requests")
	}
}

func TestCancelledRunnerRequestNeverQueuedOrReserved(t *testing.T) {
	for _, loaded := range []bool{false, true} {
		t.Run(map[bool]string{false: "cold", true: "loaded"}[loaded], func(t *testing.T) {
			s, runner, req := acquireTestRunner(t)
			if !loaded {
				delete(s.loaded, schedulerModelKey(req.model))
				s.publishLoadedSnapshotLocked()
			}
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			success, failure := s.getRunner(ctx, req.model, req.opts, nil, false, false, nil)
			select {
			case err := <-failure:
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("error = %v", err)
				}
			default:
				t.Fatal("missing cancellation error")
			}
			if len(success) != 0 || len(s.pendingReqCh) != 0 || runner.refCount != 0 {
				t.Fatal("cancelled request reserved or queued model work")
			}
		})
	}
}

func TestRunnerUnloadedDuringPingIsNotReserved(t *testing.T) {
	s, runner, req := acquireTestRunner(t)
	ping := &acquireBlockingPing{started: make(chan struct{}), finish: make(chan struct{})}
	runner.llama = ping
	done := make(chan struct{})
	var success chan *runnerRef
	go func() {
		success, _ = s.getRunner(t.Context(), req.model, req.opts, nil, false, false, nil)
		close(done)
	}()
	<-ping.started
	runner.refMu.Lock()
	runner.unloaded = true
	runner.model.Store(nil)
	runner.Options = nil
	runner.refMu.Unlock()
	close(ping.finish)
	<-done
	if len(success) != 0 || runner.refCount != 0 || len(s.pendingReqCh) != 1 {
		t.Fatal("unloaded runner was reused after its in-flight health check")
	}
}

func TestRunnerReservationRevalidatesAfterHealthCheck(t *testing.T) {
	s, runner, req := acquireTestRunner(t)
	if runner.needsReload(t.Context(), req) {
		t.Fatal("healthy fixture needs reload")
	}
	runner.refMu.Lock()
	runner.unloaded = true
	runner.model.Store(nil)
	runner.Options = nil
	runner.refMu.Unlock()
	if req.useLoadedRunner(runner, s.finishedReqCh) || runner.refCount != 0 {
		t.Fatal("stale health check allowed reservation of an unloaded runner")
	}
}

func TestRunnerIdleWaitCancelsWhileLoadLockHeld(t *testing.T) {
	s, runner, _ := acquireTestRunner(t)
	runner.refMu.Lock()
	defer runner.refMu.Unlock()
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan bool, 1)
	go func() { _, ok := s.expireIdleRunner(t.Context(), ctx, runner); done <- ok }()
	cancel()
	select {
	case ok := <-done:
		if ok {
			t.Fatal("cancelled waiter expired runner")
		}
	case <-time.After(time.Second):
		t.Fatal("idle waiter stuck behind load mutex")
	}
}

func TestContextMutexConcurrentWaiters(t *testing.T) {
	var mutex contextMutex
	mutex.Lock()
	var wg sync.WaitGroup
	var protected int
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if !mutex.lockContext(t.Context(), t.Context()) {
				return
			}
			protected++
			mutex.Unlock()
		}()
	}
	mutex.Unlock()
	wg.Wait()
	if protected != 32 {
		t.Fatalf("only %d waiters completed", protected)
	}
}
