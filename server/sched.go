package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"reflect"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/mlxrunner"
	"github.com/ollama/ollama/types/model"
)

type LlmRequest struct {
	ctx             context.Context //nolint:containedctx
	model           *Model
	opts            api.Options
	sessionDuration *api.Duration
	successCh       chan *runnerRef
	errCh           chan error
	schedAttempts   uint

	// oomRetryAttempted is set after a llama-server load crash triggers an
	// evict-all-and-retry. Prevents infinite retry on persistent load failures.
	oomRetryAttempted bool

	// numCtxAuto is true when NumCtx came from Ollama's automatic VRAM-tier
	// default rather than explicit request, model, or environment config.
	numCtxAuto bool

	// numBatchAuto is true when NumBatch came from Ollama's default options
	// rather than an explicit request or model option.
	numBatchAuto bool

	// useMMapAuto is true when UseMMap was derived by the scheduler rather than
	// explicitly requested.
	useMMapAuto bool

	// contextShift is a llama-server launch attribute resolved from the
	// request-level shift option before scheduling.
	contextShift bool
	shift        *bool
}

type Scheduler struct {
	pendingReqCh  chan *LlmRequest
	finishedReqCh chan *LlmRequest
	expiredCh     chan *runnerRef
	unloadedCh    chan any

	// loadedMu protects loaded and activeLoading
	loadedMu sync.Mutex

	// activeLoading is the model that we are currently working on loading,
	// including by evicting one or more other models. We can only load
	// one model at a time but new requests to models that already loaded can
	// happen in parallel
	activeLoading  llm.LlamaServer
	activeActivity *runnerActivityEntry
	activeStatus   *loadedModel
	loaded         map[string]*runnerRef
	loadedSnapshot atomic.Pointer[loadedRunnerSnapshot]

	loadFn          func(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool
	newServerFn     func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *gguf.Model, adapters []string, projectors []string, opts api.Options, numParallel int, config llm.LlamaServerConfig) (llm.LlamaServer, error)
	getGpuFn        func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo
	getSystemInfoFn func() ml.SystemInfo
	waitForRecovery time.Duration
}

// loadedModelsForName returns models still owned by loaded runners. The models
// are read-only after resolution, so callers may use them after releasing
// loadedMu even if a runner unloads in the meantime.
func (s *Scheduler) loadedModelsForName(name model.Name, requestedRunner string) []*Model {
	snapshot := s.loadedSnapshot.Load()
	if snapshot == nil {
		// Support a zero-value Scheduler as well. Initialized schedulers always
		// publish a registry, so live model resolution never takes this lock.
		s.loadedMu.Lock()
		snapshot = &loadedRunnerSnapshot{runners: make([]*runnerRef, 0, len(s.loaded))}
		for _, runner := range s.loaded {
			snapshot.runners = append(snapshot.runners, runner)
		}
		s.loadedMu.Unlock()
	}

	var models []*Model
	for _, runner := range snapshot.runners {
		m := runner.model.Load()
		if m == nil || !model.ParseName(m.Name).EqualFold(name) {
			continue
		}
		if requestedRunner == "" {
			// A single manifest has no alternate runner to select. For a list,
			// an explicit lower-preference runner may differ from the default.
			singleManifest := m.Digest != "" && m.ManifestDigest == m.Digest
			if m.runnerSelection != "" && !singleManifest && !manifest.IsTopRunnerPreference(m.Runner) {
				continue
			}
		} else if m.Runner != requestedRunner && m.runnerSelection != requestedRunner {
			continue
		}
		models = append(models, m)
	}
	return models
}

// Default automatic value for number of models we allow per GPU
// Model will still need to fit in VRAM, but loading many small models
// on a large GPU can cause stalling
var defaultModelsPerGPU = 3

var ErrMaxQueue = errors.New("server busy, please try again.  maximum pending requests exceeded")

func InitScheduler(ctx context.Context) *Scheduler {
	maxQueue := envconfig.MaxQueue()
	sched := &Scheduler{
		pendingReqCh:    make(chan *LlmRequest, maxQueue),
		finishedReqCh:   make(chan *LlmRequest, maxQueue),
		expiredCh:       make(chan *runnerRef, maxQueue),
		unloadedCh:      make(chan any, maxQueue),
		loaded:          make(map[string]*runnerRef),
		newServerFn:     llm.NewLlamaServer,
		getGpuFn:        discover.GPUDevices,
		getSystemInfoFn: discover.GetSystemInfo,
		waitForRecovery: 5 * time.Second,
	}
	sched.loadFn = sched.load
	sched.loadedSnapshot.Store(&loadedRunnerSnapshot{})
	return sched
}

// schedulerModelKey returns the scheduler map key for a model.
// GGUF-backed models use ModelPath; safetensors/image models without a
// ModelPath use the selected manifest digest so distinct child manifests don't
// collide.
func schedulerModelKey(m *Model) string {
	if m == nil {
		return ""
	}
	if m.ModelPath != "" {
		if len(m.ModelShardPaths) > 0 {
			// NUL cannot appear in a file path, so it unambiguously separates shard paths.
			return strings.Join(m.modelPaths(), "\x00")
		}
		return m.ModelPath
	}
	if m.ManifestDigest != "" {
		return "manifest:" + m.ManifestDigest
	}
	if m.Digest != "" {
		return "digest:" + m.Digest
	}
	if m.Name != "" {
		return "name:" + m.Name
	}
	if m.ShortName != "" {
		return "short:" + m.ShortName
	}
	return ""
}

func resolveContextShift(shift *bool, m *Model) bool {
	if shift != nil {
		return *shift
	}

	return supportsContextShift(m)
}

func supportsContextShift(m *Model) bool {
	if m == nil {
		return true
	}

	if m.Config.ModelFamily == "deepseek2" || slices.Contains(m.Config.ModelFamilies, "deepseek2") {
		return false
	}

	return true
}

func effectiveModelContext(numCtx int, f *gguf.Model) int {
	return effectiveContext(numCtx, modelTrainContext(f))
}

func modelTrainContext(f *gguf.Model) int {
	if f == nil {
		return 0
	}

	return int(f.KV().ContextLength())
}

func effectiveContext(numCtx, trainCtx int) int {
	if trainCtx > 0 && numCtx > trainCtx {
		return trainCtx
	}

	return numCtx
}

func (s *Scheduler) getRunner(c context.Context, m *Model, opts api.Options, sessionDuration *api.Duration, numCtxAuto bool, numBatchAuto bool, shift *bool) (chan *runnerRef, chan error) {
	var policy api.GPUExecutionConfig
	var policyErr error
	opts, policy, policyErr = api.NormalizeGPUExecution(opts)
	if policyErr == nil && policy.Exclusive {
		switch {
		case m.IsMLX():
			policyErr = errors.New("gpu_exclusive requires a GGUF CUDA runner; MLX is not supported")
		case opts.NumCtx < 4:
			policyErr = errors.New("gpu_exclusive requires num_ctx >= 4; context will not be silently increased")
		case m.CheckCapabilities(model.CapabilityVision) == nil && opts.NumCtx < 2048:
			policyErr = errors.New("gpu_exclusive vision models require num_ctx >= 2048; context will not be silently increased")
		}
	}
	if policyErr != nil {
		successCh, errCh := make(chan *runnerRef, 1), make(chan error, 1)
		errCh <- policyErr
		return successCh, errCh
	}
	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	if m.CheckCapabilities(model.CapabilityVision) == nil {
		// multimodal models require at least 2048 context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	contextShift := false
	if m.ModelPath != "" {
		contextShift = resolveContextShift(shift, m)
	}

	req := &LlmRequest{
		ctx:             c,
		model:           m,
		opts:            opts,
		sessionDuration: sessionDuration,
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
		numCtxAuto:      numCtxAuto,
		numBatchAuto:    numBatchAuto,
		contextShift:    contextShift,
		shift:           shift,
	}

	key := schedulerModelKey(req.model)
	var runner *runnerRef
	if snapshot := s.loadedSnapshot.Load(); snapshot != nil {
		runner = snapshot.byKey[key]
	}
	if c.Err() == nil && runner != nil && !runner.needsReload(c, req) && req.useLoadedRunner(runner, s.finishedReqCh) {
		return req.successCh, req.errCh
	}
	if err := c.Err(); err != nil {
		req.errCh <- err
	} else {
		select {
		case <-c.Done():
			req.errCh <- c.Err()
		case s.pendingReqCh <- req:
		default:
			req.errCh <- ErrMaxQueue
		}
	}
	return req.successCh, req.errCh
}

// Returns immediately, spawns go routines for the scheduler which will shutdown when ctx is done
func (s *Scheduler) Run(ctx context.Context) {
	slog.Debug("starting llm scheduler")
	go func() {
		s.processPending(ctx)
	}()

	go func() {
		s.processCompleted(ctx)
	}()
}

func (s *Scheduler) processPending(ctx context.Context) {
	maxRunners := envconfig.MaxRunners()

	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler pending loop")
			return
		case pending := <-s.pendingReqCh:
			// Block other requests until we get this pending request running
			pending.schedAttempts++

			if pending.ctx.Err() != nil {
				slog.Debug("pending request cancelled or timed out, skipping scheduling")
				continue
			}
			logutil.Trace("processing incoming request", "model", pending.model.ModelPath)

		pendingLoop:
			for {
				if ctx.Err() != nil {
					s.discardPendingLoading(pending)
					return
				}
				if pending.ctx.Err() != nil {
					s.discardPendingLoading(pending)
					break
				}
				var runnerToExpire *runnerRef
				pendingKey := schedulerModelKey(pending.model)
				s.loadedMu.Lock()
				runner := s.loaded[pendingKey]
				loadedCount := len(s.loaded)
				runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
				for _, r := range s.loaded {
					runnersSnapshot = append(runnersSnapshot, r)
				}
				s.loadedMu.Unlock()

				if runner != nil {
					if runner.needsReload(pending.ctx, pending) {
						if pending.ctx.Err() != nil {
							break
						}
						slog.Debug("reloading", "runner", runner)
						runnerToExpire = runner
					} else {
						// Runner is usable, return it
						logutil.Trace("using existing loaded runner", "model", pendingKey)
						if pending.useLoadedRunner(runner, s.finishedReqCh) || pending.ctx.Err() != nil {
							break
						}
						continue // the runner unloaded between its health check and reservation
					}
				} else if maxRunners > 0 && loadedCount >= int(maxRunners) {
					slog.Debug("max runners achieved, unloading one to make room", "runner_count", loadedCount)
					runnerToExpire = s.findRunnerToUnload()
				} else {
					// Either no models are loaded or below envconfig.MaxRunners
					// Get a refreshed GPU list
					var gpus []ml.DeviceInfo
					if pending.opts.NumGPU == 0 {
						gpus = []ml.DeviceInfo{}
					} else {
						logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
						gpus = s.getGpuFn(ctx, runnersSnapshot)
					}
					logutil.Trace("refreshing system information", "model", pending.model.ModelPath)
					systemInfo := s.getSystemInfoFn()
					if maxRunners <= 0 {
						// No user specified MaxRunners, so figure out what automatic setting to use for the next load attempt
						if pending.opts.NumGPU == 0 {
							// Need to get actual GPU list to set the correct default max models
							logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
							g := s.getGpuFn(ctx, runnersSnapshot)
							maxRunners = uint(defaultModelsPerGPU * max(len(g), 1))
						} else {
							maxRunners = uint(defaultModelsPerGPU * max(len(gpus), 1))
						}
						slog.Debug("updating default concurrency", "OLLAMA_MAX_LOADED_MODELS", maxRunners, "gpu_count", len(gpus))
					}

					// Update free memory from currently loaded models
					logutil.Trace("updating free space", "gpu_count", len(gpus), "model", pending.model.ModelPath)
					s.updateFreeSpace(gpus)
					if ctx.Err() != nil || pending.ctx.Err() != nil {
						s.discardPendingLoading(pending)
						if ctx.Err() != nil {
							return
						}
						break
					}

					if loadedCount == 0 {
						// No models loaded. Load the model but prefer the best fit.
						slog.Debug("loading first model", "model", pending.model.ModelPath)
						if s.loadFn(pending, systemInfo, gpus, false) {
							slog.Debug("first model load requested retry", "model", pending.model.ModelPath)
							continue
						}
						break
					}

					// More than one loaded model, so we have to see if the
					// new one fits
					logutil.Trace("loading additional model", "model", pending.model.ModelPath)
					needEvict := s.loadFn(pending, systemInfo, gpus, true)
					if !needEvict {
						slog.Debug("new model fits with existing models, loading")
						break
					}

					// OOM retry path: load() crashed post-spawn and we still
					// have other models resident. Evict all of them, wait for
					// every unload, then loop back to retry the load once.
					// load() has already set oomRetryAttempted so a second
					// crash falls through to the fail-fast path.
					if pending.oomRetryAttempted {
						if !s.evictAllAndWait(ctx, pending.ctx, pendingKey) {
							s.discardPendingLoading(pending)
							if ctx.Err() != nil {
								return
							}
							break
						}
						continue
					}

					runnerToExpire = s.findRunnerToUnload()
				}

				if runnerToExpire == nil {
					// While we were performing load calculations, the loaded runner(s) unloaded in parallel
					// so findRunnerToUnload returned no runners.  We'll try again and the loadedCount should be zero
					slog.Debug("runner to expire was nil, retrying")
					continue
				}
				enqueued, ok := s.expireIdleRunner(ctx, pending.ctx, runnerToExpire)
				if !ok {
					s.discardPendingLoading(pending)
					break
				}
				if !enqueued {
					continue
				}
				// Wait for the unload to happen
				slog.Debug("waiting for pending requests to complete and unload to occur", "runner", runnerToExpire)
				select {
				case <-ctx.Done():
					slog.Debug("shutting down scheduler pending loop")
					return
				case <-pending.ctx.Done():
					s.discardPendingLoading(pending)
					break pendingLoop
				case <-s.unloadedCh:
					slog.Debug("unload completed", "runner", runnerToExpire)
					continue
				}
			}
		case <-s.unloadedCh:
			// An unload request when there are no pending request can be ignored
			slog.Debug("ignoring unload event with no pending requests")
		}
	}
}

func (s *Scheduler) processCompleted(ctx context.Context) {
	// Process completed requests, expired timers, and unloading models
	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler completed loop")
			return
		case finished := <-s.finishedReqCh:
			finishedKey := schedulerModelKey(finished.model)
			s.loadedMu.Lock()
			runner := s.loaded[finishedKey]
			s.loadedMu.Unlock()
			if runner == nil {
				slog.Error("finished request signal received after model unloaded", "modelPath", finishedKey)
				continue
			}
			runner.refMu.Lock()
			runner.refCount--
			if runner.refCount <= 0 {
				runner.notifyIdleLocked()
				if runner.sessionDuration <= 0 {
					slog.Debug("runner with zero duration has gone idle, expiring to unload", "runner", runner)
					if runner.expireTimer != nil {
						runner.expireTimer.Stop()
						runner.expireTimer = nil
					}
					s.expiredCh <- runner
				} else if runner.expireTimer == nil {
					slog.Debug("runner with non-zero duration has gone idle, adding timer", "runner", runner, "duration", runner.sessionDuration)
					runner.expireTimer = time.AfterFunc(runner.sessionDuration, func() {
						slog.Debug("timer expired, expiring to unload", "runner", runner)
						runner.refMu.Lock()
						defer runner.refMu.Unlock()
						if runner.expireTimer != nil {
							runner.expireTimer.Stop()
							runner.expireTimer = nil
						}
						s.expiredCh <- runner
					})
					runner.expiresAt = time.Now().Add(runner.sessionDuration)
				} else {
					slog.Debug("runner with non-zero duration has gone idle, resetting timer", "runner", runner, "duration", runner.sessionDuration)
					runner.expireTimer.Reset(runner.sessionDuration)
					runner.expiresAt = time.Now().Add(runner.sessionDuration)
				}
			}
			slog.Debug("after processing request finished event", "runner", runner, "refCount", runner.refCount)
			runner.publishStatusLocked()
			runner.refMu.Unlock()
		case runner := <-s.expiredCh:
			slog.Debug("runner expired event received", "runner", runner)
			runner.refMu.Lock()
			if runner.refCount > 0 {
				slog.Debug("expired event with positive ref count, retrying", "runner", runner, "refCount", runner.refCount)
				go func(runner *runnerRef) {
					// We can't unload yet, but want to as soon as the current request completes
					// So queue up another expired event
					time.Sleep(10 * time.Millisecond)
					s.expiredCh <- runner
				}(runner)
				runner.refMu.Unlock()
				continue
			}

			s.loadedMu.Lock()
			slog.Debug("got lock to unload expired event", "runner", runner)
			runnerToUnload := s.loaded[runner.modelKey]
			if runnerToUnload == nil {
				// If runnerToUnload is nil, we already processed an event and
				// unloaded it. This double unload can happen if the initial
				// request is canceled and we're trying to load another model
				// that requires this one to be evicted, or the settings change
				// and require a reload
				s.loadedMu.Unlock()
				runner.refMu.Unlock()
				slog.Debug("duplicate expired event, ignoring", "runner", runner)
			} else if runner.pid != runnerToUnload.pid {
				// If the pids do not match, we likely had multiple load
				// failures for the same model in quick succession due to
				// request context canceled and are draining the queue of
				// events. Ensure the orphaned runner is properly shut down, but
				// do not delete the mismatched loaded runner, or wait for VRAM
				// convergence.
				slog.Debug("orphaned runner shutting down", "orphan", runner, "loaded", runnerToUnload)
				runner.unload()
				s.loadedMu.Unlock()
				runner.refMu.Unlock()
			} else {
				slog.Debug("starting background wait for VRAM recovery", "runner", runner)
				runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
				for _, r := range s.loaded {
					runnersSnapshot = append(runnersSnapshot, r)
				}
				finished := s.waitForVRAMRecovery(runner, runnersSnapshot)
				runner.unload()
				delete(s.loaded, runner.modelKey)
				s.publishLoadedSnapshotLocked()
				s.loadedMu.Unlock()
				slog.Debug("runner terminated and removed from list, blocking for VRAM recovery", "runner", runner)
				<-finished
				runner.refMu.Unlock()
				slog.Debug("sending an unloaded event", "runner", runner)
				s.unloadedCh <- struct{}{}
			}
		}
	}
}

// Complete the pending request and send the runner back to the requester
// Wires up a finished event after the request context is completed
// Updates session duration, and resets expiration timer
func (pending *LlmRequest) useLoadedRunner(runner *runnerRef, finished chan *LlmRequest) bool {
	if !runner.refMu.lockContext(pending.ctx, pending.ctx) {
		return false
	}
	defer runner.refMu.Unlock()
	if !runner.matchesRequestLocked(pending) {
		return false
	}
	runner.refCount++
	runner.lastUsed = time.Now()
	runner.useCount++
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if pending.sessionDuration != nil {
		runner.sessionDuration = pending.sessionDuration.Duration
	}
	runner.publishStatusLocked()
	pending.successCh <- runner
	go func() {
		<-pending.ctx.Done()
		slog.Debug("context for request finished", "runner", runner)
		finished <- pending
	}()
	return true
}

func (s *Scheduler) discardPendingLoading(req *LlmRequest) {
	wantPath := req.model.ModelPath
	if wantPath == "" {
		wantPath = req.model.ShortName
	}
	s.loadedMu.Lock()
	loading := s.activeLoading
	if loading != nil && loading.ModelPath() == wantPath {
		s.activeLoading, s.activeActivity = nil, nil
		s.activeStatus = nil
		s.publishLoadedSnapshotLocked()
	} else {
		loading = nil
	}
	s.loadedMu.Unlock()
	if loading != nil {
		loading.Close()
	}
}

func (runner *runnerRef) notifyIdleLocked() {
	if runner.idleCh != nil {
		close(runner.idleCh)
		runner.idleCh = nil
	}
}

func (runner *runnerRef) waitForIdle(ctx, requestCtx context.Context) bool {
	for {
		if ctx.Err() != nil || requestCtx.Err() != nil {
			return false
		}
		if !runner.refMu.lockContext(ctx, requestCtx) {
			return false
		}
		if runner.refCount == 0 || runner.unloaded {
			runner.refMu.Unlock()
			return true
		}
		if runner.idleCh == nil {
			runner.idleCh = make(chan struct{})
		}
		idle := runner.idleCh
		runner.refMu.Unlock()
		select {
		case <-ctx.Done():
			return false
		case <-requestCtx.Done():
			return false
		case <-idle:
		}
	}
}

// Waiting requests must not change a busy model's keep-alive policy: the
// request may be cancelled before it needs any of that model's memory.
func (s *Scheduler) expireIdleRunner(ctx, requestCtx context.Context, runner *runnerRef) (enqueued, ok bool) {
	for runner.waitForIdle(ctx, requestCtx) {
		if !runner.refMu.lockContext(ctx, requestCtx) {
			return false, false
		}
		if ctx.Err() != nil || requestCtx.Err() != nil {
			runner.refMu.Unlock()
			return false, false
		}
		if runner.unloaded {
			runner.refMu.Unlock()
			return false, true
		}
		if runner.refCount > 0 {
			runner.refMu.Unlock()
			continue
		}
		select {
		case s.expiredCh <- runner:
			if runner.expireTimer != nil {
				runner.expireTimer.Stop()
				runner.expireTimer = nil
			}
			runner.sessionDuration = 0
			runner.publishStatusLocked()
			runner.refMu.Unlock()
			return true, true
		default:
			runner.refMu.Unlock()
		}
		select {
		case <-ctx.Done():
			return false, false
		case <-requestCtx.Done():
			return false, false
		case <-time.After(10 * time.Millisecond):
		}
	}
	return false, false
}

// load creates a new model based on req and loads it. If requireFull is true then the model must be loaded fully onto GPUs
// (if any). Returns whether the scheduler needs to evict a model to make this one fit.
func (s *Scheduler) load(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool {
	loadStarted := time.Now()
	var policy api.GPUExecutionConfig
	var policyErr error
	req.opts, policy, policyErr = api.NormalizeGPUExecution(req.opts)
	if policyErr == nil && policy.Exclusive {
		if req.model.IsMLX() {
			policyErr = errors.New("gpu_exclusive requires a GGUF CUDA runner; MLX is not supported")
		} else {
			gpus = slices.DeleteFunc(slices.Clone(gpus), func(device ml.DeviceInfo) bool { return !strings.EqualFold(device.Library, "CUDA") })
			policyErr = llm.ValidateGPUExecution(req.opts, gpus, 0)
		}
	}
	if policyErr != nil {
		req.errCh <- policyErr
		return false
	}
	numParallel := max(int(envconfig.NumParallel()), 1)
	if value := req.opts.RunnerEnv["OLLAMA_NUM_PARALLEL"]; value != "" {
		if n, err := strconv.Atoi(value); err == nil && n > 0 && n <= 64 {
			numParallel = n
		}
	}
	completion := req.model.CheckCapabilities(model.CapabilityCompletion) == nil

	// Start conservatively; explicit noncausal GGUF pooling can opt into batching below.
	if !completion {
		numParallel = 1
	}

	// Some architectures are not safe with num_parallel > 1.
	// ref: https://github.com/ollama/ollama/issues/4165
	if slices.Contains([]string{"mllama", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "qwen3next", "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe", "nemotron_h_omni"}, req.model.Config.ModelFamily) && numParallel != 1 {
		numParallel = 1
		slog.Warn("model architecture does not currently support parallel requests", "architecture", req.model.Config.ModelFamily)
	}

	sessionDuration := envconfig.KeepAlive()
	if req.sessionDuration != nil {
		sessionDuration = req.sessionDuration.Duration
	}

	s.loadedMu.Lock()
	llama := s.activeLoading
	var f *gguf.Model
	loadGpus := gpus
	var launchOpts api.Options

	if llama == nil {
		var err error
		if !req.model.IsMLX() {
			var loadErr error
			f, loadErr = llm.LoadModel(req.model.ModelPath, 1024, req.model.ModelShardPaths...)
			if loadErr != nil {
				slog.Info("failed to load model metadata", "model", req.model.ModelPath, "error", loadErr)
				req.errCh <- loadErr
				s.loadedMu.Unlock()
				return false
			}
			if policy.Exclusive {
				if validationErr := llm.ValidateGPUExecution(req.opts, gpus, f.KV().BlockCount()+1); validationErr != nil {
					req.errCh <- validationErr
					s.loadedMu.Unlock()
					return false
				}
				if trainContext := int(f.KV().ContextLength()); trainContext > 0 && req.opts.NumCtx > trainContext {
					req.errCh <- fmt.Errorf("gpu_exclusive requested num_ctx=%d above model training context %d; context will not be silently reduced", req.opts.NumCtx, trainContext)
					s.loadedMu.Unlock()
					return false
				}
			}
			if !completion && f.KV().Has("pooling_type") && !f.KV().Bool("attention.causal", true) {
				numParallel = 8
				if value := os.Getenv("OLLAMA_EMBEDDING_MAX_SEQUENCES"); value != "" {
					if n, err := strconv.Atoi(value); err == nil {
						numParallel = max(1, min(32, n))
					}
				}
			}

			predictedCtx := effectiveLlamaServerContext(req.opts.NumCtx, f, numParallel)
			estimate := llm.EstimateServerVRAM(req.model.ModelPath, f, req.opts, numParallel)
			predicted := estimate.Total()
			req.contextShift = resolveContextShift(req.shift, req.model)
			config := llamaServerConfigForModel(req.model)
			config.ContextShift = req.contextShift
			// An exact prior signature is useful before placement as well as
			// preflight. Never borrow observations from a different context,
			// batch, device, draft or mmap policy.
			placementPrediction := predicted
			if observed, ok := llm.ObservedServerVRAM(req.model.ModelPath, f, req.opts, gpus, numParallel, config); ok {
				placementPrediction = observed
			}
			loadGpus, launchOpts = selectLlamaServerPlacement(systemInfo, gpus, placementPrediction, req.opts)
			availableForBatch, _, _ := availableMemoryForPlacement(systemInfo, loadGpus, launchOpts)
			flashAttention := llm.LlamaServerFlashAttentionForOptions(loadGpus, req.opts)
			req.applyAutomaticGenerationBatch(completion, predictedCtx, predicted, availableForBatch, flashAttention, loadGpus)
			launchOpts.NumBatch = req.opts.NumBatch
			var batchMemory uint64
			launchOpts.NumBatch, batchMemory = llm.LlamaServerBatch(f, launchOpts, completion)
			predictedForLoad := saturatingMemorySum(predicted, batchMemory)
			launchOpts = s.applyLlamaServerMmapDefaults(req, launchOpts, systemInfo, loadGpus, f, numParallel)
			if observed, ok := llm.ObservedServerVRAM(req.model.ModelPath, f, launchOpts, loadGpus, numParallel, config); ok {
				predictedForLoad = observed
			}

			// Pre-flight check: estimate whether the model fits in remaining memory.
			// llama-server auto-detects layers based on available VRAM, so if
			// we predict it won't fit, evict before spawning.
			if requireFull && !policy.Exclusive && !explicitPartialGPUOffload(launchOpts, f) && len(s.loaded) > 0 && len(loadGpus) > 0 {
				freeMemory, gpuFreeMemory, systemLimited := availableMemoryForPlacement(systemInfo, loadGpus, launchOpts)
				// Use 80% of free memory as threshold to leave headroom.
				if predictedForLoad > freeMemory*80/100 {
					slog.Info("llama-server model predicted to exceed available memory, evicting",
						"predicted", format.HumanBytes2(predictedForLoad),
						"predicted_num_ctx", predictedCtx,
						"predicted_kv", format.HumanBytes2(estimate.KV),
						"predicted_recurrent", format.HumanBytes2(estimate.Recurrent),
						"predicted_auxiliary", format.HumanBytes2(estimate.Auxiliary),
						"num_batch", launchOpts.NumBatch,
						"available", format.HumanBytes2(freeMemory),
						"gpu_free", format.HumanBytes2(gpuFreeMemory),
						"system_free", format.HumanBytes2(systemInfo.FreeMemory),
						"system_limited", systemLimited)
					s.loadedMu.Unlock()
					return true
				}
				slog.Info("llama-server model fits alongside existing models",
					"predicted", format.HumanBytes2(predictedForLoad),
					"predicted_num_ctx", predictedCtx,
					"num_batch", launchOpts.NumBatch,
					"available", format.HumanBytes2(freeMemory),
					"gpu_free", format.HumanBytes2(gpuFreeMemory),
					"system_free", format.HumanBytes2(systemInfo.FreeMemory),
					"system_limited", systemLimited)
			}

			llama, err = s.newServerFn(systemInfo, loadGpus, req.model.ModelPath, f, req.model.AdapterPaths, req.model.ProjectorPaths, launchOpts, numParallel, config)
			if err != nil {
				// some older models are not compatible with newer versions of llama.cpp
				// show a generalized compatibility error until there is a better way to
				// check for model compatibility
				if errors.Is(err, gguf.ErrUnsupported) || strings.Contains(err.Error(), "failed to load model") {
					err = fmt.Errorf("%v: this model may be incompatible with your version of Ollama. If you previously pulled this model, try updating it by running `ollama pull %s`", err, req.model.ShortName)
				}
			}
		} else {
			modelName := req.model.ShortName
			llama, err = mlxrunner.NewClient(modelName, req.opts.NumCtx)
		}
		if err != nil {
			slog.Info("failed to create server", "model", req.model.ShortName, "error", err)
			req.errCh <- err
			s.loadedMu.Unlock()
			return false
		}

		s.activeLoading = llama
		s.activeActivity = newRunnerActivity(req.model, llama, launchOpts)
		s.activeStatus = &loadedModel{model: req.model, runner: loadedRunnerName(req.model, llama),
			contextLength: req.opts.NumCtx, sessionDuration: sessionDuration, loading: true, llama: llama}
		s.publishLoadedSnapshotLocked()
	} else {
		wantPath := req.model.ModelPath
		if wantPath == "" {
			wantPath = req.model.ShortName
		}
		if s.activeLoading.ModelPath() != wantPath {
			panic(fmt.Errorf("attempting to load different model after eviction (original %v new %v)", s.activeLoading.ModelPath(), wantPath))
		}
	}

	s.loadedMu.Unlock()

	systemTotalMemory := systemInfo.TotalMemory
	systemFreeMemory := systemInfo.FreeMemory
	systemSwapFreeMemory := systemInfo.FreeSwap
	slog.Info("system memory", "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "free_swap", format.HumanBytes2(systemSwapFreeMemory))

	for _, gpu := range loadGpus {
		available := gpu.FreeMemory - envconfig.GpuOverhead() - gpu.MinimumMemory()
		if gpu.FreeMemory < envconfig.GpuOverhead()+gpu.MinimumMemory() {
			available = 0
		}
		slog.Info("gpu memory", "id", gpu.ID, "library", gpu.Library,
			"available", format.HumanBytes2(available),
			"free", format.HumanBytes2(gpu.FreeMemory),
			"minimum", format.HumanBytes2(gpu.MinimumMemory()),
			"overhead", format.HumanBytes2(envconfig.GpuOverhead()))
	}

	gpuIDs, err := llama.Load(req.ctx, systemInfo, loadGpus, requireFull)
	if err != nil {
		if errors.Is(err, llm.ErrLoadRequiredFull) {
			if !requireFull {
				// No other models loaded, yet we still don't fit, so report an error
				slog.Info("model is too large for system memory", "requireFull", requireFull)
				llama.Close()
				s.loadedMu.Lock()
				s.activeLoading, s.activeActivity, s.activeStatus = nil, nil, nil
				s.publishLoadedSnapshotLocked()
				s.loadedMu.Unlock()
				req.errCh <- err
				return false
			}
			return true
		}

		slog.Info("Load failed", "model", req.model.ModelPath, "error", err)
		llama.Close()
		s.loadedMu.Lock()
		s.activeLoading, s.activeActivity, s.activeStatus = nil, nil, nil
		s.publishLoadedSnapshotLocked()
		loadedCount := len(s.loaded)
		s.loadedMu.Unlock()
		otherLoaded := loadedCount > 0
		if !req.oomRetryAttempted && llm.IsOutOfMemory(err) {
			if oldNumCtx, effectiveNumCtx, newNumCtx, oldNumBatch, newNumBatch, ok := req.reduceAutoNumCtxForLoadOOM(f, numParallel, completion, systemInfo, loadGpus, launchOpts); ok {
				req.oomRetryAttempted = true
				slog.Warn("llama-server load failed; reducing automatic context and retrying once",
					"model", req.model.ModelPath,
					"old_num_ctx", oldNumCtx,
					"effective_num_ctx", effectiveNumCtx,
					"new_num_ctx", newNumCtx,
					"old_num_batch", oldNumBatch,
					"new_num_batch", newNumBatch,
					"loaded_count", loadedCount,
					"evict_all", otherLoaded,
					"error", err)
				return true
			}
		}
		if otherLoaded && !req.oomRetryAttempted && llm.IsOutOfMemory(err) {
			req.oomRetryAttempted = true
			slog.Warn("llama-server load failed; evicting all other models and retrying once", "model", req.model.ModelPath, "error", err)
			return true
		}

		req.errCh <- err
		return false
	}
	logTemplateSelection(req.model)

	// Determine if we have discrete GPUs which we should monitor VRAM usage on during shutdown
	discreteGPUs := false
iGPUScan:
	for _, devid := range gpuIDs {
		for _, dev := range loadGpus {
			if dev.DeviceID == devid {
				if !dev.Integrated {
					discreteGPUs = true
					break iGPUScan
				}
			}
		}
	}

	totalSize, vramSize := llama.MemorySize()
	trainContext := modelTrainContext(f)
	if effectiveNumCtx := llama.ContextLength(); req.model.ModelPath != "" && effectiveNumCtx > 0 {
		req.opts.NumCtx = effectiveNumCtx
		req.contextShift = resolveContextShift(req.shift, req.model)
	}
	runnerName := loadedRunnerName(req.model, llama)
	runner := &runnerRef{
		modelPath:       req.model.ModelPath,
		modelKey:        schedulerModelKey(req.model),
		llama:           llama,
		Options:         &req.opts,
		sessionDuration: sessionDuration,
		gpus:            gpuIDs,
		discreteGPUs:    discreteGPUs,
		runner:          runnerName,
		activity:        newRunnerActivity(req.model, llama, req.opts),
		totalSize:       totalSize,
		vramSize:        vramSize,
		loading:         true,
		pid:             llama.Pid(),
		numCtxAuto:      req.numCtxAuto,
		numBatchAuto:    req.numBatchAuto,
		useMMapAuto:     req.useMMapAuto,
		contextShift:    req.contextShift,
		trainContext:    trainContext,
	}
	runner.model.Store(req.model)
	runner.numParallel = numParallel
	runner.publishStatusLocked()
	runner.refMu.Lock() // hold lock until running or aborted

	s.loadedMu.Lock()
	if oldRunner, ok := s.loaded[runner.modelKey]; ok {
		// Shouldn't happen, but safeguard against leaking a runner
		slog.Warn("model was still loaded", "old_runner", oldRunner, "new_runner", runner)
		oldRunner.refMu.Lock()
		oldRunner.unload()
		oldRunner.refMu.Unlock()
	}
	s.activeLoading = nil
	s.activeActivity = nil
	s.activeStatus = nil
	s.loaded[runner.modelKey] = runner
	s.publishLoadedSnapshotLocked()
	slog.Info("loaded runners", "count", len(s.loaded))
	s.loadedMu.Unlock()

	go func() {
		defer runner.refMu.Unlock()
		err = llama.WaitUntilRunning(req.ctx)
		if err == nil {
			err = req.ctx.Err()
		}
		if err != nil {
			slog.Error("error loading llama server", "error", err)
			req.errCh <- err
			slog.Debug("triggering expiration for failed load", "runner", runner)
			s.expiredCh <- runner
			return
		}
		slog.Debug("finished setting up", "runner", runner)
		if runner.pid < 0 {
			runner.pid = llama.Pid()
		}
		runner.refCount++
		runner.loading = false
		runner.loadDuration = time.Since(loadStarted)
		runner.lastUsed = time.Now()
		runner.useCount++
		runner.publishStatusLocked()
		go func() {
			<-req.ctx.Done()
			slog.Debug("context for request finished")
			s.finishedReqCh <- req
		}()
		req.successCh <- runner
	}()

	return false
}

func (req *LlmRequest) reduceAutoNumCtxForLoadOOM(f *gguf.Model, numParallel int, completion bool, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, launchOpts api.Options) (oldNumCtx, effectiveNumCtx, newNumCtx, oldNumBatch, newNumBatch int, ok bool) {
	if policy, err := api.ResolveGPUExecution(req.opts); err != nil || policy.Exclusive {
		return 0, 0, 0, 0, 0, false
	}
	if !req.numCtxAuto {
		return 0, 0, 0, 0, 0, false
	}

	oldNumCtx = req.opts.NumCtx
	oldNumBatch = req.opts.NumBatch
	effectiveNumCtx = oldNumCtx
	if f != nil {
		if trainCtx := int(f.KV().ContextLength()); trainCtx > 0 && effectiveNumCtx > trainCtx {
			effectiveNumCtx = trainCtx
		}
	}

	newNumCtx, ok = nextLowerAutoNumCtx(effectiveNumCtx)
	if !ok || newNumCtx >= oldNumCtx {
		return 0, 0, 0, 0, 0, false
	}

	req.opts.NumCtx = newNumCtx
	predictedCtx := effectiveLlamaServerContext(req.opts.NumCtx, f, numParallel)
	predictionOpts := launchOpts
	predictionOpts.NumCtx = req.opts.NumCtx
	predictedVRAM := llm.EstimateServerVRAM(req.model.ModelPath, f, predictionOpts, numParallel).Total()
	available, _, _ := availableMemoryForPlacement(systemInfo, gpus, launchOpts)
	req.applyAutomaticGenerationBatch(completion, predictedCtx, predictedVRAM, available, llm.LlamaServerFlashAttentionForOptions(gpus, req.opts), gpus)
	newNumBatch = req.opts.NumBatch
	return oldNumCtx, effectiveNumCtx, newNumCtx, oldNumBatch, newNumBatch, true
}

func explicitPartialGPUOffload(opts api.Options, f *gguf.Model) bool {
	if opts.NumGPU <= 0 || f == nil {
		return false
	}

	return uint64(opts.NumGPU) < f.KV().BlockCount()+1
}

func effectiveLlamaServerContext(numCtx int, f *gguf.Model, numParallel int) int {
	context, parallel := effectiveModelContext(numCtx, f), max(numParallel, 1)
	if context > math.MaxInt/parallel {
		return math.MaxInt
	}
	return context * parallel
}

func saturatingMemorySum(a, b uint64) uint64 {
	if b > math.MaxUint64-a {
		return math.MaxUint64
	}
	return a + b
}

const (
	llamaServerGenerationBatchDefault     = 512
	llamaServerGenerationBatchConstrained = 256
	llamaServerGenerationBatchMedium      = 1024
	llamaServerGenerationBatchLarge       = 2048

	llamaServerGenerationBatchMediumHeadroomPercent = 75
	llamaServerGenerationBatchLargeHeadroomPercent  = 60
)

func (req *LlmRequest) applyAutomaticGenerationBatch(completion bool, effectiveCtx int, predictedVRAM, availableMemory uint64, flashAttention ml.FlashAttentionType, gpus []ml.DeviceInfo) {
	if !completion || !req.numBatchAuto {
		return
	}

	req.opts.NumBatch = automaticGenerationBatch(effectiveCtx, predictedVRAM, availableMemory, flashAttention, gpus)
}

func automaticGenerationBatch(effectiveCtx int, predictedVRAM, availableMemory uint64, flashAttention ml.FlashAttentionType, gpus []ml.DeviceInfo) int {
	if flashAttention == ml.FlashAttentionDisabled && hasCUDADevice(gpus) {
		if constrainedCUDAWithoutFlashAttention(effectiveCtx, gpus) {
			return llamaServerGenerationBatchConstrained
		}
		return llamaServerGenerationBatchDefault
	}

	batch := generationBatchForContext(effectiveCtx)
	for batch > llamaServerGenerationBatchDefault && !generationBatchFits(batch, predictedVRAM, availableMemory) {
		batch = nextLowerGenerationBatch(batch)
	}
	return batch
}

func hasCUDADevice(gpus []ml.DeviceInfo) bool {
	return slices.ContainsFunc(gpus, func(gpu ml.DeviceInfo) bool {
		return gpu.Library == "CUDA"
	})
}

func constrainedCUDAWithoutFlashAttention(effectiveCtx int, gpus []ml.DeviceInfo) bool {
	if effectiveCtx <= 4096 {
		return false
	}
	return slices.ContainsFunc(gpus, func(gpu ml.DeviceInfo) bool {
		if gpu.Library != "CUDA" {
			return false
		}
		memory := gpu.FreeMemory
		if memory == 0 || (gpu.TotalMemory > 0 && gpu.TotalMemory < memory) {
			memory = gpu.TotalMemory
		}
		return memory > 0 && memory <= 8*format.GibiByte
	})
}

func generationBatchForContext(effectiveCtx int) int {
	switch {
	case effectiveCtx > 32768:
		return llamaServerGenerationBatchLarge
	case effectiveCtx > 4096:
		return llamaServerGenerationBatchMedium
	default:
		return llamaServerGenerationBatchDefault
	}
}

func generationBatchFits(batch int, predictedVRAM, availableMemory uint64) bool {
	if predictedVRAM == 0 || availableMemory == 0 {
		return true
	}

	threshold := availableMemory * 80 / 100
	if predictedVRAM > threshold {
		return false
	}
	if !generationBatchHasHeadroom(batch, predictedVRAM, availableMemory) {
		return false
	}

	return llm.LlamaServerBatchMemory(batch) <= threshold-predictedVRAM
}

func generationBatchHasHeadroom(batch int, predictedVRAM, availableMemory uint64) bool {
	switch {
	case batch >= llamaServerGenerationBatchLarge:
		return predictedVRAM <= availableMemory*llamaServerGenerationBatchLargeHeadroomPercent/100
	case batch >= llamaServerGenerationBatchMedium:
		return predictedVRAM <= availableMemory*llamaServerGenerationBatchMediumHeadroomPercent/100
	default:
		return true
	}
}

func nextLowerGenerationBatch(batch int) int {
	switch {
	case batch > llamaServerGenerationBatchMedium:
		return llamaServerGenerationBatchMedium
	default:
		return llamaServerGenerationBatchDefault
	}
}

func nextLowerAutoNumCtx(numCtx int) (int, bool) {
	switch {
	case numCtx > 32768:
		return 32768, true
	case numCtx > 4096:
		return 4096, true
	default:
		return 0, false
	}
}

func availableMemoryForLoad(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo) (available, gpuFree uint64, systemLimited bool) {
	var sharedGPUFree uint64
	var discreteGPUFree uint64
	for _, gpu := range gpus {
		gpuFree += gpu.FreeMemory
		if gpu.Integrated {
			sharedGPUFree += gpu.FreeMemory
		} else {
			discreteGPUFree += gpu.FreeMemory
		}
	}

	// On iGPUs, GPU free memory can be a static or slowly refreshed device
	// baseline. updateFreeSpace has already subtracted known Ollama runner
	// allocations from that baseline. Current system free memory is a separate
	// live measurement that already includes those loaded runners, so use the
	// smaller value for shared-memory GPUs without discounting discrete VRAM.
	if systemInfo.FreeMemory > 0 && sharedGPUFree > 0 && systemInfo.FreeMemory < sharedGPUFree {
		return discreteGPUFree + systemInfo.FreeMemory, gpuFree, true
	}

	return gpuFree, gpuFree, false
}

func availableMemoryForPlacement(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, opts api.Options) (available, gpuFree uint64, systemLimited bool) {
	placementGpus := gpusForPlacement(gpus, opts)
	if len(placementGpus) == 1 && opts.MainGPU != nil {
		gpuFree = placementGpus[0].FreeMemory
		available = availableMemoryForGPU(systemInfo, placementGpus[0])
		systemLimited = available < gpuFree
		return available, gpuFree, systemLimited
	}

	return availableMemoryForLoad(systemInfo, placementGpus)
}

func gpusForPlacement(gpus []ml.DeviceInfo, opts api.Options) []ml.DeviceInfo {
	if opts.MainGPU != nil && *opts.MainGPU >= 0 && *opts.MainGPU < len(gpus) {
		return []ml.DeviceInfo{gpus[*opts.MainGPU]}
	}

	return gpus
}

func selectLlamaServerPlacement(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, predictedVRAM uint64, opts api.Options) ([]ml.DeviceInfo, api.Options) {
	launchOpts := opts
	if len(gpus) <= 1 || opts.NumGPU == 0 {
		return gpus, launchOpts
	}

	groups := ml.ByLibrary(gpus)
	if len(groups) == 0 {
		return gpus, launchOpts
	}

	if opts.MainGPU != nil {
		gpu, available, ok := bestExplicitMainGPU(systemInfo, groups, *opts.MainGPU)
		if !ok {
			selected := bestGPUGroupByAvailableMemory(systemInfo, groups)
			slog.Warn("requested main_gpu is outside the selected GPU group; passing value through to llama-server",
				"main_gpu", *opts.MainGPU,
				"gpu_count", len(selected))
			logSelectedGPUGroup(gpus, selected)
			return selected, launchOpts
		}

		selected, launchOpts := singleLlamaServerGPUPlacement(gpu, launchOpts)
		slog.Info("selecting requested single GPU for llama-server model",
			"requested_main_gpu", *opts.MainGPU,
			"main_gpu", *launchOpts.MainGPU,
			"id", gpu.ID,
			"filter_id", gpu.FilterID,
			"library", gpu.Library,
			"name", gpu.Name,
			"description", gpu.Description,
			"integrated", gpu.Integrated,
			"available", format.HumanBytes2(available))
		logSelectedGPUGroup(gpus, selected)
		return selected, launchOpts
	}

	if !envconfig.SchedSpread() && predictedVRAM > 0 {
		gpu, available, ok := bestSingleGPUFit(systemInfo, groups, predictedVRAM)
		if ok {
			selected, launchOpts := singleLlamaServerGPUPlacement(gpu, launchOpts)
			slog.Info("selecting single GPU for llama-server model",
				"main_gpu", *launchOpts.MainGPU,
				"id", gpu.ID,
				"filter_id", gpu.FilterID,
				"library", gpu.Library,
				"name", gpu.Name,
				"description", gpu.Description,
				"integrated", gpu.Integrated,
				"predicted", format.HumanBytes2(predictedVRAM),
				"available", format.HumanBytes2(available))
			logSelectedGPUGroup(gpus, selected)
			return selected, launchOpts
		}
	}

	selected := bestGPUGroupByAvailableMemory(systemInfo, groups)
	logSelectedGPUGroup(gpus, selected)
	return selected, launchOpts
}

func singleLlamaServerGPUPlacement(gpu ml.DeviceInfo, opts api.Options) ([]ml.DeviceInfo, api.Options) {
	mainGPU := 0
	opts.MainGPU = &mainGPU
	return []ml.DeviceInfo{gpu}, opts
}

func bestExplicitMainGPU(systemInfo ml.SystemInfo, groups [][]ml.DeviceInfo, mainGPU int) (gpu ml.DeviceInfo, available uint64, ok bool) {
	if mainGPU < 0 {
		return ml.DeviceInfo{}, 0, false
	}

	for _, group := range groups {
		if mainGPU >= len(group) {
			continue
		}
		candidate := group[mainGPU]
		candidateAvailable := availableMemoryForGPU(systemInfo, candidate)
		if !ok || betterPlacementGPU(candidate, candidateAvailable, gpu, available) {
			gpu = candidate
			available = candidateAvailable
			ok = true
		}
	}

	return gpu, available, ok
}

func bestSingleGPUFit(systemInfo ml.SystemInfo, groups [][]ml.DeviceInfo, predictedVRAM uint64) (gpu ml.DeviceInfo, available uint64, ok bool) {
	for _, group := range groups {
		for _, candidate := range group {
			candidateAvailable := availableMemoryForGPU(systemInfo, candidate)
			if predictedVRAM > candidateAvailable*80/100 {
				continue
			}
			if !ok || betterPlacementGPU(candidate, candidateAvailable, gpu, available) {
				gpu = candidate
				available = candidateAvailable
				ok = true
			}
		}
	}

	return gpu, available, ok
}

func betterPlacementGPU(candidate ml.DeviceInfo, candidateAvailable uint64, current ml.DeviceInfo, currentAvailable uint64) bool {
	if candidate.Integrated != current.Integrated {
		return !candidate.Integrated
	}

	return candidateAvailable > currentAvailable
}

func bestGPUGroupByAvailableMemory(systemInfo ml.SystemInfo, groups [][]ml.DeviceInfo) []ml.DeviceInfo {
	var best []ml.DeviceInfo
	var bestAvailable uint64
	for _, group := range groups {
		available, _, _ := availableMemoryForLoad(systemInfo, group)
		if best == nil || betterPlacementGroup(group, available, best, bestAvailable) {
			best = group
			bestAvailable = available
		}
	}

	return best
}

func betterPlacementGroup(candidate []ml.DeviceInfo, candidateAvailable uint64, current []ml.DeviceInfo, currentAvailable uint64) bool {
	candidateDiscrete := hasDiscreteGPU(candidate)
	currentDiscrete := hasDiscreteGPU(current)
	if candidateDiscrete != currentDiscrete {
		return candidateDiscrete
	}

	return candidateAvailable > currentAvailable
}

func hasDiscreteGPU(gpus []ml.DeviceInfo) bool {
	for _, gpu := range gpus {
		if !gpu.Integrated {
			return true
		}
	}
	return false
}

func availableMemoryForGPU(systemInfo ml.SystemInfo, gpu ml.DeviceInfo) uint64 {
	if gpu.Integrated && systemInfo.FreeMemory > 0 && systemInfo.FreeMemory < gpu.FreeMemory {
		return systemInfo.FreeMemory
	}

	return gpu.FreeMemory
}

func logSelectedGPUGroup(all, selected []ml.DeviceInfo) {
	if len(selected) == 0 || len(selected) == len(all) {
		return
	}

	slog.Info("selecting GPU backend for llama-server model",
		"library", selected[0].Library,
		"gpu_count", len(selected),
		"available_gpu_count", len(all))
}

func (s *Scheduler) applyLlamaServerMmapDefaults(req *LlmRequest, launchOpts api.Options, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, f *gguf.Model, numParallel int) api.Options {
	if policy, err := api.ResolveGPUExecution(req.opts); err == nil && policy.Exclusive {
		if req.opts.UseMMap == nil {
			enabled := true
			req.opts.UseMMap = &enabled
			req.useMMapAuto = true
		}
		launchOpts.UseMMap = req.opts.UseMMap
		return launchOpts
	}
	predictionOpts := launchOpts
	predictionOpts.NumCtx = req.opts.NumCtx
	predictedVRAM := llm.EstimateServerVRAM(req.model.ModelPath, f, predictionOpts, numParallel).Total()
	availableVRAM, _, _ := availableMemoryForPlacement(systemInfo, gpus, launchOpts)

	if reason := disableMmapDefaultReason(runtime.GOOS, req.opts, gpus, f.KV().BlockCount(), predictedVRAM, availableVRAM); reason != "" {
		useMmap := false
		req.opts.UseMMap = &useMmap
		req.useMMapAuto = true
		slog.Info("disabling mmap for llama-server load by default",
			"model", req.model.ModelPath,
			"reason", reason)
	} else {
		s.maybeDisableMmapForHostPressure(req, launchOpts, systemInfo, gpus, f, numParallel)
	}

	launchOpts.UseMMap = req.opts.UseMMap
	return launchOpts
}

func disableMmapDefaultReason(goos string, opts api.Options, gpus []ml.DeviceInfo, blockCount, predictedVRAM, availableVRAM uint64) string {
	if opts.UseMMap != nil {
		return ""
	}
	if opts.NumGPU == 0 || len(gpus) == 0 || allDevicesLibrary(gpus, "cpu") {
		return "cpu"
	}
	if goos == "windows" && hasDeviceLibrary(gpus, "cuda") {
		return "windows_cuda"
	}
	if hasDeviceLibrary(gpus, "metal") {
		if opts.NumGPU > 0 && blockCount > 0 && uint64(opts.NumGPU) < blockCount+1 {
			return "metal_partial_offload"
		}
		if opts.NumGPU < 0 && predictedVRAM > 0 && availableVRAM > 0 && predictedVRAM > availableVRAM {
			return "metal_partial_offload"
		}
	}
	return ""
}

func hasDeviceLibrary(gpus []ml.DeviceInfo, library string) bool {
	for _, gpu := range gpus {
		if strings.EqualFold(gpu.Library, library) {
			return true
		}
	}
	return false
}

func allDevicesLibrary(gpus []ml.DeviceInfo, library string) bool {
	if len(gpus) == 0 {
		return false
	}
	for _, gpu := range gpus {
		if !strings.EqualFold(gpu.Library, library) {
			return false
		}
	}
	return true
}

func (s *Scheduler) maybeDisableMmapForHostPressure(req *LlmRequest, launchOpts api.Options, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, f *gguf.Model, numParallel int) {
	modelSize := modelFileSize(req.model.modelPaths()...)
	loadedMmapSize := s.loadedMmapModelSizeLocked()
	predictionOpts := launchOpts
	predictionOpts.NumCtx = req.opts.NumCtx
	predictedVRAM := llm.EstimateServerVRAM(req.model.ModelPath, f, predictionOpts, numParallel).Total()
	availableVRAM, _, _ := availableMemoryForPlacement(systemInfo, gpus, launchOpts)
	placementGpus := gpusForPlacement(gpus, launchOpts)

	if !disableMmapForHostPressure(runtime.GOOS, req.opts, systemInfo, placementGpus, modelSize, loadedMmapSize, predictedVRAM, availableVRAM) {
		return
	}

	useMmap := false
	req.opts.UseMMap = &useMmap
	req.useMMapAuto = true
	slog.Info("disabling mmap for llama-server load due to host memory pressure",
		"model", req.model.ModelPath,
		"model_size", format.HumanBytes2(modelSize),
		"loaded_mmap_size", format.HumanBytes2(loadedMmapSize),
		"headroom", format.HumanBytes2(mmapHostPressureHeadroom(systemInfo.TotalMemory)),
		"system_free", format.HumanBytes2(systemInfo.FreeMemory),
		"system_total", format.HumanBytes2(systemInfo.TotalMemory),
		"predicted_vram", format.HumanBytes2(predictedVRAM),
		"available_vram", format.HumanBytes2(availableVRAM),
	)
}

func disableMmapForHostPressure(goos string, opts api.Options, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelSize, loadedMmapSize, predictedVRAM, availableVRAM uint64) bool {
	if opts.UseMMap != nil || goos != "linux" || modelSize == 0 || systemInfo.FreeMemory == 0 || !allDiscreteGPUs(gpus) {
		return false
	}

	// Only back off mmap when we still expect the model to fit on discrete GPU.
	// If VRAM is already tight, disabling mmap can make partial CPU offload
	// worse by turning file-backed mappings into anonymous memory.
	if predictedVRAM == 0 || availableVRAM == 0 || predictedVRAM > availableVRAM*80/100 {
		return false
	}

	pressure := modelSize + loadedMmapSize + mmapHostPressureHeadroom(systemInfo.TotalMemory)
	return systemInfo.FreeMemory < pressure
}

func allDiscreteGPUs(gpus []ml.DeviceInfo) bool {
	if len(gpus) == 0 {
		return false
	}
	for _, gpu := range gpus {
		if gpu.Integrated {
			return false
		}
	}
	return true
}

func mmapHostPressureHeadroom(totalMemory uint64) uint64 {
	if totalMemory == 0 {
		return 8 * format.GigaByte
	}
	return max(8*format.GigaByte, totalMemory/10)
}

func modelFileSize(paths ...string) uint64 {
	var size uint64
	for _, path := range paths {
		if path == "" {
			continue
		}
		info, err := os.Stat(path)
		if err != nil || info.Size() < 0 || uint64(info.Size()) > math.MaxUint64-size {
			return 0
		}
		size += uint64(info.Size())
	}
	return size
}

func (s *Scheduler) loadedMmapModelSizeLocked() uint64 {
	var total uint64
	for _, r := range s.loaded {
		if !runnerUsesMmap(r) {
			continue
		}
		if size := modelFileSize(r.model.Load().modelPaths()...); size > 0 {
			total += size
		} else {
			total += r.totalSize
		}
	}
	return total
}

func runnerUsesMmap(r *runnerRef) bool {
	if r == nil || r.Options == nil || r.Options.UseMMap == nil {
		return true
	}
	return *r.Options.UseMMap
}

func (s *Scheduler) updateFreeSpace(allGpus []ml.DeviceInfo) {
	if len(allGpus) == 0 {
		return
	}
	predMap := map[ml.DeviceID]uint64{} // Sum up the total predicted usage per GPU for all runners
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()
	for _, r := range runners {
		r.refMu.Lock()
		if r.llama != nil {
			for _, gpu := range allGpus {
				predMap[gpu.DeviceID] += r.llama.VRAMByGPU(gpu.DeviceID)
			}
		} else {
			slog.Warn("unexpected nil runner reference, memory prediction may be incorrect")
		}
		r.refMu.Unlock()
	}

	// Now that we've summed up all the GPU usage predictions across all the loaded runners, update the gpu list
	for i := range allGpus {
		if p, ok := predMap[allGpus[i].DeviceID]; ok {
			slog.Debug("gpu reported", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "available", format.HumanBytes2(allGpus[i].FreeMemory))
			if p > allGpus[i].TotalMemory {
				// Shouldn't happen
				slog.Warn("predicted usage exceeds VRAM", "gpu", allGpus[i].ID, "totalMemory", allGpus[i].TotalMemory, "predicted", p)
				allGpus[i].FreeMemory = 0
			} else if (allGpus[i].TotalMemory - p) < allGpus[i].FreeMemory { // predicted free is smaller than reported free, use it
				// TODO maybe we should just always trust our numbers, since cuda's free memory reporting is laggy
				// and we might unload models we didn't actually need to.  The risk is if some other GPU intensive app is loaded
				// after we start our first runner, then we'll never account for that, so picking the smallest free value seems prudent.
				allGpus[i].FreeMemory = allGpus[i].TotalMemory - p
			}
			slog.Info("updated VRAM based on existing loaded models", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "total", format.HumanBytes2(allGpus[i].TotalMemory), "available", format.HumanBytes2(allGpus[i].FreeMemory))
		}
	}
}

// TODO consolidate sched_types.go
type runnerRef struct {
	activity     *runnerActivityEntry // immutable; readable without the load/unload lock
	lastUsed     time.Time
	loadDuration time.Duration
	useCount     uint64
	refMu        contextMutex
	refCount     uint // prevent unloading if > 0
	idleCh       chan struct{}
	unloaded     bool

	llama        llm.LlamaServer
	pid          int
	loading      bool          // True only during initial load, then false forever
	gpus         []ml.DeviceID // Recorded at time of provisioning
	discreteGPUs bool          // True if all devices are discrete GPUs - used to skip VRAM recovery check for iGPUs
	runner       string
	vramSize     uint64
	totalSize    uint64

	sessionDuration time.Duration
	expireTimer     *time.Timer
	expiresAt       time.Time

	// model is atomic so lookups can read it without contending on refMu.
	model        atomic.Pointer[Model]
	status       atomic.Pointer[loadedModel]
	modelPath    string
	modelKey     string
	numParallel  int
	numCtxAuto   bool
	numBatchAuto bool
	useMMapAuto  bool
	contextShift bool
	trainContext int
	*api.Options
}

// The refMu must already be held when calling unload
func (runner *runnerRef) unload() {
	runner.notifyIdleLocked()
	runner.unloaded = true
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if runner.llama != nil {
		runner.llama.Close()
	}
	runner.model.Store(nil)
	runner.status.Store(nil)
	runner.Options = nil
	runner.gpus = nil
	runner.contextShift = false
}

func (runner *runnerRef) needsReload(ctx context.Context, req *LlmRequest) bool {
	slog.Debug("evaluating already loaded", "model", schedulerModelKey(req.model))
	if !runner.refMu.lockContext(ctx, ctx) {
		return true
	}
	if !runner.matchesRequestLocked(req) {
		runner.refMu.Unlock()
		return true
	}

	timeout := 10 * time.Second
	if runner.loading {
		timeout = 2 * time.Minute // Initial load can take a long time for big models on slow systems...
	}
	llama := runner.llama
	runner.refMu.Unlock()
	// A health check is I/O, never a runner critical section. An unload can
	// proceed while it waits; validate again before granting a reference.
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if llama.Ping(ctx) != nil || !runner.refMu.lockContext(ctx, ctx) {
		return true
	}
	defer runner.refMu.Unlock()
	return !runner.matchesRequestLocked(req)
}

// The caller holds refMu. The same validation runs after Ping and directly
// before reservation, closing the health-check/unload race.
func (runner *runnerRef) matchesRequestLocked(req *LlmRequest) bool {
	if runner.unloaded || runner.Options == nil || runner.llama == nil || req.model == nil {
		return false
	}

	// Don't reload runner if num_gpu=-1 was provided
	optsExisting := runner.Options.Runner
	optsNew := req.opts.Runner
	optsNew.NumCtx = effectiveContext(optsNew.NumCtx, runner.trainContext)
	if runner.numCtxAuto && req.numCtxAuto {
		optsNew.NumCtx = optsExisting.NumCtx
	}
	if runner.numBatchAuto && req.numBatchAuto {
		optsNew.NumBatch = optsExisting.NumBatch
	}
	if runner.useMMapAuto && optsNew.UseMMap == nil {
		optsNew.UseMMap = optsExisting.UseMMap
	}
	if optsNew.NumGPU < 0 {
		optsExisting.NumGPU = -1
		optsNew.NumGPU = -1
	}

	contextShift := req.contextShift
	if req.model.ModelPath != "" {
		contextShift = resolveContextShift(req.shift, req.model)
	}
	if runner.contextShift != contextShift {
		return false
	}

	loaded := runner.model.Load()
	if loaded == nil ||
		!reflect.DeepEqual(loaded.AdapterPaths, req.model.AdapterPaths) || // have the adapters changed?
		!reflect.DeepEqual(loaded.ProjectorPaths, req.model.ProjectorPaths) || // have the projectors changed?
		loaded.DraftPath != req.model.DraftPath || !slices.Equal(loaded.DraftShardPaths, req.model.DraftShardPaths) ||
		(!loaded.IsMLX() && !reflect.DeepEqual(optsExisting, optsNew)) { // have the runner options changed?
		return false
	}

	return true
}

// Free memory reporting on GPUs can lag for a while even after the runner
// exits, so we have to keep checking until we see the available memory recover,
// otherwise subsequent model loads will get far less layers loaded or worse
// case, may completely fall back to CPU mode.
// This routine must be called before the runner unloads so it can establish
// a before and after GPU memory allocation.  The returned channel
// will be notified when we're done waiting, or have timed out and should
// proceed anyway
func (s *Scheduler) waitForVRAMRecovery(runner *runnerRef, runners []ml.FilteredRunnerDiscovery) chan any {
	finished := make(chan any, 1)

	// CPU, Metal and iGPUs don't need checking, so no waiting required
	if len(runner.gpus) == 0 || !runner.discreteGPUs ||
		(len(runner.gpus) == 1 && runner.gpus[0].Library == "Metal") {
		finished <- struct{}{}
		slog.Debug("no need to wait for VRAM recovery", "runner", runner)
		return finished
	}
	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), s.waitForRecovery)
	devices := slices.Clone(runner.gpus)

	// Establish a baseline before we unload
	gpusBefore := s.getGpuFn(ctx, runners)
	freeMemoryBefore, baselineComplete := gpuFreeMemoryForDevices(gpusBefore, devices)
	freeMemoryNow := freeMemoryBefore

	go func() {
		// typical convergence is 0.5-1.5s - If it takes too long to discover and converge, let the scheduler estimate VRAM usage
		defer cancel()
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				// Query GPUs, look for free to go back up
				gpusNow := s.getGpuFn(ctx, runners)
				var complete bool
				freeMemoryNow, complete = gpuFreeMemoryForDevices(gpusNow, devices)
				increased := baselineComplete && complete && freeMemoryNow > freeMemoryBefore
				if increased {
					logutil.Trace("gpu VRAM convergence", "percent", int(float32(freeMemoryNow-freeMemoryBefore)/float32(runner.vramSize)*100))
				} else {
					logutil.Trace("gpu VRAM convergence", "percent", 0)
				}
				// If we're within ~75% of the estimated memory usage recovered, bail out
				if increased && float32(freeMemoryNow-freeMemoryBefore) > float32(runner.vramSize)*0.75 {
					slog.Debug(fmt.Sprintf("gpu VRAM free memory converged after %0.2f seconds", time.Since(start).Seconds()), "free_before", format.HumanBytes2(freeMemoryBefore), "free_now", format.HumanBytes2(freeMemoryNow), "runner", runner)
					finished <- struct{}{}
					return
				}
			case <-ctx.Done():
				slog.Debug("gpu VRAM usage didn't recover within timeout", "seconds", time.Since(start).Seconds(), "free_before", format.HumanBytes2(freeMemoryBefore), "free_now", format.HumanBytes2(freeMemoryNow), "runner", runner)
				finished <- struct{}{}
				return
			}
		}
	}()
	return finished
}

// An unrelated GPU gaining free memory is not evidence that this runner's
// allocations were released. Missing discovery data must not establish recovery.
func gpuFreeMemoryForDevices(gpus []ml.DeviceInfo, devices []ml.DeviceID) (uint64, bool) {
	var free uint64
	for _, device := range devices {
		found := false
		for _, gpu := range gpus {
			if gpu.DeviceID == device {
				free = saturatingMemorySum(free, gpu.FreeMemory)
				found = true
				break
			}
		}
		if !found {
			return 0, false
		}
	}
	return free, len(devices) > 0
}

// LogValue may run from goroutines that already hold refMu (see the scheduler
// debug logs), so the unload-mutable fields are read only under TryLock and
// omitted when the lock is contended.
func (runner *runnerRef) LogValue() slog.Value {
	if runner == nil {
		return slog.StringValue("nil")
	}
	modelID := runner.modelPath
	if modelID == "" {
		modelID = runner.modelKey
	}
	attrs := []slog.Attr{}
	if runner.refMu.TryLock() {
		if m := runner.model.Load(); m != nil {
			attrs = append(attrs, slog.String("name", m.Name))
		}
		if len(runner.gpus) > 0 {
			attrs = append(attrs,
				slog.Any("inference", slices.Clone(runner.gpus)),
			)
		}
		attrs = append(attrs, slog.Int("pid", runner.pid))
		if runner.Options != nil {
			attrs = append(attrs, slog.Int("num_ctx", runner.Options.NumCtx))
		}
		runner.refMu.Unlock()
	}
	attrs = append(attrs,
		slog.String("size", format.HumanBytes2(runner.totalSize)),
		slog.String("vram", format.HumanBytes2(runner.vramSize)),
		slog.Int("parallel", runner.numParallel),
		slog.String("model", modelID),
	)
	return slog.GroupValue(attrs...)
}

// Implements discover.RunnerDiscovery
func (runner *runnerRef) GetPort() int {
	if runner.llama != nil {
		return runner.llama.GetPort()
	}
	return -1
}

func (runner *runnerRef) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	if runner.llama != nil {
		return runner.llama.GetDeviceInfos(ctx)
	}
	return nil
}

func (runner *runnerRef) GetActiveDeviceIDs() []ml.DeviceID {
	return runner.gpus
}

func (runner *runnerRef) HasExited() bool {
	if runner.llama != nil {
		return runner.llama.HasExited()
	}
	return true
}

type ByDurationAndName []*runnerRef

func (a ByDurationAndName) Len() int      { return len(a) }
func (a ByDurationAndName) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByDurationAndName) Less(i, j int) bool {
	// Primary sort by session duration (uint64 to handle negatives)
	d1 := uint64(a[i].sessionDuration)
	d2 := uint64(a[j].sessionDuration)
	if d1 != d2 {
		return d1 < d2
	}
	// Secondary sort by model key/path lex order
	n1 := a[i].modelPath
	if n1 == "" {
		n1 = a[i].modelKey
	}
	n2 := a[j].modelPath
	if n2 == "" {
		n2 = a[j].modelKey
	}
	return n1 < n2
}

// TODO - future consideration to pick runners based on size
// type BySize []*runnerRef
// func (a BySize) Len() int           { return len(a) }
// func (a BySize) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
// func (a BySize) Less(i, j int) bool { return a[i].vramSize < a[j].vramSize }

// evictAllAndWait synchronously expires every currently loaded runner except
// the one being loaded (matched by modelKey) and waits for all unload events
// to drain. Returns false if the context was cancelled mid-wait so the caller
// can exit the scheduling loop. Used by the OOM retry path in processPending.
func (s *Scheduler) evictAllAndWait(ctx, requestCtx context.Context, keepKey string) bool {
	s.loadedMu.Lock()
	runnersToExpire := make([]*runnerRef, 0, len(s.loaded))
	for key, r := range s.loaded {
		if key == keepKey {
			continue
		}
		runnersToExpire = append(runnersToExpire, r)
	}
	s.loadedMu.Unlock()

	if len(runnersToExpire) == 0 {
		return true
	}

	slog.Info("evicting all other loaded models for OOM retry", "count", len(runnersToExpire))
	for _, runner := range runnersToExpire {
		if !runner.waitForIdle(ctx, requestCtx) {
			return false
		}
	}
	unloads := 0
	for _, runner := range runnersToExpire {
		enqueued, ok := s.expireIdleRunner(ctx, requestCtx, runner)
		if !ok {
			return false
		}
		if enqueued {
			unloads++
		}
	}

	// Wait for every unload event. Each runner produces exactly one
	// unloadedCh signal when its cleanup finishes.
	for range unloads {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler during evict-all wait")
			return false
		case <-requestCtx.Done():
			return false
		case <-s.unloadedCh:
		}
	}
	return true
}

func (s *Scheduler) expireRunnersForRuntimeOOM(model *Model, err error) {
	if !llm.IsOutOfMemory(err) {
		return
	}

	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, runner := range s.loaded {
		runners = append(runners, runner)
	}
	s.loadedMu.Unlock()

	if len(runners) == 0 {
		return
	}

	slog.Warn("runtime OOM detected; expiring loaded models to clear memory before next request", "model", schedulerModelKey(model), "error", err)
	for _, runner := range runners {
		runner.refMu.Lock()
		if runner.expireTimer != nil {
			runner.expireTimer.Stop()
			runner.expireTimer = nil
		}
		runner.sessionDuration = 0
		runner.publishStatusLocked()
		if runner.refCount <= 0 {
			s.expiredCh <- runner
		}
		runner.refMu.Unlock()
	}
}

// findRunnerToUnload finds a runner to unload to make room for a new model
func (s *Scheduler) findRunnerToUnload() *runnerRef {
	s.loadedMu.Lock()
	runnerList := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runnerList = append(runnerList, r)
	}
	s.loadedMu.Unlock()
	if len(runnerList) == 0 {
		slog.Debug("no loaded runner to unload")
		return nil
	}

	// In the future we can enhance the algorithm to be smarter about picking the optimal runner to unload
	// e.g., if we have multiple options, will one make room for the request?
	now := time.Now()
	values := make(map[*runnerRef]float64, len(runnerList))
	durations := make(map[*runnerRef]time.Duration, len(runnerList))
	names := make(map[*runnerRef]string, len(runnerList))
	allKnown := true
	for _, runner := range runnerList {
		runner.refMu.Lock()
		values[runner] = runnerEvictionValue(runner, now)
		durations[runner] = runner.sessionDuration
		names[runner] = runner.modelPath
		if names[runner] == "" {
			names[runner] = runner.modelKey
		}
		runner.refMu.Unlock()
		allKnown = allKnown && values[runner] >= 0
	}
	sort.SliceStable(runnerList, func(i, j int) bool {
		a, b := runnerList[i], runnerList[j]
		if durations[a] != durations[b] {
			return uint64(durations[a]) < uint64(durations[b])
		}
		if allKnown && values[a] != values[b] {
			return values[a] < values[b]
		}
		return names[a] < names[b]
	})

	// First try to find a runner that's already idle
	for _, runner := range runnerList {
		runner.refMu.Lock()
		rc := runner.refCount
		runner.refMu.Unlock()
		if rc == 0 {
			slog.Debug("found an idle runner to unload", "runner", runner)
			return runner
		}
	}
	// None appear idle, just wait for the one with the shortest duration
	slog.Debug("no idle runners, picking the shortest duration", "runner_count", len(runnerList), "runner", runnerList[0])
	return runnerList[0]
}

func (s *Scheduler) unloadAllRunners() {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()

	if s.activeLoading != nil {
		slog.Debug("shutting down currently loading runner")
		s.activeLoading.Close()
		s.activeLoading = nil
		s.activeActivity = nil
		s.activeStatus = nil
		s.publishLoadedSnapshotLocked()
	}

	for model, runner := range s.loaded {
		if runner.llama != nil {
			slog.Debug("shutting down runner", "model", model)
			runner.llama.Close()
		}
	}
}

func (s *Scheduler) expireRunner(model *Model) {
	modelKey := schedulerModelKey(model)
	var runners []*runnerRef

	s.loadedMu.Lock()
	if runner, ok := s.loaded[modelKey]; ok {
		runners = append(runners, runner)
	} else if model != nil && model.Name != "" {
		for _, runner := range s.loaded {
			if m := runner.model.Load(); m != nil && m.Name == model.Name {
				runners = append(runners, runner)
			}
		}
	}
	s.loadedMu.Unlock()

	for _, runner := range runners {
		runner.refMu.Lock()
		runner.expiresAt = time.Now()
		if runner.expireTimer != nil {
			runner.expireTimer.Stop()
			runner.expireTimer = nil
		}
		runner.sessionDuration = 0
		runner.publishStatusLocked()
		if runner.refCount <= 0 {
			s.expiredCh <- runner
		}
		runner.refMu.Unlock()
	}
}

// loadedModel is a point-in-time snapshot of a loaded runner's state, safe to
// use without holding any scheduler locks.
type loadedModel struct {
	model           *Model
	runner          string
	size            int64
	sizeVRAM        int64
	contextLength   int
	expiresAt       time.Time
	sessionDuration time.Duration
	loading         bool
	llama           llm.LlamaServer
}

type loadedRunnerSnapshot struct {
	runners []*runnerRef
	byKey   map[string]*runnerRef
	loading *loadedModel
}

func loadedRunnerName(model *Model, runner llm.LlamaServer) string {
	name := model.Runner
	if model.IsMLX() && name == "" {
		return manifest.RunnerMLX
	}
	if llm.IsLlamaCPP(runner) {
		// The loaded process is authoritative even for legacy manifests.
		return manifest.RunnerLlamaCPP
	}
	return name
}

// The caller holds loadedMu; published registries are immutable.
func (s *Scheduler) publishLoadedSnapshotLocked() {
	snapshot := &loadedRunnerSnapshot{runners: make([]*runnerRef, 0, len(s.loaded)), byKey: make(map[string]*runnerRef, len(s.loaded)), loading: s.activeStatus}
	for key, runner := range s.loaded {
		snapshot.runners = append(snapshot.runners, runner)
		snapshot.byKey[key] = runner
	}
	s.loadedSnapshot.Store(snapshot)
}

// The caller holds refMu, or has not published this runner yet.
func (runner *runnerRef) publishStatusLocked() {
	status := &loadedModel{
		model: runner.model.Load(), runner: runner.runner, size: int64(runner.totalSize),
		sizeVRAM: int64(runner.vramSize), expiresAt: runner.expiresAt,
		sessionDuration: runner.sessionDuration, loading: runner.loading, llama: runner.llama,
	}
	if runner.Options != nil {
		status.contextLength = runner.Options.NumCtx
	}
	runner.status.Store(status)
}

// loadedModels returns a snapshot of the currently loaded models for status
// reporting without exposing the scheduler's internal runner bookkeeping.
func (s *Scheduler) loadedModels() []loadedModel {
	registry := s.loadedSnapshot.Load()
	if registry == nil {
		return nil
	}
	models := make([]loadedModel, 0, len(registry.runners))
	if registry.loading != nil {
		models = append(models, registry.loading.current())
	}
	for _, r := range registry.runners {
		status := r.status.Load()
		m := r.model.Load()
		if m == nil || status == nil {
			// Unloaded after the snapshot above was taken
			continue
		}
		lm := *status
		lm.model = m
		models = append(models, lm.current())
	}
	return models
}

func (status loadedModel) current() loadedModel {
	// Both backends expose local memory/context snapshots, without HTTP or
	// scheduler locks. Memory may grow while the native process is loading.
	if status.llama != nil {
		status.contextLength = status.llama.ContextLength()
		total, vram := status.llama.MemorySize()
		status.size, status.sizeVRAM = int64(total), int64(vram)
	}
	if status.expiresAt.IsZero() {
		status.expiresAt = time.Now().Add(status.sessionDuration)
	}
	return status
}
