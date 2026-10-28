package llm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"strings"
)

const cudaModuleLoadingEnv = "CUDA_MODULE_LOADING"

// Only the CUDA runtime's shared-object initialization failure (error 303)
// qualifies. An exit code, generic module/driver failure or OOM alone does not.
func isCUDASharedObjectInitFailure(message string) bool {
	matched := false
	for _, line := range strings.Split(strings.ToLower(message), "\n") {
		if strings.Contains(line, "cudaerrorsharedobjectinitfailed") ||
			(strings.Contains(line, "cuda error") && strings.Contains(line, "shared object initialization failed")) {
			matched = true
		} else if strings.Contains(line, "cuda error:") {
			// A later specific CUDA failure is not error 303. Source-location
			// lines ending in just "CUDA error" do not replace the diagnosis.
			matched = false
		}
	}
	return matched
}

func cudaModuleLoadingExplicit(extraEnvs map[string]string) bool {
	for key, value := range extraEnvs {
		if strings.EqualFold(key, cudaModuleLoadingEnv) && strings.TrimSpace(value) != "" {
			return true
		}
	}
	for _, entry := range os.Environ() {
		key, value, ok := strings.Cut(entry, "=")
		if ok && strings.EqualFold(key, cudaModuleLoadingEnv) && strings.TrimSpace(value) != "" {
			return true
		}
	}
	return false
}

func (s *llamaServerRunner) shouldRetryCUDAEagerModuleLoading(ctx context.Context, loadErr error, goos string) bool {
	if goos != "windows" || loadErr == nil || ctx.Err() != nil || s.loadReady || s.cudaModuleLoadingRetried ||
		errors.Is(loadErr, context.Canceled) || errors.Is(loadErr, context.DeadlineExceeded) || cudaModuleLoadingExplicit(s.launch.extraEnvs) {
		return false
	}
	usesCUDA := false
	for _, device := range s.launch.gpus {
		usesCUDA = usesCUDA || strings.EqualFold(device.Library, "CUDA")
	}
	message := loadErr.Error()
	lower := strings.ToLower(message)
	if strings.Contains(lower, "cuda error") || strings.Contains(lower, "cudaerror") {
		return usesCUDA && isCUDASharedObjectInitFailure(message)
	}
	return usesCUDA && isCUDASharedObjectInitFailure(s.lastErrMsg())
}

// The restart callback is the same startProcess used by normal startup. Keeping
// that boundary explicit lets the lifecycle be exercised with a fake child,
// without loading a model or a CUDA library in tests.
func (s *llamaServerRunner) retryWithCUDAEagerModuleLoading(ctx context.Context, loadErr error, restart func() error) (bool, error) {
	if !s.shouldRetryCUDAEagerModuleLoading(ctx, loadErr, runtime.GOOS) {
		return false, nil
	}
	s.cudaModuleLoadingRetried = true
	slog.Warn("llama-server CUDA module initialization failed; retrying startup once with eager module loading",
		"model", s.modelPath, "error", loadErr)
	if err := s.stopProcess(); err != nil {
		return false, fmt.Errorf("llama-server startup failed before CUDA_MODULE_LOADING=EAGER retry: %w; stopping failed child: %v", loadErr, err)
	}
	s.resetLoadAccounting()
	if err := ctx.Err(); err != nil {
		return false, fmt.Errorf("CUDA_MODULE_LOADING=EAGER startup retry cancelled before restart: %w", err)
	}
	// Child-local only. Preserve model, context, sampling, layer/fit settings,
	// GPU selection and all other environment entries; never mutate the parent.
	s.launch.extraEnvs = cloneStringMap(s.launch.extraEnvs)
	for key := range s.launch.extraEnvs {
		if strings.EqualFold(key, cudaModuleLoadingEnv) {
			delete(s.launch.extraEnvs, key)
		}
	}
	s.launch.extraEnvs[cudaModuleLoadingEnv] = "EAGER"
	if err := ctx.Err(); err != nil {
		return false, fmt.Errorf("CUDA_MODULE_LOADING=EAGER startup retry cancelled before restart: %w", err)
	}
	if err := restart(); err != nil {
		return false, fmt.Errorf("llama-server startup failed before CUDA_MODULE_LOADING=EAGER retry: %w; starting replacement child: %v", loadErr, err)
	}
	return true, nil
}

func (s *llamaServerRunner) waitUntilRunningWithStartupRetries(ctx context.Context, wait func(context.Context) error, restart func() error) error {
	lastRetry := ""
	for {
		err := wait(ctx)
		if err == nil {
			s.loadReady = true
			return nil
		}
		if ctx.Err() != nil {
			return fmt.Errorf("llama-server startup cancelled: %w", ctx.Err())
		}
		// EAGER is one startup workaround, not an escalating recovery loop. In
		// particular its failure cannot trigger a new CPU/projector fallback.
		if lastRetry == "CUDA_MODULE_LOADING=EAGER" {
			return fmt.Errorf("llama-server startup failed after CUDA_MODULE_LOADING=EAGER retry: %w", err)
		}
		retried, retryErr := s.retryWithCUDAEagerModuleLoading(ctx, err, restart)
		if retryErr != nil {
			return retryErr
		}
		if retried {
			lastRetry = "CUDA_MODULE_LOADING=EAGER"
			continue
		}
		// Preserve the pre-existing projector-OOM workaround. Each workaround
		// has its own once flag, so startup attempts remain bounded.
		retried, retryErr = s.retryWithMMProjCPUOffload(err)
		if retryErr != nil {
			return retryErr
		}
		if retried {
			lastRetry = "projector CPU offload"
			continue
		}
		if lastRetry != "" {
			return fmt.Errorf("llama-server startup failed after %s retry: %w", lastRetry, err)
		}
		return err
	}
}
