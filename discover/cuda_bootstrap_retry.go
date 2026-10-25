package discover

import (
	"context"
	"log/slog"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

type cudaBootstrapProbe func(context.Context, map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error)

// Retry a fast, empty CUDA result once before caching CPU-only discovery.
// Successful discovery, errors, timeouts and driver failures keep their path.
func retryEmptyCUDABootstrap(ctx context.Context, dirs []string, env map[string]string, probe cudaBootstrapProbe) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
	return runEmptyCUDABootstrapRetry(ctx, dirs, env, probe, time.Now, waitForCUDAEmptyRetry)
}

func runEmptyCUDABootstrapRetry(ctx context.Context, dirs []string, env map[string]string, probe cudaBootstrapProbe, now func() time.Time, wait func(context.Context) error) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	started := now()
	devices, status, err := probe(ctx, env)
	if err != nil || len(devices) != 0 || env["GGML_CUDA_INIT"] == "1" || lastDiscoveryStatusError(status) != "" || now().Sub(started) >= 5*time.Second || !hasCUDABootstrapLibrary(dirs) {
		return devices, status, err
	}
	if err := ctx.Err(); err != nil {
		return devices, status, err
	}
	if err := wait(ctx); err != nil {
		return devices, status, err
	}
	if err := ctx.Err(); err != nil {
		return devices, status, err
	}
	slog.Debug("retrying fast empty CUDA discovery once", "library_paths", dirs)
	return probe(ctx, env)
}

func hasCUDABootstrapLibrary(dirs []string) bool {
	for _, dir := range dirs {
		base := strings.ToLower(filepath.Base(dir))
		if base == "cuda" || strings.HasPrefix(base, "cuda_") {
			return true
		}
		// Local builds may keep the CUDA module beside the CPU/base libraries.
		if files, _ := filepath.Glob(filepath.Join(dir, "*ggml-cuda*")); len(files) != 0 {
			return true
		}
	}
	return false
}

func waitForCUDAEmptyRetry(ctx context.Context) error {
	timer := time.NewTimer(100 * time.Millisecond)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return ctx.Err()
	}
}
