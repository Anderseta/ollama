package discover

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func TestEmptyCUDABootstrapRetriesOnce(t *testing.T) {
	calls, waits := 0, 0
	probe := func(ctx context.Context, env map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
		calls++
		if env["CUDA_VISIBLE_DEVICES"] != "0" {
			t.Fatal("retry changed the selected device environment")
		}
		if calls == 1 {
			return nil, nil, nil
		}
		return []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}}}, nil, nil
	}
	wait := func(context.Context) error { waits++; return nil }
	devices, _, err := runEmptyCUDABootstrapRetry(context.Background(), []string{filepath.Join(t.TempDir(), "cuda_v12")}, map[string]string{"CUDA_VISIBLE_DEVICES": "0"}, probe, time.Now, wait)
	if err != nil || len(devices) != 1 || devices[0].Library != "CUDA" || calls != 2 || waits != 1 {
		t.Fatalf("devices=%v err=%v calls=%d waits=%d", devices, err, calls, waits)
	}
}

func TestEmptyCUDABootstrapDoesNotRetryFailures(t *testing.T) {
	driverFailure := llm.NewStatusWriter(io.Discard)
	driverFailure.SetLastError("CUDA driver version is insufficient for CUDA runtime version")
	for _, tc := range []struct {
		name    string
		devices []ml.DeviceInfo
		status  *llm.StatusWriter
		err     error
		library string
		elapsed time.Duration
	}{
		{name: "already found GPU", devices: []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}}}, library: "cuda_v12"},
		{name: "timeout", err: context.DeadlineExceeded, library: "cuda_v12"},
		{name: "driver error", err: errors.New("incompatible CUDA driver"), library: "cuda_v12"},
		{name: "driver status", status: driverFailure, library: "cuda_v12"},
		{name: "slow empty probe", library: "cuda_v12", elapsed: 5 * time.Second},
		{name: "CPU library", library: "cpu"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls, waits, clockCalls := 0, 0, 0
			base := time.Now()
			now := func() time.Time {
				clockCalls++
				if clockCalls == 1 {
					return base
				}
				return base.Add(tc.elapsed)
			}
			probe := func(context.Context, map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
				calls++
				return tc.devices, tc.status, tc.err
			}
			_, _, err := runEmptyCUDABootstrapRetry(context.Background(), []string{filepath.Join(t.TempDir(), tc.library)}, nil, probe, now, func(context.Context) error { waits++; return nil })
			if !errors.Is(err, tc.err) || calls != 1 || waits != 0 {
				t.Fatalf("err=%v calls=%d waits=%d", err, calls, waits)
			}
		})
	}
}

func TestEmptyCUDABootstrapCancelsBeforeRetry(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	probe := func(context.Context, map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
		calls++
		return nil, nil, nil
	}
	wait := func(ctx context.Context) error { cancel(); return waitForCUDAEmptyRetry(ctx) }
	_, _, err := runEmptyCUDABootstrapRetry(ctx, []string{"cuda_v12"}, nil, probe, time.Now, wait)
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
	calls = 0
	_, _, err = runEmptyCUDABootstrapRetry(ctx, []string{"cuda_v12"}, nil, probe, time.Now, wait)
	if !errors.Is(err, context.Canceled) || calls != 0 {
		t.Fatalf("pre-canceled err=%v calls=%d", err, calls)
	}
}

func TestEmptyCUDABootstrapDoesNotRetryInitValidation(t *testing.T) {
	calls, waits := 0, 0
	probe := func(context.Context, map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
		calls++
		return nil, nil, nil
	}
	_, _, err := runEmptyCUDABootstrapRetry(context.Background(), []string{"cuda_v12"}, map[string]string{"GGML_CUDA_INIT": "1"}, probe, time.Now,
		func(context.Context) error { waits++; return nil })
	if err != nil || calls != 1 || waits != 0 {
		t.Fatalf("err=%v calls=%d waits=%d", err, calls, waits)
	}
}

func TestEmptyCUDABootstrapBaseLibraryIsBounded(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "ggml-cuda.dll"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	calls, waits := 0, 0
	probe := func(context.Context, map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
		calls++
		return nil, nil, nil
	}
	devices, _, err := runEmptyCUDABootstrapRetry(context.Background(), []string{dir}, nil, probe, time.Now,
		func(context.Context) error { waits++; return nil })
	if err != nil || len(devices) != 0 || calls != 2 || waits != 1 {
		t.Fatalf("devices=%v err=%v calls=%d waits=%d", devices, err, calls, waits)
	}
}

func TestEmptyCUDABootstrapCancelledAtWaitBoundary(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls := 0
	probe := func(context.Context, map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
		calls++
		return nil, nil, nil
	}
	_, _, err := runEmptyCUDABootstrapRetry(ctx, []string{"cuda_v12"}, nil, probe, time.Now,
		func(context.Context) error { cancel(); return nil })
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}

func TestCancelledGPUDiscoveryDoesNotCacheEmptyResult(t *testing.T) {
	deviceMu.Lock()
	previousBootstrapped, previousDevices, previousDirs := bootstrapped, devices, libDirs
	bootstrapped, devices, libDirs = false, nil, nil
	deviceMu.Unlock()
	t.Cleanup(func() {
		deviceMu.Lock()
		bootstrapped, devices, libDirs = previousBootstrapped, previousDevices, previousDirs
		deviceMu.Unlock()
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result := GPUDevices(ctx, nil); len(result) != 0 {
		t.Fatalf("cancelled discovery returned %v", result)
	}
	deviceMu.Lock()
	defer deviceMu.Unlock()
	if bootstrapped {
		t.Fatal("cancelled request permanently cached empty GPU discovery")
	}
}
