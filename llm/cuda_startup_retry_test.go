package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

const cuda303TestLog = "CUDA error: shared object initialization failed\n" +
	"  current device: 0, in function ggml_cuda_flash_attn_ext_mma_f16_case\n" +
	"  cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)\n" +
	"C:/native/ggml-cuda.cu:108: CUDA error\n"

func cudaRetryTestRunner() *llamaServerRunner {
	options := api.DefaultOptions()
	options.NumCtx = 131072
	options.NumGPU = 33
	runner := &llamaServerRunner{
		status: NewStatusWriter(io.Discard),
		launch: llamaServerLaunchConfig{
			modelPath: "same-model.gguf", modelArch: "qwen35", draftType: draftTypeMTP,
			opts: options, numParallel: 1, kvCacheType: "q8_0",
			gpus:      []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA", ID: "0"}}},
			extraEnvs: map[string]string{"KEEP": "same-value", "cuda_module_loading": ""},
		},
		vramByDevice: map[string]uint64{}, systemFreeAtLoad: map[string]uint64{},
	}
	runner.output = &memoryParsingWriter{inner: runner.status, runner: runner}
	return runner
}

func TestShouldRetryCUDAEagerModuleLoading(t *testing.T) {
	t.Setenv(cudaModuleLoadingEnv, "")
	tests := []struct {
		name   string
		goos   string
		change func(*llamaServerRunner, *error)
		want   bool
	}{
		{name: "Windows CUDA captured real crash", goos: "windows", want: true},
		{name: "Linux CUDA preserved", goos: "linux"},
		{name: "Metal preserved", goos: "windows", change: func(s *llamaServerRunner, _ *error) { s.launch.gpus[0].Library = "Metal" }},
		{name: "ROCm preserved", goos: "windows", change: func(s *llamaServerRunner, _ *error) { s.launch.gpus[0].Library = "ROCm" }},
		{name: "once only", goos: "windows", change: func(s *llamaServerRunner, _ *error) { s.cudaModuleLoadingRetried = true }},
		{name: "never after ready", goos: "windows", change: func(s *llamaServerRunner, _ *error) { s.loadReady = true }},
		{name: "explicit child LAZY", goos: "windows", change: func(s *llamaServerRunner, _ *error) { s.launch.extraEnvs[cudaModuleLoadingEnv] = "LAZY" }},
		{name: "explicit child EAGER", goos: "windows", change: func(s *llamaServerRunner, _ *error) { s.launch.extraEnvs[cudaModuleLoadingEnv] = "EAGER" }},
		{name: "explicit case insensitive child", goos: "windows", change: func(s *llamaServerRunner, _ *error) { s.launch.extraEnvs["cuda_module_loading"] = "LAZY" }},
		{name: "cancel error", goos: "windows", change: func(_ *llamaServerRunner, e *error) { *e = context.Canceled }},
		{name: "deadline error", goos: "windows", change: func(_ *llamaServerRunner, e *error) { *e = context.DeadlineExceeded }},
		{name: "no error", goos: "windows", change: func(_ *llamaServerRunner, e *error) { *e = nil }},
		{name: "CUDA OOM is not 303", goos: "windows", change: func(s *llamaServerRunner, e *error) {
			s.status.SetLastError("")
			*e = errors.New("CUDA error: out of memory")
		}},
		{name: "kernel error is not 303 even with stale status", goos: "windows", change: func(_ *llamaServerRunner, e *error) {
			*e = errors.New("CUDA error: an illegal memory access was encountered")
		}},
		{name: "generic module failure is not 303", goos: "windows", change: func(s *llamaServerRunner, e *error) {
			s.status.SetLastError("")
			*e = errors.New("CUDA error: no kernel image is available for execution on the device")
		}},
		{name: "bare numeric error is not enough", goos: "windows", change: func(s *llamaServerRunner, e *error) { s.status.SetLastError(""); *e = errors.New("error 303") }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := cudaRetryTestRunner()
			_, _ = s.status.Write([]byte(cuda303TestLog))
			if !strings.Contains(s.lastErrMsg(), "shared object initialization failed") {
				t.Fatal("real error prefix was not captured")
			}
			var err error = errors.New("llama-server process has terminated")
			if tt.change != nil {
				tt.change(s, &err)
			}
			if got := s.shouldRetryCUDAEagerModuleLoading(context.Background(), err, tt.goos); got != tt.want {
				t.Fatalf("retry=%v, want %v", got, tt.want)
			}
		})
	}
	for _, mode := range []string{"EAGER", "LAZY"} {
		t.Run("explicit parent "+mode, func(t *testing.T) {
			t.Setenv(cudaModuleLoadingEnv, mode)
			s := cudaRetryTestRunner()
			if s.shouldRetryCUDAEagerModuleLoading(context.Background(), errors.New(cuda303TestLog), "windows") {
				t.Fatal("explicit parent CUDA_MODULE_LOADING was overridden")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if cudaRetryTestRunner().shouldRetryCUDAEagerModuleLoading(ctx, errors.New(cuda303TestLog), "windows") {
		t.Fatal("cancelled request retried")
	}
	if isCUDASharedObjectInitFailure(cuda303TestLog + "CUDA error: an illegal memory access was encountered\n") {
		t.Fatal("a later distinct CUDA error was retried as 303")
	}
}

// This helper is a normal Go test child, not llama-server. It never loads CUDA,
// reads a model, installs anything, or touches the production Ollama endpoint.
func TestCUDAEagerRetryProcessHelper(t *testing.T) {
	if os.Getenv("GO_WANT_CUDA_EAGER_RETRY_HELPER") != "1" {
		return
	}
	for {
		time.Sleep(time.Second)
	}
}

func cudaRetryFakeChild(t *testing.T, s *llamaServerRunner, afterReap func()) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestCUDAEagerRetryProcessHelper$")
	cmd.Env = append(os.Environ(), "GO_WANT_CUDA_EAGER_RETRY_HELPER=1")
	cmd.SysProcAttr = LlamaServerSysProcAttr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	s.cmd = cmd
	s.done = make(chan struct{})
	go func() {
		s.doneErr = cmd.Wait()
		if afterReap != nil {
			afterReap()
		}
		close(s.done)
	}()
	t.Cleanup(func() { _ = s.stopProcess() })
}

func TestCUDAEagerRetryLifecycle(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the retry intentionally applies to Windows only; predicates cover other platforms")
	}
	t.Setenv(cudaModuleLoadingEnv, "")
	for _, secondError := range []error{nil, errors.New(cuda303TestLog), errors.New("out of memory")} {
		t.Run(fmt.Sprint(secondError), func(t *testing.T) {
			s := cudaRetryTestRunner()
			s.launch.projectors = []string{"same-projector.gguf"}
			s.launch.modelArch, s.launch.modelLayers, s.launch.mmprojMemory = "qwen3vl", 33, 1<<20
			s.launch.gpus[0].FreeMemory = 24 << 30
			if !s.shouldRetryMMProjCPUOffload(errors.New("out of memory")) {
				t.Fatal("fixture must make the old projector workaround eligible to test that EAGER failure does not trigger it")
			}
			cudaRetryFakeChild(t, s, nil)
			originalEnv := s.launch.extraEnvs
			before := s.launch
			_, _ = s.output.Write([]byte("using device CUDA0 (test) (0000:01:00.0) - 15000 MiB free\n" +
				"load_tensors: CUDA0 model buffer size = 123.00 MiB\n" +
				"wildllama_activity {\"kind\":\"memory\",\"stage\":\"cuda\",\"memory\":{\"vram_bytes\":999}}\n" +
				"partial old buffer line"))
			s.memCPUMappedModel, s.moeRAMWeightLimit, s.gpuLayerOverflow = 1234, 5678, 9
			waits, starts := 0, 0
			wait := func(context.Context) error {
				waits++
				if waits == 1 {
					s.status.SetLastError(cuda303TestLog)
					return errors.New(cuda303TestLog)
				}
				return secondError
			}
			restart := func() error {
				starts++
				select {
				case <-s.done:
				default:
					t.Fatal("replacement started before old process was reaped")
				}
				if s.cmd.ProcessState == nil {
					t.Fatal("old process had no completed Wait")
				}
				if total, gpu := s.MemorySize(); total != 0 || gpu != 0 || s.status.ActivitySnapshot().Memory != nil {
					t.Fatal("old totals or managed VRAM survived the restart")
				}
				if len(s.output.buffers) != 0 || len(s.output.tail) != 0 || s.output.boundedExpertCache != 0 ||
					len(s.vramByDevice) != 0 || len(s.systemFreeAtLoad) != 0 || s.memCPUMappedModel != 0 || s.moeRAMWeightLimit != 0 || s.gpuLayerOverflow != 0 {
					t.Fatal("old parser/accounting state survived the restart")
				}
				if s.lastErrMsg() != "" || s.launch.extraEnvs[cudaModuleLoadingEnv] != "EAGER" || s.launch.extraEnvs["KEEP"] != "same-value" {
					t.Fatal("retry error/env reset was incorrect")
				}
				if _, exists := s.launch.extraEnvs["cuda_module_loading"]; exists {
					t.Fatal("case alias could override EAGER nondeterministically")
				}
				// All model/context/sampling/placement settings remain identical.
				after := s.launch
				after.extraEnvs = before.extraEnvs
				if !reflect.DeepEqual(before, after) {
					t.Fatal("retry changed model configuration")
				}
				if originalEnv[cudaModuleLoadingEnv] != "" || originalEnv["KEEP"] != "same-value" || os.Getenv(cudaModuleLoadingEnv) != "" {
					t.Fatal("retry mutated shared or parent environment")
				}
				return nil
			}
			err := s.waitUntilRunningWithStartupRetries(context.Background(), wait, restart)
			if starts != 1 || waits != 2 || !s.cudaModuleLoadingRetried || s.mmprojOffloadOOMRetried || s.launch.forceNoMMProjOffload {
				t.Fatalf("retry loop/CPU fallback: starts=%d waits=%d", starts, waits)
			}
			if secondError == nil {
				if err != nil || !s.loadReady {
					t.Fatalf("replacement not ready: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "after CUDA_MODULE_LOADING=EAGER retry") || !errors.Is(err, secondError) || s.loadReady {
				t.Fatalf("second failure was not terminal/clear: %v", err)
			}
		})
	}
}

func TestCUDAEagerRetryCancellationAfterReap(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows-only lifecycle")
	}
	t.Setenv(cudaModuleLoadingEnv, "")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s := cudaRetryTestRunner()
	cudaRetryFakeChild(t, s, cancel)
	starts := 0
	retried, err := s.retryWithCUDAEagerModuleLoading(ctx, errors.New(cuda303TestLog), func() error { starts++; return nil })
	if retried || !errors.Is(err, context.Canceled) || starts != 0 || s.launch.extraEnvs[cudaModuleLoadingEnv] != "" {
		t.Fatalf("cancel after reap restarted/changed env: retried=%v starts=%d err=%v", retried, starts, err)
	}
}
