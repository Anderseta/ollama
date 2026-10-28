package llm

import (
	"context"
	"net/http"
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestRunnerProcessExitHelper(t *testing.T) {
	if os.Getenv("WILDLLAMA_TEST_PROCESS_EXIT") != "1" {
		return
	}
	time.Sleep(30 * time.Millisecond)
}

func TestRunnerHealthAndExitConcurrent(t *testing.T) {
	runner := promptProgressRunner(t, func(http.ResponseWriter, *http.Request) { t.Error("unexpected inference") })
	cmd := exec.Command(os.Args[0], "-test.run=^TestRunnerProcessExitHelper$")
	cmd.Env = append(os.Environ(), "WILDLLAMA_TEST_PROCESS_EXIT=1")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	runner.cmd = cmd
	runner.done = make(chan struct{})
	t.Cleanup(func() { _ = cmd.Process.Kill(); <-runner.done })
	go func() { _ = cmd.Wait(); close(runner.done) }()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	for {
		_ = runner.Ping(ctx)
		_ = runner.HasExited()
		select {
		case <-runner.done:
			if !runner.HasExited() {
				t.Fatal("exit not visible after process was reaped")
			}
			return
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		default:
		}
	}
}
