package server

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/llm"
)

type activityFixture struct{}

func (activityFixture) RuntimeActivity() llm.RuntimeActivity {
	return llm.RuntimeActivity{Phase: "loading", Progress: func() *float64 { v := .5; return &v }()}
}

func TestRuntimeActivityDoesNotWaitForModelLoading(t *testing.T) {
	s := &Server{sched: &Scheduler{loaded: make(map[string]*runnerRef)}}
	s.sched.loadedMu.Lock()
	defer s.sched.loadedMu.Unlock()
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	completed := make(chan struct{})
	go func() { s.RuntimeActivityHandler(c); close(completed) }()
	select {
	case <-completed:
	case <-time.After(time.Second):
		t.Fatal("activity blocked behind model load")
	}
	var result struct {
		Busy bool `json:"scheduler_busy"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil || !result.Busy || recorder.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("bad activity response: %s", recorder.Body.String())
	}
}

func TestRuntimeActivityReportsImmutableRunnerObserver(t *testing.T) {
	runner := &runnerRef{activity: &runnerActivityEntry{reporter: activityFixture{}, model: "fixture:latest", pid: 42, context: 4096}}
	runner.model.Store(&Model{ShortName: "fixture:latest"})
	s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{"private-model-path": runner}}}
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	s.RuntimeActivityHandler(c)
	var result struct {
		Runners []runnerActivityResponse `json:"runners"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil || len(result.Runners) != 1 || result.Runners[0].Activity.Phase != "loading" || result.Runners[0].PID != 42 {
		t.Fatalf("bad runner state: %s", recorder.Body.String())
	}
	runner.model.Store(nil)
	recorder = httptest.NewRecorder()
	c, _ = gin.CreateTestContext(recorder)
	s.RuntimeActivityHandler(c)
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil || len(result.Runners) != 0 {
		t.Fatal("unloaded runner still reported")
	}
}
