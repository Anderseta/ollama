package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
)

func TestGPUFirstDiagnosticsSurviveFinalCompletion(t *testing.T) {
	const diagnostics = `{"counter_scope":"shared_model_context_delta","cpu_heavy_ops_submitted":0,"gpu_heavy_ops_submitted":42,"ssd":{"bytes":1048576}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"content":"OK","stop":false}`)
		fmt.Fprintf(w, "data: {\"content\":\"\",\"stop\":true,\"timings\":{\"prompt_n\":1,\"predicted_n\":1,\"gpu_first\":%s}}\n", diagnostics)
	}))
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	runner := &llamaServerRunner{port: port, cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1)}
	opts := api.DefaultOptions()
	var final CompletionResponse
	err = runner.Completion(t.Context(), CompletionRequest{Prompt: "test", Options: &opts}, func(response CompletionResponse) {
		if response.Done {
			final = response
		}
		if !response.Done && len(response.GPUFirst) != 0 {
			t.Error("diagnostics attached to an unfinished token")
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	// Exercise the public response encoding as well as parsing the native SSE.
	encoded, err := json.Marshal(api.GenerateResponse{Metrics: api.Metrics{GPUFirst: final.GPUFirst}})
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		GPUFirst json.RawMessage `json:"gpu_first"`
	}
	if err := json.Unmarshal(encoded, &result); err != nil {
		t.Fatal(err)
	}
	if string(result.GPUFirst) != diagnostics {
		t.Fatalf("diagnostics lost: %s", encoded)
	}
}

func TestGPUFirstDiagnosticsAbsentByDefault(t *testing.T) {
	encoded, err := json.Marshal(api.GenerateResponse{})
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &result); err != nil {
		t.Fatal(err)
	}
	if _, ok := result["gpu_first"]; ok {
		t.Fatalf("default diagnostics should be omitted: %s", encoded)
	}
}
