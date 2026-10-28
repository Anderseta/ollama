package llm

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestNativeSpeculativeMetrics(t *testing.T) {
	for _, chat := range []bool{false, true} {
		for _, test := range []struct {
			name, timings   string
			draft, accepted int
			present         bool
		}{
			{"accepted", `"draft_n":6,"draft_n_accepted":4`, 6, 4, true},
			{"zero_accepted", `"draft_n":6,"draft_n_accepted":0`, 6, 0, true},
			{"absent", `"predicted_n":1`, 0, 0, false},
			{"invalid", `"draft_n":2,"draft_n_accepted":3`, 0, 0, false},
			{"negative", `"draft_n":-1,"draft_n_accepted":0`, 0, 0, false},
			{"incomplete", `"draft_n":5`, 0, 0, false},
		} {
			t.Run(fmt.Sprintf("chat_%v/%s", chat, test.name), func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path == "/health" {
						fmt.Fprint(w, `{"status":"ok"}`)
						return
					}
					w.Header().Set("Content-Type", "text/event-stream")
					if chat {
						fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"OK"},"finish_reason":null}]}`)
						fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{%s}}\n", test.timings)
					} else {
						fmt.Fprintln(w, `data: {"content":"OK","stop":false}`)
						fmt.Fprintf(w, "data: {\"content\":\"\",\"stop\":true,\"timings\":{%s}}\n", test.timings)
					}
				}))
				defer srv.Close()
				u, _ := url.Parse(srv.URL)
				port, _ := strconv.Atoi(u.Port())
				runner := &llamaServerRunner{port: port, cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1), status: NewStatusWriter(io.Discard)}
				opts := api.DefaultOptions()
				var draft, accepted *int
				check := func(done bool, proposed, used *int) {
					if !done && (proposed != nil || used != nil) {
						t.Fatal("draft counters leaked into a partial token")
					}
					if done {
						draft, accepted = proposed, used
					}
				}
				var err error
				if chat {
					err = runner.Chat(t.Context(), ChatRequest{Messages: []api.Message{{Role: "user", Content: "fixture"}}, Options: &opts}, func(r ChatResponse) { check(r.Done, r.SpeculativeDraftTokens, r.SpeculativeAcceptedTokens) })
				} else {
					err = runner.Completion(t.Context(), CompletionRequest{Prompt: "fixture", Options: &opts}, func(r CompletionResponse) { check(r.Done, r.SpeculativeDraftTokens, r.SpeculativeAcceptedTokens) })
				}
				if err != nil {
					t.Fatal(err)
				}
				if test.present {
					if draft == nil || accepted == nil || *draft != test.draft || *accepted != test.accepted {
						t.Fatalf("invalid counters: %v %v", draft, accepted)
					}
				} else if draft != nil || accepted != nil {
					t.Fatal("absent or invalid metrics must stay omitted")
				}
				activity := runner.RuntimeActivity()
				if (activity.SpeculativeDraftTokens != nil) != test.present || (activity.SpeculativeObservedAt > 0) != test.present {
					t.Fatalf("activity disagrees with final metrics: %+v", activity)
				}
			})
		}
	}
}

func TestSpeculativeActivityRejectsOldGenerationAndCopies(t *testing.T) {
	w := NewStatusWriter(io.Discard)
	first := w.beginSpeculativeGeneration()
	second := w.beginSpeculativeGeneration()
	draft, accepted := 8, 0
	w.observeSpeculativeGeneration(first, &draft, &accepted)
	if w.ActivitySnapshot().SpeculativeDraftTokens != nil {
		t.Fatal("an older generation overwrote current activity")
	}
	w.observeSpeculativeGeneration(second, &draft, &accepted)
	snapshot := w.ActivitySnapshot()
	*snapshot.SpeculativeDraftTokens = 100
	if *w.ActivitySnapshot().SpeculativeDraftTokens != 8 {
		t.Fatal("snapshot counters share pointers")
	}
	w.beginSpeculativeGeneration()
	if snapshot := w.ActivitySnapshot(); snapshot.SpeculativeDraftTokens != nil || snapshot.SpeculativeObservedAt != 0 {
		t.Fatal("new request retained stale counters")
	}
}
