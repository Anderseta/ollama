package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os/exec"
	"strconv"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func promptProgressRunner(t *testing.T, handler http.HandlerFunc) *llamaServerRunner {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		if request["return_progress"] != true || request["stream"] != true {
			t.Errorf("native request did not enable streaming progress: %+v", request)
		}
		handler(w, r)
	}))
	t.Cleanup(server.Close)
	u, _ := url.Parse(server.URL)
	port, _ := strconv.Atoi(u.Port())
	return &llamaServerRunner{port: port, cmd: &exec.Cmd{}, sem: semaphore.NewWeighted(1), options: api.Options{Runner: api.Runner{NumCtx: 2048}}}
}

func TestLlamaServerPromptProgressAndFinalMetrics(t *testing.T) {
	for _, chat := range []bool{false, true} {
		t.Run(map[bool]string{false: "completion", true: "chat"}[chat], func(t *testing.T) {
			runner := promptProgressRunner(t, func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				prefix := `"content":"", "stop":false`
				if chat {
					prefix = `"choices":[{"delta":{"content":null},"finish_reason":null}]`
				}
				for _, p := range []string{
					`{"total":100,"cache":60,"processed":60,"time_ms":0}`,
					`{"total":100,"cache":60,"processed":80,"time_ms":10}`,
					`{"total":100,"cache":60,"processed":100,"time_ms":20}`,
				} {
					fmt.Fprintf(w, "data: {%s,\"prompt_progress\":%s}\n\n", prefix, p)
				}
				if chat {
					fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"resposta\"},\"finish_reason\":\"stop\"}],\"timings\":{\"cache_n\":60,\"prompt_n\":40,\"predicted_n\":1}}\n\ndata: [DONE]\n\n")
				} else {
					fmt.Fprint(w, "data: {\"content\":\"resposta\",\"stop\":true,\"timings\":{\"cache_n\":60,\"prompt_n\":40,\"predicted_n\":1}}\n\n")
				}
			})
			var measurements []*api.PromptProgress
			var text string
			var finalCount, promptCount, cachedCount, evalCount int
			check := func(p *api.PromptProgress, content string, done bool, pc int, cc *int, ec int) {
				if p != nil {
					if done || content != "" || pc != 0 || cc != nil || ec != 0 {
						t.Error("prefill invented output/final metrics")
					}
					measurements = append(measurements, p)
				}
				text += content
				if done {
					finalCount++
					promptCount = pc
					if cc != nil {
						cachedCount = *cc
					}
					evalCount = ec
				}
			}
			opts := api.DefaultOptions()
			var err error
			if chat {
				err = runner.Chat(t.Context(), ChatRequest{Options: &opts}, func(r ChatResponse) {
					check(r.PromptProgress, r.Message.Content, r.Done, r.PromptEvalCount, r.PromptEvalCachedCount, r.EvalCount)
				})
			} else {
				err = runner.Completion(t.Context(), CompletionRequest{Prompt: "input", Options: &opts}, func(r CompletionResponse) {
					check(r.PromptProgress, r.Content, r.Done, r.PromptEvalCount, r.PromptEvalCachedCount, r.EvalCount)
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(measurements) != 2 || measurements[0].Processed != 0 || measurements[1].Processed != 40 || measurements[1].Cached != 60 || measurements[1].ElapsedMS != 20 {
				t.Fatalf("progress = %+v", measurements)
			}
			if text != "resposta" || finalCount != 1 || promptCount != 100 || cachedCount != 60 || evalCount != 1 {
				t.Fatalf("text=%q finals=%d prompt=%d cached=%d eval=%d", text, finalCount, promptCount, cachedCount, evalCount)
			}
		})
	}
}

func TestLlamaServerPrefillCancellationReleasesSlot(t *testing.T) {
	for _, chat := range []bool{false, true} {
		t.Run(map[bool]string{false: "completion", true: "chat"}[chat], func(t *testing.T) {
			serverCancelled := make(chan struct{})
			runner := promptProgressRunner(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: {\"content\":\"\",\"stop\":false,\"choices\":[],\"prompt_progress\":{\"total\":100000,\"cache\":0,\"processed\":0,\"time_ms\":0}}\n\n")
				w.(http.Flusher).Flush()
				<-r.Context().Done()
				close(serverCancelled)
			})
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			done := make(chan error, 1)
			seen := make(chan struct{}, 1)
			opts := api.DefaultOptions()
			go func() {
				if chat {
					done <- runner.Chat(ctx, ChatRequest{Options: &opts}, func(r ChatResponse) {
						if r.Done {
							t.Error("cancelled prefill declared complete")
						}
						if r.PromptProgress != nil {
							seen <- struct{}{}
						}
					})
				} else {
					done <- runner.Completion(ctx, CompletionRequest{Prompt: "input", Options: &opts}, func(r CompletionResponse) {
						if r.Done {
							t.Error("cancelled prefill declared complete")
						}
						if r.PromptProgress != nil {
							seen <- struct{}{}
						}
					})
				}
			}()
			select {
			case <-seen:
			case <-time.After(time.Second):
				t.Fatal("no progress before slow prefill finished")
			}
			cancel()
			select {
			case err := <-done:
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancel error = %v", err)
				}
			case <-time.After(time.Second):
				t.Fatal("cancelled prefill retained request slot")
			}
			select {
			case <-serverCancelled:
			case <-time.After(time.Second):
				t.Fatal("native request was not cancelled")
			}
			if !runner.sem.TryAcquire(1) {
				t.Fatal("prefill cancellation leaked semaphore")
			}
			runner.sem.Release(1)
		})
	}
}

func TestPromptProgressRejectsInvalidMeasurements(t *testing.T) {
	for _, p := range []*llamaServerPromptProgress{nil, {}, {Total: 100, Cache: -1}, {Total: 100, Processed: 101}, {Total: 100, Cache: 60, Processed: 41}, {Total: 100, TimeMS: -1}} {
		var emitter llamaServerProgressEmitter
		if emitter.next(p) != nil {
			t.Fatalf("invalid progress accepted: %+v", p)
		}
	}
}
