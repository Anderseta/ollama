package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
)

func TestSpeculativeMetricsSurviveLocalRoutes(t *testing.T) {
	for _, mode := range []string{"native-chat", "rendered-chat", "generate"} {
		for _, stream := range []bool{false, true} {
			name := mode + "/nonstream"
			if stream {
				name = mode + "/stream"
			}
			t.Run(name, func(t *testing.T) {
				t.Setenv("OLLAMA_MODELS", t.TempDir())
				t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
				t.Setenv("OLLAMA_GO_TEMPLATE", "")
				gin.SetMode(gin.TestMode)
				draft, accepted := 7, 0
				fixture := &mockRunner{
					ChatFn: func(_ context.Context, _ llm.ChatRequest, fn func(llm.ChatResponse)) error {
						fn(llm.ChatResponse{Message: api.Message{Content: "OK"}})
						fn(llm.ChatResponse{Done: true, SpeculativeDraftTokens: &draft, SpeculativeAcceptedTokens: &accepted})
						return nil
					},
					CompletionFn: func(_ context.Context, _ llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
						fn(llm.CompletionResponse{Content: "OK"})
						fn(llm.CompletionResponse{Done: true, SpeculativeDraftTokens: &draft, SpeculativeAcceptedTokens: &accepted})
						return nil
					},
				}
				s := newServerWithMockRunner(t, fixture)
				template := ""
				if mode == "rendered-chat" {
					t.Setenv("OLLAMA_GO_TEMPLATE", "1")
					template = `{{range .Messages}}{{.Content}}{{end}}`
				}
				createMinimalGGUFModel(t, s, "speculative-metrics", gguftest.KV{"tokenizer.chat_template": "{{ messages[0]['content'] }}"}, template, nil)
				handler := s.ChatHandler
				var request any = api.ChatRequest{Model: "speculative-metrics", Messages: []api.Message{{Role: "user", Content: "fixture"}}, Stream: &stream}
				if mode == "generate" {
					handler = s.GenerateHandler
					request = api.GenerateRequest{Model: "speculative-metrics", Prompt: "fixture", Stream: &stream}
				}
				w := createRequest(t, handler, request)
				if w.Code != http.StatusOK {
					t.Fatal(w.Body.String())
				}
				decoder := json.NewDecoder(w.Body)
				var final api.Metrics
				for {
					var frame struct {
						Done bool `json:"done"`
						api.Metrics
					}
					if err := decoder.Decode(&frame); err == io.EOF {
						break
					} else if err != nil {
						t.Fatal(err)
					}
					if frame.Done {
						final = frame.Metrics
					} else if frame.SpeculativeDraftTokens != nil || frame.SpeculativeAcceptedTokens != nil {
						t.Fatal("partial response has terminal draft metrics")
					}
				}
				if final.SpeculativeDraftTokens == nil || final.SpeculativeAcceptedTokens == nil || *final.SpeculativeDraftTokens != draft || *final.SpeculativeAcceptedTokens != 0 {
					t.Fatalf("MTP counters lost: %+v", final)
				}
			})
		}
	}
}
