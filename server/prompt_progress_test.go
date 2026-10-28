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

func TestPromptProgressSurvivesAllLocalRoutes(t *testing.T) {
	for _, mode := range []string{"native-chat", "rendered-chat", "generate"} {
		for _, stream := range []bool{true, false} {
			name := mode + "/nonstream"
			if stream {
				name = mode + "/stream"
			}
			t.Run(name, func(t *testing.T) {
				t.Setenv("OLLAMA_MODELS", t.TempDir())
				t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
				t.Setenv("OLLAMA_GO_TEMPLATE", "")
				gin.SetMode(gin.TestMode)
				progress := &api.PromptProgress{Total: 100, Cached: 60, Processed: 0, ElapsedMS: 12}
				cached := 60
				fixture := &mockRunner{
					ChatFn: func(_ context.Context, _ llm.ChatRequest, fn func(llm.ChatResponse)) error {
						fn(llm.ChatResponse{PromptProgress: progress})
						fn(llm.ChatResponse{Message: api.Message{Content: "resposta"}})
						fn(llm.ChatResponse{Done: true, PromptEvalCount: 100, PromptEvalCachedCount: &cached, EvalCount: 1})
						return nil
					},
					CompletionFn: func(_ context.Context, _ llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
						fn(llm.CompletionResponse{PromptProgress: progress})
						fn(llm.CompletionResponse{Content: "resposta"})
						fn(llm.CompletionResponse{Done: true, PromptEvalCount: 100, PromptEvalCachedCount: &cached, EvalCount: 1})
						return nil
					},
				}
				s := newServerWithMockRunner(t, fixture)
				kv := gguftest.KV{"tokenizer.chat_template": "{{ messages[0]['content'] }}"}
				tmpl := ""
				if mode == "rendered-chat" {
					t.Setenv("OLLAMA_GO_TEMPLATE", "1")
					tmpl = `{{range .Messages}}{{.Content}}{{end}}`
				}
				createMinimalGGUFModel(t, s, "prefill-progress", kv, tmpl, nil)
				handler := s.ChatHandler
				var request any = api.ChatRequest{Model: "prefill-progress", Messages: []api.Message{{Role: "user", Content: "hello"}}, Stream: &stream}
				if mode == "generate" {
					handler = s.GenerateHandler
					request = api.GenerateRequest{Model: "prefill-progress", Prompt: "hello", Stream: &stream}
				}
				w := createRequest(t, handler, request)
				if w.Code != http.StatusOK {
					t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
				}
				decoder := json.NewDecoder(w.Body)
				var progressCount, doneCount int
				var text string
				var final api.Metrics
				for {
					var frame struct {
						PromptProgress *api.PromptProgress `json:"prompt_progress"`
						Message        api.Message         `json:"message"`
						Response       string              `json:"response"`
						Done           bool                `json:"done"`
						api.Metrics
					}
					if err := decoder.Decode(&frame); err == io.EOF {
						break
					} else if err != nil {
						t.Fatal(err)
					}
					if frame.PromptProgress != nil {
						progressCount++
						if *frame.PromptProgress != *progress || frame.Done || frame.Message.Content != "" || frame.Response != "" || len(frame.Message.ToolCalls) != 0 {
							t.Fatalf("invalid progress frame: %+v", frame)
						}
					}
					text += frame.Message.Content + frame.Response
					if frame.Done {
						doneCount++
						final = frame.Metrics
					}
				}
				expected := 0
				if stream {
					expected = 1
				}
				if progressCount != expected || doneCount != 1 || text != "resposta" || final.PromptEvalCount != 100 || final.PromptEvalCachedCount == nil || *final.PromptEvalCachedCount != 60 || final.EvalCount != 1 {
					t.Fatalf("progress=%d done=%d text=%q final=%+v", progressCount, doneCount, text, final)
				}
			})
		}
	}
}
