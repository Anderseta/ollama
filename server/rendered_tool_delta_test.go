package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
)

func TestRenderedChatPreservesTextAndToolsInSameDelta(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, terminal := range []bool{false, true} {
			name := "nonstream"
			if stream {
				name = "stream"
			}
			if terminal {
				name += "-terminal"
			}
			t.Run(name, func(t *testing.T) {
				t.Setenv("OLLAMA_MODELS", t.TempDir())
				t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
				t.Setenv("OLLAMA_GO_TEMPLATE", "1")
				gin.SetMode(gin.TestMode)
				fixture := &mockRunner{CompletionFn: func(_ context.Context, _ llm.CompletionRequest, callback func(llm.CompletionResponse)) error {
					callback(llm.CompletionResponse{Content: `Vou consultar. <tool_call>{"name":"lookup","arguments":{"query":"fonte completa"}}</tool_call>`, Done: terminal, DoneReason: llm.DoneReasonStop})
					if !terminal {
						callback(llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop})
					}
					return nil
				}}
				s := newServerWithMockRunner(t, fixture)
				tmpl := `{{range .Messages}}{{.Content}}{{if .ToolCalls}}<tool_call>{{range .ToolCalls}}{"name":"{{.Function.Name}}","arguments":{{.Function.Arguments}}}{{end}}</tool_call>{{end}}{{end}}{{if .Tools}}tools{{end}}`
				createMinimalGGUFModel(t, s, "mixed-tool-delta", gguftest.KV{}, tmpl, nil)
				w := createRequest(t, s.ChatHandler, api.ChatRequest{
					Model: "mixed-tool-delta", Messages: []api.Message{{Role: "user", Content: "pesquise"}}, Stream: &stream,
					Tools: []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "lookup", Parameters: api.ToolFunctionParameters{Type: "object"}}}},
				})
				if w.Code != http.StatusOK {
					t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
				}
				decoder := json.NewDecoder(w.Body)
				var content strings.Builder
				var calls []api.ToolCall
				var final api.ChatResponse
				for {
					var response api.ChatResponse
					if err := decoder.Decode(&response); err == io.EOF {
						break
					} else if err != nil {
						t.Fatal(err)
					}
					content.WriteString(response.Message.Content)
					calls = append(calls, response.Message.ToolCalls...)
					final = response
				}
				if !final.Done || !strings.Contains(content.String(), "Vou consultar.") || strings.Contains(content.String(), "<tool_call>") || len(calls) != 1 {
					t.Fatalf("content=%q calls=%+v terminal=%v", content.String(), calls, final.Done)
				}
				query, ok := calls[0].Function.Arguments.Get("query")
				if calls[0].Function.Name != "lookup" || calls[0].ID == "" || !ok || query != "fonte completa" {
					t.Fatalf("tool identity or arguments lost: %+v", calls[0])
				}
			})
		}
	}
}
