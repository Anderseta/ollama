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

func TestNativeChatToolCallTruncationFlag(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, truncated := range []bool{false, true} {
			name := "nonstream"
			if stream {
				name = "stream"
			}
			if truncated {
				name += "-truncated"
			}
			t.Run(name, func(t *testing.T) {
				t.Setenv("OLLAMA_MODELS", t.TempDir())
				t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
				t.Setenv("OLLAMA_GO_TEMPLATE", "")
				gin.SetMode(gin.TestMode)
				fixture := &mockRunner{ChatFn: func(_ context.Context, _ llm.ChatRequest, callback func(llm.ChatResponse)) error {
					callback(llm.ChatResponse{Message: api.Message{Content: "answer", Thinking: "thought"}})
					callback(llm.ChatResponse{Message: api.Message{Role: "assistant"}, Done: true, DoneReason: llm.DoneReasonLength, ToolCallsTruncated: truncated, EvalCount: 7})
					return nil
				}}
				s := newServerWithMockRunner(t, fixture)
				createMinimalGGUFModel(t, s, "terminal-model", gguftest.KV{"tokenizer.chat_template": "{{ messages[0]['content'] }}"}, "", nil)
				w := createRequest(t, s.ChatHandler, api.ChatRequest{Model: "terminal-model", Messages: []api.Message{{Role: "user", Content: "hello"}}, Stream: &stream})
				if w.Code != http.StatusOK {
					t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
				}
				decoder := json.NewDecoder(w.Body)
				var final api.ChatResponse
				var content, thinking string
				for {
					var response api.ChatResponse
					if err := decoder.Decode(&response); err == io.EOF {
						break
					} else if err != nil {
						t.Fatal(err)
					}
					content += response.Message.Content
					thinking += response.Message.Thinking
					if len(response.Message.ToolCalls) != 0 {
						t.Fatalf("unexpected tool calls: %+v", response.Message.ToolCalls)
					}
					final = response
				}
				if !final.Done || final.DoneReason != "length" || final.ToolCallsTruncated != truncated || final.EvalCount != 7 || content != "answer" || thinking != "thought" {
					t.Fatalf("terminal state=%+v content=%q thinking=%q", final, content, thinking)
				}
			})
		}
	}
}
