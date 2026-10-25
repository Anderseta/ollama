package llm

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestNativeChatPreservesAssistantReasoningOnly(t *testing.T) {
	for _, role := range []string{"assistant", "user", "system", "tool"} {
		t.Run(role, func(t *testing.T) {
			message := Message{Role: role, Content: "visible answer", Thinking: "  reasoning before a tool\n"}
			converted, err := llamaServerChatMessage(message)
			if err != nil {
				t.Fatal(err)
			}
			reasoning, exists := converted["reasoning_content"]
			if role == "assistant" {
				if !exists || reasoning != message.Thinking {
					t.Fatalf("assistant reasoning was not preserved byte-for-byte: %#v", reasoning)
				}
			} else if exists {
				t.Fatalf("%s must not acquire assistant reasoning", role)
			}
			if converted["content"] != message.Content {
				t.Fatal("reasoning conversion changed visible content")
			}
		})
	}
	converted, err := llamaServerChatMessage(Message{Role: "assistant"})
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := converted["reasoning_content"]; exists {
		t.Fatal("empty reasoning should not override the template's default")
	}
}

func TestNativeChatReasoningSurvivesToolHistorySerialization(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("path", "example.txt")
	opts := api.DefaultOptions()
	runner := &llamaServerRunner{}
	body, err := runner.llamaServerChatRequest(ChatRequest{
		Options: &opts,
		Messages: []api.Message{
			{Role: "user", Content: "inspect the file"},
			{Role: "assistant", Thinking: "First inspect the relevant file.", ToolCalls: []api.ToolCall{{
				ID: "call-1", Function: api.ToolCallFunction{Name: "read", Arguments: args},
			}}},
			{Role: "tool", ToolCallID: "call-1", Content: "example contents"},
		},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Messages []struct {
			Reasoning string            `json:"reasoning_content"`
			Calls     []json.RawMessage `json:"tool_calls"`
			CallID    string            `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Messages[1].Reasoning != "First inspect the relevant file." || len(decoded.Messages[1].Calls) != 1 || decoded.Messages[2].CallID != "call-1" {
		t.Fatal("native tool follow-up lost reasoning, its call, or the matching result")
	}
}

func TestNativeChatTemplateTokenCountCompatibility(t *testing.T) {
	for _, test := range []struct {
		name       string
		response   string
		wantCount  int
		wantTokens int
		wantError  bool
	}{
		{"combined count", `{"prompt":"<s>rendered prompt","token_count":17}`, 17, 0, false},
		{"valid zero", `{"prompt":"","token_count":0}`, 0, 0, false},
		{"older payload", `{"prompt":"<s>rendered prompt"}`, 3, 1, false},
		{"invalid count", `{"prompt":"bad","token_count":-1}`, 0, 0, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			applyCalls, tokenizeCalls := 0, 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch r.URL.Path {
				case "/apply-template":
					applyCalls++
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					if body["count_tokens"] != true {
						t.Error("missing additive count_tokens request")
					}
					_, _ = w.Write([]byte(test.response))
				case "/tokenize":
					tokenizeCalls++
					var body struct {
						Content    string `json:"content"`
						AddSpecial bool   `json:"add_special"`
					}
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					if !body.AddSpecial || body.Content != "<s>rendered prompt" {
						t.Error("fallback must use the unchanged full prompt and generation's BOS policy")
					}
					_, _ = w.Write([]byte(`{"tokens":[1,2,3]}`))
				default:
					t.Errorf("unexpected endpoint %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			runner := &llamaServerRunner{port: server.Listener.Addr().(*net.TCPAddr).Port}
			opts := api.DefaultOptions()
			_, count, err := runner.ApplyChatTemplateWithTokenCount(context.Background(), ChatRequest{
				Options: &opts, Messages: []api.Message{{Role: "user", Content: "hello"}},
			})
			if (err != nil) != test.wantError || count != test.wantCount {
				t.Fatalf("count=%d, err=%v; want count=%d, error=%v", count, err, test.wantCount, test.wantError)
			}
			if applyCalls != 1 || tokenizeCalls != test.wantTokens {
				t.Fatalf("HTTP calls apply=%d tokenize=%d; expected 1 and %d", applyCalls, tokenizeCalls, test.wantTokens)
			}
		})
	}
}

func TestNativeChatTemplateCountDoesNotRetryInvalidTemplate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/apply-template" {
			t.Errorf("invalid template must not be retried through %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid tool history"}}`))
	}))
	defer server.Close()
	runner := &llamaServerRunner{port: server.Listener.Addr().(*net.TCPAddr).Port}
	opts := api.DefaultOptions()
	_, _, err := runner.ApplyChatTemplateWithTokenCount(context.Background(), ChatRequest{Options: &opts})
	if err == nil || !strings.Contains(err.Error(), "invalid tool history") {
		t.Fatalf("template error was hidden: %v", err)
	}
}
