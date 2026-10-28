package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

// The fake runner is a local HTTP server, not a model or a helper process.
// splitDirs is a sentinel: Close would remove it, which must never happen for
// a clean but incomplete HTTP response.
func terminalTestRunner(t *testing.T, stream string, callback func()) (*llamaServerRunner, *atomic.Int32, string) {
	t.Helper()
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		if r.URL.Path != "/completion" && r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected runner endpoint: %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		requests.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, stream)
		if callback != nil {
			callback()
		}
	}))
	t.Cleanup(server.Close)
	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	sentinel := filepath.Join(t.TempDir(), ".ollama-split-keep")
	if err := os.Mkdir(sentinel, 0o700); err != nil {
		t.Fatal(err)
	}
	runner := &llamaServerRunner{
		port: port, cmd: &exec.Cmd{}, sem: semaphore.NewWeighted(1),
		options:   api.Options{Runner: api.Runner{NumCtx: 2048}},
		splitDirs: []string{sentinel},
	}
	return runner, &requests, sentinel
}

func TestLlamaServerChatRequiresTerminalMarker(t *testing.T) {
	for _, tc := range []struct{ name, stream, content string }{
		{"empty", "", ""},
		{"partial", "data: {\"choices\":[{\"delta\":{\"content\":\"partial text\"},\"finish_reason\":null}]}\n\n", "partial text"},
		{"done_without_finish_reason", "data: {\"choices\":[{\"delta\":{\"content\":\"partial text\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n", "partial text"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runner, requests, sentinel := terminalTestRunner(t, tc.stream, nil)
			var content strings.Builder
			opts := api.DefaultOptions()
			err := runner.Chat(t.Context(), ChatRequest{Options: &opts}, func(response ChatResponse) {
				if response.Done {
					t.Error("incomplete response declared done")
				}
				content.WriteString(response.Message.Content)
			})
			if err == nil || !strings.Contains(err.Error(), "chat stream ended without a completion marker") {
				t.Fatalf("Chat error = %v", err)
			}
			if content.String() != tc.content {
				t.Fatalf("partial output = %q, want %q", content.String(), tc.content)
			}
			if requests.Load() != 1 {
				t.Fatalf("request count = %d, want one without retry", requests.Load())
			}
			if _, err := os.Stat(sentinel); err != nil {
				t.Fatalf("runner resources removed: %v", err)
			}
			if !runner.sem.TryAcquire(1) {
				t.Fatal("request semaphore was not released")
			}
			runner.sem.Release(1)
		})
	}
}

func TestLlamaServerCompletionRequiresTerminalMarker(t *testing.T) {
	for _, tc := range []struct{ name, stream, content string }{
		{"empty", "", ""},
		{"partial", "data: {\"content\":\"partial text\",\"stop\":false}\n\n", "partial text"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runner, requests, sentinel := terminalTestRunner(t, tc.stream, nil)
			var content strings.Builder
			opts := api.DefaultOptions()
			err := runner.Completion(t.Context(), CompletionRequest{Prompt: "input", Options: &opts}, func(response CompletionResponse) {
				if response.Done {
					t.Error("incomplete response declared done")
				}
				content.WriteString(response.Content)
			})
			if err == nil || !strings.Contains(err.Error(), "completion stream ended without a completion marker") {
				t.Fatalf("Completion error = %v", err)
			}
			if content.String() != tc.content {
				t.Fatalf("partial output = %q, want %q", content.String(), tc.content)
			}
			if requests.Load() != 1 {
				t.Fatalf("request count = %d, want one without retry", requests.Load())
			}
			if _, err := os.Stat(sentinel); err != nil {
				t.Fatalf("runner resources removed: %v", err)
			}
			if !runner.sem.TryAcquire(1) {
				t.Fatal("request semaphore was not released")
			}
			runner.sem.Release(1)
		})
	}
}

func TestLlamaServerMissingTerminalPreservesCancellation(t *testing.T) {
	for _, chat := range []bool{false, true} {
		t.Run(strconv.FormatBool(chat), func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			runner, _, sentinel := terminalTestRunner(t, "", cancel)
			opts := api.DefaultOptions()
			var err error
			if chat {
				err = runner.Chat(ctx, ChatRequest{Options: &opts}, func(ChatResponse) {})
			} else {
				err = runner.Completion(ctx, CompletionRequest{Prompt: "input", Options: &opts}, func(CompletionResponse) {})
			}
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("error = %v, want context.Canceled", err)
			}
			if _, err := os.Stat(sentinel); err != nil {
				t.Fatalf("runner resources removed: %v", err)
			}
		})
	}
}

func TestLlamaServerChatToolCallLimit(t *testing.T) {
	for _, tc := range []struct {
		name, reason, secondArguments string
		wantCalls                     int
		truncated, wantError          bool
	}{
		{"valid_stop", "stop", `{"path":"complete"}`, 2, false, false},
		{"valid_length", "length", `{"path":"complete"}`, 2, false, false},
		{"valid_max_tokens", "max_tokens", `{"path":"complete"}`, 2, false, false},
		{"valid_empty_object_length", "length", `{}`, 2, false, false},
		{"missing_arguments_length_discards_entire_batch", "length", ``, 0, true, false},
		{"blank_arguments_max_tokens_discards_entire_batch", "max_tokens", " \n\t", 0, true, false},
		{"null_arguments_length_discards_entire_batch", "length", `null`, 0, true, false},
		{"incomplete_length_discards_entire_batch", "length", `{"path":"do-not-expose`, 0, true, false},
		{"incomplete_max_tokens_discards_entire_batch", "max_tokens", `{"path":"do-not-expose`, 0, true, false},
		{"invalid_stop_is_error", "stop", `{"path":"do-not-expose`, 0, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A complete call arrives before an incomplete sibling. None of the
			// batch may reach the caller until all arguments are validated.
			var stream strings.Builder
			for i, arguments := range []string{`{"path":"first"}`, tc.secondArguments} {
				delta, err := json.Marshal(map[string]any{"choices": []any{map[string]any{
					"delta": map[string]any{"tool_calls": []any{map[string]any{
						"index": i, "id": fmt.Sprintf("call_%d", i), "type": "function",
						"function": map[string]any{"name": "write_file", "arguments": arguments},
					}}},
					"finish_reason": nil,
				}}})
				if err != nil {
					t.Fatal(err)
				}
				fmt.Fprintf(&stream, "data: %s\n\n", delta)
			}
			fmt.Fprintf(&stream, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\",\"reasoning_content\":\"thought\"},\"finish_reason\":%q}],\"timings\":{\"cache_n\":3,\"prompt_n\":5,\"prompt_ms\":12,\"predicted_n\":7,\"predicted_ms\":23}}\n\ndata: [DONE]\n\n", tc.reason)
			runner, requests, sentinel := terminalTestRunner(t, stream.String(), nil)
			opts := api.DefaultOptions()
			var responses []ChatResponse
			err := runner.Chat(t.Context(), ChatRequest{Options: &opts}, func(response ChatResponse) {
				if !response.Done && len(response.Message.ToolCalls) != 0 {
					t.Error("tool call leaked before final batch validation")
				}
				responses = append(responses, response)
			})
			if tc.wantError {
				if err == nil || !strings.Contains(err.Error(), "invalid tool call arguments") {
					t.Fatalf("Chat error = %v, want invalid arguments", err)
				}
				if len(responses) != 0 {
					t.Fatalf("invalid stop batch emitted responses: %+v", responses)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				if len(responses) != 1 || !responses[0].Done {
					t.Fatalf("terminal responses = %+v, want one", responses)
				}
				response := responses[0]
				wantReason := DoneReasonLength
				if tc.reason == "stop" {
					wantReason = DoneReasonStop
				}
				if response.DoneReason != wantReason || response.ToolCallsTruncated != tc.truncated || len(response.Message.ToolCalls) != tc.wantCalls {
					t.Fatalf("terminal batch = %+v; want reason=%v truncated=%v calls=%d", response, wantReason, tc.truncated, tc.wantCalls)
				}
				if response.Message.Content != "answer" || response.Message.Thinking != "thought" || response.PromptEvalCount != 8 || response.PromptEvalCachedCount == nil || *response.PromptEvalCachedCount != 3 || response.EvalCount != 7 || response.PromptEvalDuration != 12*time.Millisecond || response.EvalDuration != 23*time.Millisecond {
					t.Fatalf("content or metrics lost: %+v", response)
				}
				wire, err := json.Marshal(response)
				if err != nil {
					t.Fatal(err)
				}
				if tc.truncated && strings.Contains(string(wire), "do-not-expose") {
					t.Fatalf("partial arguments exposed: %s", wire)
				}
				if !tc.truncated && strings.Contains(string(wire), "tool_calls_truncated") {
					t.Fatalf("false truncation flag was not omitted: %s", wire)
				}
			}
			if requests.Load() != 1 {
				t.Fatalf("request count = %d, want one", requests.Load())
			}
			if _, err := os.Stat(sentinel); err != nil {
				t.Fatalf("runner resources removed: %v", err)
			}
		})
	}
}

func TestLlamaServerCompletionTerminalMarker(t *testing.T) {
	for _, stopType := range []string{"eos", "limit"} {
		t.Run(stopType, func(t *testing.T) {
			stream := fmt.Sprintf("data: {\"content\":\"answer\",\"stop\":true,\"stop_type\":%q}\n\n", stopType)
			runner, _, _ := terminalTestRunner(t, stream, nil)
			opts := api.DefaultOptions()
			var responses []CompletionResponse
			err := runner.Completion(t.Context(), CompletionRequest{Prompt: "input", Options: &opts}, func(response CompletionResponse) {
				responses = append(responses, response)
			})
			if err != nil {
				t.Fatal(err)
			}
			wantReason := DoneReasonStop
			if stopType == "limit" {
				wantReason = DoneReasonLength
			}
			if len(responses) != 1 || !responses[0].Done || responses[0].DoneReason != wantReason || responses[0].Content != "answer" {
				t.Fatalf("terminal responses = %+v, want done/%v", responses, wantReason)
			}
		})
	}
}

func TestLlamaServerCompletionRepeatGuard(t *testing.T) {
	for _, tc := range []struct {
		name            string
		pieces          []string
		count           int
		terminalContent string
		wantError       bool
	}{
		{"empty_events", []string{""}, 150, "answer", false},
		{"distinct_whitespace", []string{" ", "\n", "\t"}, 150, "answer", false},
		{"terminal_does_not_count", []string{"token"}, 101, "token", false},
		{"actual_repetition_is_limited", []string{"token"}, 102, "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stream strings.Builder
			for i := 0; i < tc.count; i++ {
				fmt.Fprintf(&stream, "data: {\"content\":%q,\"stop\":false}\n\n", tc.pieces[i%len(tc.pieces)])
			}
			fmt.Fprintf(&stream, "data: {\"content\":%q,\"stop\":true,\"stop_type\":\"eos\"}\n\n", tc.terminalContent)
			runner, requests, sentinel := terminalTestRunner(t, stream.String(), nil)
			opts := api.DefaultOptions()
			var done bool
			err := runner.Completion(t.Context(), CompletionRequest{Prompt: "input", Options: &opts}, func(response CompletionResponse) {
				if response.Done {
					done = true
					if response.DoneReason != DoneReasonStop || response.Content != tc.terminalContent {
						t.Fatalf("terminal response = %+v", response)
					}
				}
			})
			if tc.wantError {
				if err == nil || !strings.Contains(err.Error(), "token repeat limit reached") || done {
					t.Fatalf("repeat error=%v done=%v", err, done)
				}
			} else if err != nil || !done {
				t.Fatalf("valid completion error=%v done=%v", err, done)
			}
			if requests.Load() != 1 {
				t.Fatalf("request count = %d, want one", requests.Load())
			}
			if _, err := os.Stat(sentinel); err != nil {
				t.Fatalf("runner resources removed: %v", err)
			}
		})
	}
}
