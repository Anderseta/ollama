package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestAccumulatedToolCallsSparseIndices(t *testing.T) {
	accs := map[int]*llamaServerToolCallAccumulator{}
	for _, index := range []int{math.MaxInt, 3, 0} {
		acc := &llamaServerToolCallAccumulator{index: index}
		acc.append(fmt.Sprint(index), "empty", " \n")
		accs[index] = acc
	}
	accs[5] = nil // An absent call does not alter the ordering of real calls.
	calls, err := accumulatedToolCalls(accs, true)
	if err != nil {
		t.Fatal(err)
	}
	var indices []int
	for _, call := range calls {
		indices = append(indices, call.Function.Index)
	}
	if !reflect.DeepEqual(indices, []int{0, 3, math.MaxInt}) {
		t.Fatalf("tool call indices = %v", indices)
	}
	if _, err := accumulatedToolCalls(map[int]*llamaServerToolCallAccumulator{-1: {}}, true); err == nil {
		t.Fatal("negative tool call index was accepted")
	}
}

func TestAccumulatedToolCallsRequireCompleteArgumentsAtLimit(t *testing.T) {
	for _, tc := range []struct {
		name, function, arguments string
		allowMissing, wantError   bool
	}{
		{"legacy empty at normal stop", "lookup", " \n", true, false},
		{"explicit empty object at length", "lookup", " {} ", false, false},
		{"missing object at length", "lookup", " \n", false, true},
		{"null is not an object", "lookup", "null", true, true},
		{"array is not an object", "lookup", "[]", false, true},
		{"incomplete object", "lookup", `{"query":`, false, true},
		{"missing function name", "", "{}", false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			acc := &llamaServerToolCallAccumulator{}
			acc.append("call_0", tc.function, tc.arguments)
			calls, err := accumulatedToolCalls(map[int]*llamaServerToolCallAccumulator{0: acc}, tc.allowMissing)
			if (err != nil) != tc.wantError || (err == nil && len(calls) != 1) || (err != nil && calls != nil) {
				t.Fatalf("calls=%+v error=%v, want error=%v", calls, err, tc.wantError)
			}
		})
	}
}

func TestLlamaServerChatFragmentedToolCalls(t *testing.T) {
	contents := []string{strings.Repeat("texto com ação, ", 4096), ""}
	argumentRunes := make([][]rune, len(contents))
	for index, content := range contents {
		encoded, err := json.Marshal(map[string]string{"content": content})
		if err != nil {
			t.Fatal(err)
		}
		argumentRunes[index] = []rune(string(encoded))
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			// Deliberately interleave two calls, split IDs and names, and send a
			// large argument in small UTF-8-safe fragments as SSE requires.
			for offset := 0; offset < len(argumentRunes[0]); offset += 7 {
				for index, runes := range argumentRunes {
					if offset >= len(runes) {
						continue
					}
					delta := llamaServerChatToolCall{Index: index * 17}
					if offset == 0 {
						delta.ID = "call_"
						delta.Function.Name = "write_"
					} else if offset == 7 {
						delta.ID = fmt.Sprint(index)
						delta.Function.Name = "file"
					}
					delta.Function.Arguments = string(runes[offset:min(offset+7, len(runes))])
					event, err := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []llamaServerChatToolCall{delta}}}}})
					if err != nil {
						t.Error(err)
						return
					}
					fmt.Fprintf(w, "data: %s\n\n", event)
				}
			}
			fmt.Fprintln(w, `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`)
			fmt.Fprintln(w, `data: [DONE]`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()
	parts := strings.Split(srv.URL, ":")
	var port int
	fmt.Sscanf(parts[len(parts)-1], "%d", &port)
	runner := &llamaServerRunner{port: port, cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1), options: api.Options{Runner: api.Runner{NumCtx: 2048}}}
	var responses []ChatResponse
	err := runner.Chat(t.Context(), ChatRequest{Messages: []api.Message{{Role: "user", Content: "write two files"}}}, func(response ChatResponse) {
		responses = append(responses, response)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(responses) != 1 || !responses[0].Done || len(responses[0].Message.ToolCalls) != len(contents) {
		t.Fatalf("unexpected response count or final tools: %d", len(responses))
	}
	for index, call := range responses[0].Message.ToolCalls {
		if call.ID != fmt.Sprintf("call_%d", index) || call.Function.Name != "write_file" || call.Function.Index != index*17 {
			t.Errorf("incorrect reconstructed identity at call %d", index)
		}
		value, ok := call.Function.Arguments.Get("content")
		if !ok || value != contents[index] {
			t.Errorf("incorrect reconstructed content at call %d", index)
		}
	}
}

// Arguments can contain complete source files, so exercise token-sized deltas
// with payloads substantially larger than a typical short function argument.
func BenchmarkToolCallAccumulator(b *testing.B) {
	for _, size := range []int{4 << 10, 64 << 10, 256 << 10} {
		b.Run(fmt.Sprintf("%dKiB", size>>10), func(b *testing.B) {
			payload := `{"content":"` + strings.Repeat("x", size) + `"}`
			var deltas []llamaServerChatToolCall
			for offset := 0; offset < len(payload); offset += 8 {
				delta := llamaServerChatToolCall{}
				delta.Function.Arguments = payload[offset:min(offset+8, len(payload))]
				deltas = append(deltas, delta)
			}
			deltas[0].ID = "call_write"
			deltas[0].Function.Name = "write_file"
			b.SetBytes(int64(len(payload)))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				acc := &llamaServerToolCallAccumulator{}
				for _, delta := range deltas {
					acc.append(delta.ID, delta.Function.Name, delta.Function.Arguments)
				}
				calls, err := accumulatedToolCalls(map[int]*llamaServerToolCallAccumulator{0: acc}, true)
				if err != nil || len(calls) != 1 {
					b.Fatalf("failed to reconstruct tool call: %v", err)
				}
			}
		})
	}
}
