package server

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPromptTruncationKeepsParallelToolResultsWithCall(t *testing.T) {
	messages := []api.Message{
		{Role: "user", Content: "question"},
		{Role: "assistant", ToolCalls: []api.ToolCall{{ID: "a"}, {ID: "b"}}},
		{Role: "tool", ToolCallID: "b", Content: "second result"},
		{Role: "tool", ToolCallID: "a", Content: "first result"},
		{Role: "assistant", Content: "answer"},
		{Role: "user", Content: "next"},
	}
	starts := promptTruncationStarts(messages)
	for i, want := range []bool{true, true, false, false, true, true} {
		if starts[i] != want {
			t.Fatalf("cut %d: got %v, want %v", i, starts[i], want)
		}
	}
}
