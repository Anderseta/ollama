package llm

import (
	"github.com/ollama/ollama/api"
	"testing"
)

func TestNativeChatReasoningBudgetDoesNotChangeThinking(t *testing.T) {
	runner := &llamaServerRunner{}
	for _, budget := range []int{-1, 0, 1024} {
		opts := api.DefaultOptions()
		opts.ReasoningBudget = &budget
		body, err := runner.llamaServerChatRequest(ChatRequest{Options: &opts, Messages: []api.Message{{Role: "user", Content: "hello"}}}, false)
		if err != nil {
			t.Fatal(err)
		}
		if body["reasoning_budget_tokens"] != budget {
			t.Fatal("explicit budget lost")
		}
		if _, exists := body["chat_template_kwargs"]; exists {
			t.Fatal("setting a budget changed thinking mode")
		}
		if body["n_predict"] != opts.NumPredict {
			t.Fatal("thinking budget changed total prediction limit")
		}
	}
	bad := -2
	opts := api.DefaultOptions()
	opts.ReasoningBudget = &bad
	if _, err := runner.llamaServerChatRequest(ChatRequest{Options: &opts}, false); err == nil {
		t.Fatal("direct caller bypassed budget validation")
	}
}
