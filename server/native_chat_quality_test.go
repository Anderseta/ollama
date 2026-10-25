package server

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

type nativeChatCountingRunner struct {
	mockRunner
	tokenizeCalls int
}

func (r *nativeChatCountingRunner) Tokenize(ctx context.Context, prompt string) ([]int, error) {
	r.tokenizeCalls++
	return r.mockRunner.Tokenize(ctx, prompt)
}

func nativeChatQualityPrompt(req llm.ChatRequest) string {
	var prompt strings.Builder
	for _, message := range req.Messages {
		prompt.WriteString(message.Role + " " + message.Content + " ")
	}
	return prompt.String()
}

func TestNativeChatTruncationKeepsParallelToolsWithOwner(t *testing.T) {
	for _, withIDs := range []bool{false, true} {
		t.Run(map[bool]string{false: "legacy without IDs", true: "parallel results in reverse order"}[withIDs], func(t *testing.T) {
			messages := []api.Message{
				{Role: "system", Content: "instructions"},
				{Role: "user", Content: "old"},
				{Role: "assistant", Content: "plan", ToolCalls: []api.ToolCall{{ID: "a"}, {ID: "b"}}},
				{Role: "tool", Content: "second"},
				{Role: "tool", Content: "first"},
				{Role: "user", Content: "latest"},
			}
			if withIDs {
				messages[3].ToolCallID, messages[4].ToolCallID = "b", "a"
			}
			calls := 0
			runner := &nativeChatCountingRunner{mockRunner: mockRunner{TemplateFn: func(_ context.Context, req llm.ChatRequest) (string, error) {
				calls++
				owner := false
				for _, message := range req.Messages {
					owner = owner || len(message.ToolCalls) > 0
					if message.Role == "tool" && !owner {
						t.Error("template received an orphan tool result")
					}
				}
				return nativeChatQualityPrompt(req), nil
			}}}
			got, err := truncateNativeChatMessages(context.Background(), nil, runner, &api.Options{Runner: api.Runner{NumCtx: 6}}, llm.ChatRequest{Messages: messages}, true)
			if err != nil {
				t.Fatal(err)
			}
			want := []api.Message{messages[0], messages[5]}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("truncated history=%#v, want system and latest user", got)
			}
			if calls != 3 || runner.tokenizeCalls != 3 {
				t.Fatalf("render/tokenize calls=%d/%d; want three distinct valid candidates", calls, runner.tokenizeCalls)
			}
		})
	}
}

func TestNativeChatOversizedLastToolGroupRetainsItsOwner(t *testing.T) {
	messages := []api.Message{
		{Role: "system", Content: "instructions"},
		{Role: "user", Content: "old"},
		{Role: "assistant", Content: "plan", ToolCalls: []api.ToolCall{{ID: "a"}, {ID: "b"}}},
		{Role: "tool", ToolCallID: "b", Content: "second"},
		{Role: "tool", ToolCallID: "a", Content: "first"},
	}
	runner := &mockRunner{TemplateFn: func(_ context.Context, req llm.ChatRequest) (string, error) {
		return nativeChatQualityPrompt(req), nil
	}}
	got, err := truncateNativeChatMessages(context.Background(), nil, runner, &api.Options{Runner: api.Runner{NumCtx: 2}}, llm.ChatRequest{Messages: messages}, true)
	if err != nil {
		t.Fatal(err)
	}
	if want := append([]api.Message{messages[0]}, messages[2:]...); !reflect.DeepEqual(got, want) {
		t.Fatalf("oversized group must keep its assistant owner and both results: %#v", got)
	}
}

func TestNativeChatCountingIsLocalToOnePreparation(t *testing.T) {
	messages := []api.Message{{Role: "system"}, {Role: "user", Content: "old"}, {Role: "assistant"}, {Role: "user", Content: "latest"}}
	renders := 0
	runner := &nativeChatCountingRunner{mockRunner: mockRunner{TemplateFn: func(_ context.Context, req llm.ChatRequest) (string, error) {
		renders++
		// A legal template may ignore old turns. Count its actual output once.
		return "always the same prompt", nil
	}}}
	for range 2 {
		_, err := truncateNativeChatMessages(context.Background(), nil, runner, &api.Options{Runner: api.Runner{NumCtx: 1}}, llm.ChatRequest{Messages: messages}, true)
		if err != nil {
			t.Fatal(err)
		}
	}
	if renders != 6 || runner.tokenizeCalls != 2 {
		t.Fatalf("renders/tokenizes=%d/%d; expected 3/1 per request, without cross-request prompt retention", renders, runner.tokenizeCalls)
	}
}

type nativeChatCombinedCounter struct {
	mockRunner
	countCalls int
}

func (r *nativeChatCombinedCounter) ApplyChatTemplateWithTokenCount(_ context.Context, req llm.ChatRequest) (string, int, error) {
	r.countCalls++
	return "native exact prompt", 7, nil
}

func TestNativeChatUsesOptionalCombinedCount(t *testing.T) {
	runner := &nativeChatCombinedCounter{mockRunner: mockRunner{TemplateFn: func(context.Context, llm.ChatRequest) (string, error) {
		t.Error("combined count must not call the separate template method")
		return "", nil
	}}}
	messages := []api.Message{{Role: "system", Content: "instructions"}, {Role: "user", Content: "hello"}}
	got, err := truncateNativeChatMessages(context.Background(), nil, runner, &api.Options{Runner: api.Runner{NumCtx: 7}}, llm.ChatRequest{Messages: messages}, true)
	if err != nil || !reflect.DeepEqual(got, messages) || runner.countCalls != 1 {
		t.Fatalf("got=%#v err=%v calls=%d; want one combined count and unchanged history", got, err, runner.countCalls)
	}
}

func TestNativeChatTruncationStopsOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	runner := &mockRunner{TemplateFn: func(context.Context, llm.ChatRequest) (string, error) {
		t.Error("canceled preparation must not render")
		return "", nil
	}}
	_, err := truncateNativeChatMessages(ctx, nil, runner, &api.Options{Runner: api.Runner{NumCtx: 2}}, llm.ChatRequest{Messages: []api.Message{{Role: "user"}, {Role: "assistant"}}}, true)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error=%v; want context cancellation", err)
	}
}
