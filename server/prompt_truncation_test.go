package server

import (
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func longTruncationHistory() []api.Message {
	messages := []api.Message{{Role: "system", Content: "Keep these instructions."}}
	for i := 0; i < 512; i++ {
		messages = append(messages, api.Message{Role: "user", Content: fmt.Sprintf("Question %d", i)}, api.Message{Role: "assistant", Content: "Answer"})
	}
	return append(messages, api.Message{Role: "user", Content: "Latest question"})
}

func TestFittingPromptStartMatchesLinearWithFewerCounts(t *testing.T) {
	messages := longTruncationHistory()
	for _, budget := range []int{1, 4, 15, 100, 900, 2000} {
		linearCalls, fastCalls := 0, 0
		linear := func(i int) (int, error) { linearCalls++; return len(promptMessagesFrom(messages, i)), nil }
		fast := func(i int) (int, error) { fastCalls++; return len(promptMessagesFrom(messages, i)), nil }
		want, err := fittingPromptStart(t.Context(), messages, budget, -1, linear)
		if err != nil {
			t.Fatal(err)
		}
		got, err := fittingPromptStart(t.Context(), messages, budget, len(messages)-1, fast)
		if err != nil || got != want {
			t.Fatalf("budget=%d start=%d want=%d err=%v", budget, got, want, err)
		}
		if fastCalls > 13 {
			t.Fatalf("budget=%d excessive counts=%d", budget, fastCalls)
		}
		t.Logf("budget=%d exact sizing calls: %d -> %d; same retained messages", budget, linearCalls, fastCalls)
	}
}

func TestNativeLongTruncationMatchesLinearForReviewedTemplates(t *testing.T) {
	for _, fixture := range []struct{ name, path string }{
		{"qwen3.5", "testdata/qwen35_chat_template.jinja"},
		{"qwen3.8", "../model/renderers/testdata/qwen38_chat_template.jinja"},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			source, err := os.ReadFile(fixture.path)
			if err != nil {
				t.Fatal(err)
			}
			m := &Model{Config: testConfigWithRenderer(fixture.name), metadata: ggufMetadata{KV: map[string]any{"tokenizer.chat_template": string(source), "general.architecture": "qwen35"}}}
			messages := longTruncationHistory()
			// Include parallel tools, whose results arrive in reverse order.
			messages[100].ToolCalls = []api.ToolCall{{ID: "one"}, {ID: "two"}}
			messages[101] = api.Message{Role: "tool", ToolCallID: "two", Content: "second result"}
			messages[102] = api.Message{Role: "tool", ToolCallID: "one", Content: "first result"}
			original := slices.Clone(messages)
			if monotonicPromptLimit(m, messages, true) < 0 || monotonicPromptLimit(m, messages, false) < 0 {
				t.Fatal("reviewed template was not recognized")
			}
			for _, budget := range []int{120, 500, 1700} {
				counts := 0
				runner := &mockRunner{TemplateFn: func(ctx context.Context, req llm.ChatRequest) (string, error) {
					counts++
					return renderPrompt(m, req.Messages, req.Tools, nil)
				}}
				opts := &api.Options{Runner: api.Runner{NumCtx: budget}}
				req := llm.ChatRequest{Messages: messages}
				want, err := truncateNativeChatMessages(t.Context(), nil, runner, opts, req, true)
				if err != nil {
					t.Fatal(err)
				}
				linear := counts
				counts = 0
				got, err := truncateNativeChatMessages(t.Context(), m, runner, opts, req, true)
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("budget=%d different retained history, err=%v", budget, err)
				}
				if counts > 13 || counts >= linear {
					t.Fatalf("counts fast/linear=%d/%d", counts, linear)
				}
				if !reflect.DeepEqual(messages, original) {
					t.Fatal("preparation mutated original messages")
				}
				t.Logf("budget=%d render/tokenize calls %d -> %d", budget, linear, counts)
			}
		})
	}
}

func TestUnknownTemplateKeepsEarliestNonMonotonicFit(t *testing.T) {
	messages := longTruncationHistory()
	runner := &mockRunner{TemplateFn: func(_ context.Context, req llm.ChatRequest) (string, error) {
		if len(req.Messages) == len(messages)-2 {
			return "fits", nil
		}
		return strings.Repeat("too large ", 100), nil
	}}
	m := &Model{metadata: ggufMetadata{KV: map[string]any{"tokenizer.chat_template": "custom non-monotonic template"}}}
	got, err := truncateNativeChatMessages(t.Context(), m, runner, &api.Options{Runner: api.Runner{NumCtx: 4}}, llm.ChatRequest{Messages: messages}, true)
	if err != nil || !reflect.DeepEqual(got, promptMessagesFrom(messages, 3)) {
		t.Fatalf("must retain earliest fit in arbitrary template: err=%v length=%d", err, len(got))
	}
}

func TestMonotonicPromptLimitRejectsUnreviewedInputs(t *testing.T) {
	m := &Model{Config: testConfigWithRenderer("qwen3.5")}
	messages := longTruncationHistory()
	if monotonicPromptLimit(m, messages, false) < 0 {
		t.Fatal("expected reviewed renderer")
	}
	for _, message := range []api.Message{{Role: "system"}, {Role: "developer"}, {Role: "user", Images: []api.ImageData{[]byte("image")}}} {
		changed := slices.Clone(messages)
		changed[10] = message
		if monotonicPromptLimit(m, changed, false) >= 0 {
			t.Fatalf("unsafe fast path for %#v", message)
		}
	}
	if monotonicPromptLimit(m, messages, true) >= 0 {
		t.Fatal("renderer name must not authorize an arbitrary native template")
	}
}

func TestFittingPromptStartStopsDuringBinarySearch(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	calls := 0
	messages := longTruncationHistory()
	_, err := fittingPromptStart(ctx, messages, 20, len(messages)-1, func(i int) (int, error) {
		calls++
		if calls == 2 {
			cancel()
		}
		return len(messages) - i, nil
	})
	if !errors.Is(err, context.Canceled) || calls != 2 {
		t.Fatalf("calls=%d err=%v", calls, err)
	}
}

func BenchmarkPromptTruncation(b *testing.B) {
	messages := longTruncationHistory()
	for _, fast := range []bool{false, true} {
		b.Run(fmt.Sprintf("fast=%t", fast), func(b *testing.B) {
			limit := -1
			if fast {
				limit = len(messages) - 1
			}
			b.ReportAllocs()
			for b.Loop() {
				_, err := fittingPromptStart(b.Context(), messages, 20, limit, func(i int) (int, error) {
					// Mirror the word tokenizer used by the HTTP runner test fixture.
					return len(strings.Fields(nativeChatQualityPrompt(llm.ChatRequest{Messages: promptMessagesFrom(messages, i)}))), nil
				})
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
