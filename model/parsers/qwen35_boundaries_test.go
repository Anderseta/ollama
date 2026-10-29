package parsers

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwen35ToolBoundaryPrecedesLiteralThinkingTag(t *testing.T) {
	payload := "O código contém </think> e </thi como texto literal."
	call := "<tool_call><function=write><parameter=content>" + payload + "</parameter></function></tool_call>"
	for _, closeThinking := range []string{"", "</think>"} {
		input := "<think>Vou gravar" + closeThinking + call + "Pronto."
		for split := 0; split <= len(input); split++ {
			t.Run(fmt.Sprintf("closed=%t/split=%d", closeThinking != "", split), func(t *testing.T) {
				parser := ParserForName("qwen3.5")
				parser.Init(nil, nil, &api.ThinkValue{Value: true})
				var content, thinking strings.Builder
				var calls []api.ToolCall
				for i, chunk := range []string{input[:split], input[split:]} {
					c, th, got, err := parser.Add(chunk, i == 1)
					if err != nil {
						t.Fatal(err)
					}
					content.WriteString(c)
					thinking.WriteString(th)
					calls = append(calls, got...)
				}
				if content.String() != "Pronto." || thinking.String() != "Vou gravar" || len(calls) != 1 {
					t.Fatalf("chunking changed boundaries: content=%q, thinking=%q, calls=%d", content.String(), thinking.String(), len(calls))
				}
				value, _ := calls[0].Function.Arguments.Get("content")
				if calls[0].Function.Name != "write" || value != payload {
					t.Fatalf("tool argument lost literal thinking tags: %v", calls[0])
				}
			})
		}
	}
}

func TestQwen35TerminalThinkingTail(t *testing.T) {
	for _, text := range []string{"Reflexão \n\t", "Reflexão </thi", "Reflexão <tool_", "<thi"} {
		for _, separateDone := range []bool{false, true} {
			t.Run(fmt.Sprintf("%q/separateDone=%t", text, separateDone), func(t *testing.T) {
				parser := ParserForName("qwen3.5")
				parser.Init(nil, nil, &api.ThinkValue{Value: true})
				content, thinking, calls, err := parser.Add(text, !separateDone)
				if separateDone {
					c, th, more, finalErr := parser.Add("", true)
					content += c
					thinking += th
					calls = append(calls, more...)
					if finalErr != nil {
						t.Fatal(finalErr)
					}
				}
				if err != nil || content != "" || thinking != text || len(calls) != 0 {
					t.Fatalf("terminal text lost: content=%q, thinking=%q, calls=%v, err=%v", content, thinking, calls, err)
				}
				c, th, more, finalErr := parser.Add("", true)
				if c != "" || th != "" || len(more) != 0 || finalErr != nil {
					t.Fatal("terminal text was emitted twice")
				}
			})
		}
	}
}

func TestQwen35MissingThinkingCloseWithTruncatedTool(t *testing.T) {
	parser := ParserForName("qwen3.5")
	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	input := "Vou gravar<tool_call><function=write><parameter=content>literal <"
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil || thinking != "Vou gravar" || content != strings.TrimPrefix(input, "Vou gravar") || len(calls) != 0 {
		t.Fatalf("truncated tool crossed thinking boundary: content=%q, thinking=%q, calls=%v, err=%v", content, thinking, calls, err)
	}
}
