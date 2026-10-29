package parsers

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwenLongToolCallChunkBoundaries(t *testing.T) {
	payload := strings.Repeat("ação 🙂 </tool_callx> </tool_ ", 512)
	input := "before<tool_call><function=write><parameter=content>" + payload +
		"</parameter></function></tool_call>between<tool_call><function=next>" +
		"<parameter=value>ok</parameter></function></tool_call>after"
	for _, name := range []string{"qwen3-coder", "qwen3.5"} {
		for _, size := range []int{1, 3, 11, 32, 4096, len(input)} {
			t.Run(fmt.Sprintf("%s/%d", name, size), func(t *testing.T) {
				parser := ParserForName(name)
				parser.Init(nil, nil, &api.ThinkValue{Value: false})
				var text strings.Builder
				var calls []api.ToolCall
				for start := 0; start < len(input); start += size {
					end := min(start+size, len(input))
					content, thinking, got, err := parser.Add(input[start:end], end == len(input))
					if err != nil || thinking != "" {
						t.Fatalf("Add at %d: thinking=%q, err=%v", start, thinking, err)
					}
					text.WriteString(content)
					calls = append(calls, got...)
				}
				if text.String() != "beforebetweenafter" || len(calls) != 2 {
					t.Fatalf("content=%q, calls=%d", text.String(), len(calls))
				}
				want := []api.ToolCall{
					{Function: api.ToolCallFunction{Name: "write", Arguments: testArgs(map[string]any{"content": payload}), Index: 0}},
					{Function: api.ToolCallFunction{Name: "next", Arguments: testArgs(map[string]any{"value": "ok"}), Index: 1}},
				}
				for i := range want {
					if !toolCallEqual(calls[i], want[i]) {
						t.Fatalf("call %d differs from the complete tool arguments", i)
					}
				}
			})
		}
	}
}

func TestQwenCoderInitDiscardsIncompletePreviousStream(t *testing.T) {
	for _, previous := range []string{"<tool_", "<tool_call><function=abandoned>" + strings.Repeat("x", 4096)} {
		parser := &Qwen3CoderParser{}
		parser.Init(nil, nil, nil)
		if _, _, _, err := parser.Add(previous, false); err != nil {
			t.Fatal(err)
		}
		parser.Init(nil, nil, nil)
		content, _, calls, err := parser.Add("new answer<tool_call><function=current></function></tool_call>", true)
		if err != nil || content != "new answer" || len(calls) != 1 || calls[0].Function.Name != "current" || calls[0].Function.Index != 0 {
			t.Fatalf("previous stream leaked after Init: content=%q, calls=%v, err=%v", content, calls, err)
		}
	}
}

func TestQwenToolStreamEmptyFinalChunk(t *testing.T) {
	for _, name := range []string{"qwen3-coder", "qwen3.5"} {
		for _, suffix := range []string{"answer \n", "answer <tool_", "<tool_call><function=incomplete>"} {
			t.Run(name+"/"+suffix, func(t *testing.T) {
				parser := ParserForName(name)
				parser.Init(nil, nil, &api.ThinkValue{Value: false})
				content, _, calls, err := parser.Add(suffix, false)
				if err != nil || len(calls) != 0 {
					t.Fatalf("nonterminal chunk: calls=%v, err=%v", calls, err)
				}
				tail, _, calls, err := parser.Add("", true)
				if err != nil || len(calls) != 0 || content+tail != suffix {
					t.Fatalf("terminal chunk lost content: got=%q, want=%q, calls=%v, err=%v", content+tail, suffix, calls, err)
				}
			})
		}
	}
}

func BenchmarkQwenStreamedToolCall(b *testing.B) {
	for _, name := range []string{"qwen3-coder", "qwen3.5"} {
		for _, size := range []int{8 << 10, 64 << 10, 256 << 10} {
			b.Run(fmt.Sprintf("%s/%dKiB", name, size>>10), func(b *testing.B) {
				payload := strings.Repeat("abcdefgh", size/8)
				input := "<tool_call><function=write><parameter=content>" + payload + "</parameter></function></tool_call>"
				b.SetBytes(int64(len(input)))
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					parser := ParserForName(name)
					parser.Init(nil, nil, &api.ThinkValue{Value: false})
					count := 0
					for start := 0; start < len(input); start += 32 {
						end := min(start+32, len(input))
						_, _, calls, err := parser.Add(input[start:end], end == len(input))
						if err != nil {
							b.Fatal(err)
						}
						count += len(calls)
					}
					if count != 1 {
						b.Fatalf("got %d complete tool calls", count)
					}
				}
			})
		}
	}
}
