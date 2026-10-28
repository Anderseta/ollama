package server

import (
	"context"
	"crypto/sha256"
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
)

// Only these reviewed ChatML renderers remove whole, independently delimited
// message blocks. Arbitrary Jinja/Go templates need the exact linear fallback:
// their output can grow, or change entirely, when an old message is removed.
func monotonicPromptLimit(m *Model, messages []api.Message, native bool) int {
	if m == nil || len(messages) < 32 {
		return -1
	}
	if native {
		if !strings.HasPrefix(m.metadata.String("general.architecture"), "qwen") {
			return -1
		}
		source := strings.ReplaceAll(m.metadata.String("tokenizer.chat_template"), "\r\n", "\n")
		switch fmt.Sprintf("%x", sha256.Sum256([]byte(source))) {
		case "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715", // Qwen3.5
			"514d5304fcac63bda60e6faf860d5657010cf43cbc4c633c8bf678ee184e7a37": // Qwen3.8
		default:
			return -1
		}
	} else {
		switch resolveRendererName(m) {
		case "qwen3.5", "qwen3.8", "tev1", "d3":
		default:
			return -1
		}
		if m.Config.Renderer == "" {
			return -1
		}
	}
	lastQuery := -1
	for i, message := range messages {
		if len(message.Images) > 0 {
			return -1
		}
		switch message.Role {
		case "system":
			if i != 0 {
				return -1
			}
		case "user":
			content := strings.TrimSpace(message.Content)
			if !(strings.HasPrefix(content, "<tool_response>") && strings.HasSuffix(content, "</tool_response>")) {
				lastQuery = i
			}
		case "assistant", "tool":
		default:
			return -1
		}
	}
	// Keeping the latest real user query preserves historical-thinking policy.
	starts := promptTruncationStarts(messages)
	for i := 0; i <= lastQuery; i++ {
		if messages[i].Role == "tool" && starts[i] {
			return -1
		}
	}
	return lastQuery
}

func promptMessagesFrom(messages []api.Message, start int) []api.Message {
	if start == 0 {
		return slices.Clone(messages)
	}
	system := make([]api.Message, 0)
	for _, message := range messages[:start] {
		if message.Role == "system" {
			system = append(system, message)
		}
	}
	return append(system, messages[start:]...)
}

func fittingPromptStart(ctx context.Context, messages []api.Message, budget, monotonicLimit int, count func(int) (int, error)) (int, error) {
	valid := promptTruncationStarts(messages)
	starts := make([]int, 0, len(messages))
	changed := true
	for i := range messages {
		if i > 0 && messages[i-1].Role != "system" {
			changed = true
		}
		if valid[i] && changed {
			starts = append(starts, i)
			changed = false
		}
	}
	if len(starts) == 0 {
		return 0, ctx.Err()
	}
	fits := func(index int) (bool, error) {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		size, err := count(starts[index])
		if err == nil {
			err = ctx.Err()
		}
		return size <= budget, err
	}
	ok, err := fits(0)
	if err != nil || ok {
		return starts[0], err
	}
	next := 1
	if len(starts) >= 32 && monotonicLimit >= 0 {
		high := len(starts) - 1
		for high > 0 && starts[high] > monotonicLimit {
			high--
		}
		if high > 1 {
			ok, err = fits(high)
			if err != nil {
				return 0, err
			}
			if ok {
				low := 0
				for high-low > 1 {
					mid := low + (high-low)/2
					ok, err = fits(mid)
					if err != nil {
						return 0, err
					}
					if ok {
						high = mid
					} else {
						low = mid
					}
				}
				return starts[high], nil
			}
			next = high + 1
		}
	}
	for i := next; i < len(starts); i++ {
		ok, err = fits(i)
		if err != nil || ok {
			return starts[i], err
		}
	}
	// An oversized final tool group must retain the assistant that called it.
	return starts[len(starts)-1], nil
}
