package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/template"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

// chatPrompt accepts a list of messages and returns the prompt and media that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool, think *api.ThinkValue, truncate bool) (prompt string, media []llm.MediaData, _ error) {
	var system []api.Message

	// TODO: This is only a truncation heuristic; llama-server handles the
	// actual image/media inputs. Replace this with projector/model-aware media
	// token accounting so image history is neither over-packed nor over-trimmed.
	// Clip images are represented as 768 tokens, each an embedding.
	imageNumTokens := 768

	currMsgIdx := 0

	if truncate {
		var err error
		currMsgIdx, err = fittingPromptStart(ctx, msgs, opts.NumCtx, monotonicPromptLimit(m, msgs, false), func(i int) (int, error) {
			candidate := promptMessagesFrom(msgs, i)
			p, err := renderPrompt(m, candidate, tools, think)
			if err != nil {
				return 0, err
			}
			s, err := tokenize(ctx, p)
			if err != nil {
				return 0, err
			}
			ctxLen := len(s)
			if m.ProjectorPaths != nil {
				for _, msg := range candidate {
					ctxLen += imageNumTokens * len(msg.Images)
				}
			}
			return ctxLen, nil
		})
		if err != nil {
			return "", nil, err
		}
		system = nil
		for _, message := range msgs[:currMsgIdx] {
			if message.Role == "system" {
				system = append(system, message)
			}
		}
	}

	if currMsgIdx > 0 {
		slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[currMsgIdx:]))
	}

	renderMsgs, media, err := imageTaggedMessages(m, msgs, currMsgIdx, false)
	if err != nil {
		return "", nil, err
	}

	// truncate any messages that do not fit into the context window
	p, err := renderPrompt(m, append(system, renderMsgs[currMsgIdx:]...), tools, think)
	if err != nil {
		return "", nil, err
	}

	return p, media, nil
}

// A tool response must retain the assistant call that produced it.
func promptTruncationStarts(messages []api.Message) []bool {
	changes := make([]int, len(messages)+1)
	owners := make(map[string]int)
	lastOwner := -1
	for index, message := range messages {
		if message.Role == "user" {
			lastOwner = -1
		}
		if message.Role == "assistant" && len(message.ToolCalls) > 0 {
			lastOwner = index
			for _, call := range message.ToolCalls {
				if call.ID != "" {
					owners[call.ID] = index
				}
			}
		}
		if message.Role == "tool" {
			owner := lastOwner
			if message.ToolCallID != "" {
				if found, exists := owners[message.ToolCallID]; exists {
					owner = found
				}
			}
			if owner >= 0 && owner < index {
				changes[owner+1]++
				changes[index+1]--
			}
		}
	}
	result := make([]bool, len(messages))
	active := 0
	for index := range messages {
		active += changes[index]
		result[index] = active == 0
	}
	return result
}

func imageTaggedMessages(m *Model, msgs []api.Message, start int, clearImages bool) ([]api.Message, []llm.MediaData, error) {
	renderMsgs := slices.Clone(msgs)
	var media []llm.MediaData

	for cnt, msg := range renderMsgs[start:] {
		if slices.Contains(m.Config.ModelFamilies, "mllama") && len(msg.Images) > 1 {
			return nil, nil, errors.New("this model only supports one image while more than one image requested")
		}

		var prefix string
		prompt := msg.Content

		for _, i := range msg.Images {
			mediaData := llm.NewMediaData(len(media), i)
			media = append(media, mediaData)

			if m.Config.Renderer != "" {
				continue
			}

			// The prompt marker is still image-named for compatibility with
			// existing templates and llama-server media marker replacement.
			imgTag := fmt.Sprintf("[img-%d]", mediaData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}
		}

		if m.Config.Renderer == "" {
			renderMsgs[start+cnt].Content = prefix + prompt
		}
		if clearImages {
			renderMsgs[start+cnt].Images = nil
		}
	}

	return renderMsgs, media, nil
}

func renderPrompt(m *Model, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if m.Config.Renderer != "" {
		rendererName := resolveRendererName(m)
		rendered, err := renderers.RenderWithRenderer(rendererName, msgs, tools, think)
		if err != nil {
			return "", err
		}
		return rendered, nil
	}

	var b bytes.Buffer
	thinkVal := false
	thinkLevel := ""
	if think != nil {
		thinkVal = think.Bool()
		thinkLevel = think.String()
	}
	if err := m.Template.Execute(&b, template.Values{Messages: msgs, Tools: tools, Think: thinkVal, ThinkLevel: thinkLevel, IsThinkSet: think != nil}); err != nil {
		return "", err
	}
	return b.String(), nil
}
