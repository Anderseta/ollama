package server

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

type embeddingInput struct {
	text  string
	media [][]byte
}

func prepareEmbeddingBatch(ctx context.Context, runner llm.LlamaServer, inputs []embeddingInput,
	remember func(string, []int), tokens func(string) ([]int, error),
	prepare func(string) (string, bool, error), single func(embeddingInput) ([]float32, int, error),
) ([][]float32, int, bool, error) {
	batcher, ok := runner.(llm.EmbeddingBatchRunner)
	if !ok || len(inputs) < 2 {
		return nil, 0, false, nil
	}
	maxItems, maxTokens := batcher.EmbeddingBatchLimits()
	if maxItems < 2 || maxTokens <= 0 {
		return nil, 0, false, nil
	}
	texts := make([]string, 0, len(inputs))
	seen := make(map[string]bool)
	for _, item := range inputs {
		if len(item.media) != 0 {
			return nil, 0, false, nil
		}
		if len(item.text) > 16*1024*1024 {
			return nil, 0, false, nil
		}
		if !seen[item.text] {
			seen[item.text] = true
			texts = append(texts, item.text)
		}
	}
	for offset := 0; offset < len(texts); {
		end, size := offset, 0
		for end < len(texts) && end-offset < 64 && len(texts[end]) <= 16*1024*1024-size {
			size += len(texts[end])
			end++
		}
		part := texts[offset:end]
		result, err := batcher.TokenizeBatch(ctx, part)
		if err != nil {
			if strings.Contains(err.Error(), "invalid tokenization batch length") {
				return nil, 0, false, nil
			}
			return nil, 0, true, err
		}
		if len(result) != len(part) {
			return nil, 0, true, fmt.Errorf("invalid tokenization batch length")
		}
		for i, value := range result {
			remember(part[i], value)
		}
		offset = end
	}
	type job struct {
		text    string
		cost    int
		indices []int
	}
	jobs := make([]job, 0, len(texts))
	byText := make(map[string]int)
	for index, input := range inputs {
		text, _, err := prepare(input.text)
		if err != nil {
			return nil, 0, true, err
		}
		if existing, found := byText[text]; found {
			jobs[existing].indices = append(jobs[existing].indices, index)
			continue
		}
		tokenized, err := tokens(text)
		if err != nil {
			return nil, 0, true, err
		}
		byText[text] = len(jobs)
		jobs = append(jobs, job{text: text, cost: len(tokenized) + 2, indices: []int{index}})
	}
	slices.SortStableFunc(jobs, func(a, b job) int { return a.cost - b.cost })
	result := make([][]float32, len(inputs))
	total := 0
	for offset := 0; offset < len(jobs); {
		end, cost := offset, 0
		for end < len(jobs) && end-offset < maxItems && jobs[end].cost <= maxTokens-cost {
			cost += jobs[end].cost
			end++
		}
		if end == offset {
			end++
		}
		group := jobs[offset:end]
		var vectors [][]float32
		var count int
		var err error
		if len(group) > 1 {
			values := make([]string, len(group))
			for i, job := range group {
				values[i] = job.text
			}
			vectors, count, err = batcher.EmbeddingBatch(ctx, values)
		}
		var status api.StatusError
		if len(group) == 1 || (errors.As(err, &status) && status.StatusCode == 413) {
			vectors, count, err = make([][]float32, len(group)), 0, nil
			for i, job := range group {
				var n int
				vectors[i], n, err = single(embeddingInput{text: job.text})
				if err != nil {
					break
				}
				count += n
			}
		}
		if err != nil {
			return nil, 0, true, err
		}
		if len(vectors) != len(group) {
			return nil, 0, true, fmt.Errorf("invalid embedding batch result")
		}
		for i, job := range group {
			for _, index := range job.indices {
				result[index] = slices.Clone(vectors[i])
			}
		}
		total += count
		offset = end
	}
	return result, total, true, nil
}
