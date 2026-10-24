package server

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

type batchFixture struct {
	mockLlm
	groups [][]string
	fail   bool
}

func (f *batchFixture) EmbeddingBatchLimits() (int, int) { return 3, 12 }
func (f *batchFixture) TokenizeBatch(ctx context.Context, inputs []string) ([][]int, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([][]int, len(inputs))
	for i, value := range inputs {
		result[i] = make([]int, len(strings.Fields(value)))
	}
	return result, nil
}
func (f *batchFixture) EmbeddingBatch(ctx context.Context, inputs []string) ([][]float32, int, error) {
	f.groups = append(f.groups, inputs)
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if f.fail {
		return nil, 0, api.StatusError{StatusCode: 413, ErrorMessage: "too large"}
	}
	result := make([][]float32, len(inputs))
	for i, value := range inputs {
		result[i] = []float32{float32(len(value)), 1}
	}
	return result, len(inputs), nil
}
func TestEmbeddingBatchGroupingDedupAndFallback(t *testing.T) {
	for _, fail := range []bool{false, true} {
		fixture := &batchFixture{fail: fail}
		cache := map[string][]int{}
		singles := 0
		inputs := []embeddingInput{{text: "one two"}, {text: "short"}, {text: "one two"}, {text: "three four five"}}
		result, _, used, err := prepareEmbeddingBatch(t.Context(), fixture, inputs,
			func(text string, tokens []int) { cache[text] = tokens },
			func(text string) ([]int, error) { return cache[text], nil },
			func(text string) (string, bool, error) { return text, false, nil },
			func(item embeddingInput) ([]float32, int, error) {
				singles++
				return []float32{float32(len(item.text)), 1}, 1, nil
			})
		if err != nil || !used || len(result) != len(inputs) {
			t.Fatalf("batch result: %v %v", used, err)
		}
		if len(fixture.groups) != 1 {
			t.Fatalf("expected one native batch, got %d", len(fixture.groups))
		}
		if fail && singles != 3 {
			t.Fatalf("fallback should process only unique inputs: %d", singles)
		}
		for i, item := range inputs {
			if result[i][0] != float32(len(item.text)) {
				t.Fatal("order changed")
			}
		}
		result[0][0] = -1
		if result[2][0] < 0 {
			t.Fatal("duplicate outputs alias each other")
		}
	}
}
