package server

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"unsafe"
)

type embeddingRetentionFixture struct {
	mockLlm
	cache                                            map[string][]int
	liveTokenBytes, peakTokenBytes                   int
	tokenBatches, extraTokenizations, embeddingCalls int
	groups                                           [][]string
	captureGroups, requireReleased                   bool
	onTokenize                                       func()
}

func (f *embeddingRetentionFixture) EmbeddingBatchLimits() (int, int) { return 8, 8192 }
func (f *embeddingRetentionFixture) TokenizeBatch(ctx context.Context, inputs []string) ([][]int, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.tokenBatches++
	result := make([][]int, len(inputs))
	for i := range inputs {
		result[i] = make([]int, 1024)
	}
	if f.onTokenize != nil {
		f.onTokenize()
	}
	return result, nil
}
func (f *embeddingRetentionFixture) remember(text string, tokens []int) {
	if old, ok := f.cache[text]; ok {
		f.liveTokenBytes -= cap(old) * int(unsafe.Sizeof(int(0)))
	}
	f.cache[text] = tokens
	f.liveTokenBytes += cap(tokens) * int(unsafe.Sizeof(int(0)))
	f.peakTokenBytes = max(f.peakTokenBytes, f.liveTokenBytes)
}
func (f *embeddingRetentionFixture) tokens(text string) ([]int, error) {
	if value, ok := f.cache[text]; ok {
		return value, nil
	}
	f.extraTokenizations++
	value := make([]int, 1024)
	f.remember(text, value)
	return value, nil
}
func (f *embeddingRetentionFixture) clear() {
	clear(f.cache)
	f.liveTokenBytes = 0
}
func (f *embeddingRetentionFixture) EmbeddingBatch(ctx context.Context, values []string) ([][]float32, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if f.requireReleased && f.liveTokenBytes != 0 {
		return nil, 0, errors.New("token arrays retained during inference")
	}
	f.embeddingCalls++
	if f.captureGroups {
		f.groups = append(f.groups, values)
	}
	result := make([][]float32, len(values))
	for i := range values {
		result[i] = make([]float32, 768)
		result[i][0] = float32(len(values[i]))
	}
	return result, len(values) * 1024, nil
}
func (f *embeddingRetentionFixture) prepare(text string) (string, bool, error) {
	_, err := f.tokens(text)
	return text, false, err
}
func (f *embeddingRetentionFixture) single(item embeddingInput) ([]float32, int, error) {
	f.embeddingCalls++
	result := make([]float32, 768)
	result[0] = float32(len(item.text))
	return result, 1024, nil
}

func TestEmbeddingBatchTokenWindowPreservesGlobalPacking(t *testing.T) {
	f := &embeddingRetentionFixture{cache: make(map[string][]int), captureGroups: true, requireReleased: true}
	inputs := make([]embeddingInput, 131)
	for i := range inputs {
		inputs[i].text = fmt.Sprintf("record-%03d", i)
	}
	inputs = append(inputs, inputs[65], inputs[0])
	prepare := func(text string) (string, bool, error) {
		if _, err := f.tokens(text); err != nil {
			return "", false, err
		}
		if text == "record-000" || text == "record-130" {
			return "shared", true, nil
		}
		return text, false, nil
	}
	result, count, used, err := prepareEmbeddingBatch(t.Context(), f, inputs, f.remember, f.clear, f.tokens, prepare, f.single)
	if err != nil || !used {
		t.Fatalf("used=%v err=%v", used, err)
	}
	if len(result) != len(inputs) || count != 130*1024 {
		t.Fatalf("result=%d tokens=%d", len(result), count)
	}
	if f.tokenBatches != 3 || f.extraTokenizations != 1 {
		t.Fatalf("token batches=%d extra=%d", f.tokenBatches, f.extraTokenizations)
	}
	if f.peakTokenBytes > 65*1024*int(unsafe.Sizeof(int(0))) || f.liveTokenBytes != 0 {
		t.Fatalf("peak=%d live=%d", f.peakTokenBytes, f.liveTokenBytes)
	}
	// 130 unique prepared texts pack globally into 18 groups of 7 and one of 4.
	// Packing each 64-item tokenization window separately would add requests.
	if f.embeddingCalls != 19 || len(f.groups) != 19 {
		t.Fatalf("global groups changed: %d", f.embeddingCalls)
	}
	for i, group := range f.groups {
		want := 7
		if i == 18 {
			want = 4
		}
		if len(group) != want {
			t.Fatalf("group%d size=%d want=%d", i, len(group), want)
		}
	}
	for i, item := range inputs {
		text := item.text
		if text == "record-000" || text == "record-130" {
			text = "shared"
		}
		if result[i][0] != float32(len(text)) {
			t.Fatalf("input%d order/content changed", i)
		}
	}
	result[0][0] = -1
	if result[130][0] < 0 || result[132][0] < 0 {
		t.Fatal("duplicate normalized outputs alias")
	}
}

func TestEmbeddingBatchTokenWindowReleasesOnFailure(t *testing.T) {
	for _, cancelRequest := range []bool{false, true} {
		t.Run(fmt.Sprintf("cancel=%v", cancelRequest), func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			f := &embeddingRetentionFixture{cache: make(map[string][]int)}
			if cancelRequest {
				f.onTokenize = cancel
			}
			inputs := make([]embeddingInput, 130)
			for i := range inputs {
				inputs[i].text = fmt.Sprintf("record-%03d", i)
			}
			failure := errors.New("invalid prepared text")
			prepare := func(text string) (string, bool, error) {
				if text == "record-065" {
					return "", false, failure
				}
				return f.prepare(text)
			}
			_, _, used, err := prepareEmbeddingBatch(ctx, f, inputs, f.remember, f.clear, f.tokens, prepare, f.single)
			want := failure
			if cancelRequest {
				want = context.Canceled
			}
			if !used || !errors.Is(err, want) {
				t.Fatalf("used=%v err=%v", used, err)
			}
			if f.liveTokenBytes != 0 || len(f.cache) != 0 || f.embeddingCalls != 0 {
				t.Fatalf("failed preparation retained tokens or inferred: live=%d cache=%d calls=%d", f.liveTokenBytes, len(f.cache), f.embeddingCalls)
			}
		})
	}
}

func BenchmarkEmbeddingTokenRetention(b *testing.B) {
	inputs := make([]embeddingInput, 4096)
	for i := range inputs {
		inputs[i].text = strings.Repeat("document ", 500) + fmt.Sprintf("%05d", i)
	}
	b.ReportAllocs()
	var last *embeddingRetentionFixture
	for b.Loop() {
		f := &embeddingRetentionFixture{cache: make(map[string][]int)}
		result, _, used, err := prepareEmbeddingBatch(b.Context(), f, inputs, f.remember, f.clear, f.tokens, f.prepare, f.single)
		if err != nil || !used || len(result) != len(inputs) {
			b.Fatalf("invalid embedding result: used=%v err=%v", used, err)
		}
		last = f
	}
	b.ReportMetric(float64(last.peakTokenBytes), "peak-token-B")
	b.ReportMetric(float64(last.tokenBatches), "tokenize-batches/op")
	b.ReportMetric(float64(last.extraTokenizations), "extra-tokenize/op")
	b.ReportMetric(float64(last.embeddingCalls), "embedding-calls/op")
}
