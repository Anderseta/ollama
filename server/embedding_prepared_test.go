package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

type preparedEmbeddingFixture struct {
	mockLlm
	tokenBatches, tokenizations, detokenizations int
	embedded                                     []string
	rejectLong                                   bool
}

func (f *preparedEmbeddingFixture) EmbeddingBatchLimits() (int, int) { return 3, 3 }
func (f *preparedEmbeddingFixture) TokenizeBatch(_ context.Context, texts []string) ([][]int, error) {
	f.tokenBatches++
	result := make([][]int, len(texts))
	for i, text := range texts {
		result[i] = preparedFixtureTokens(text)
	}
	return result, nil
}
func preparedFixtureTokens(text string) []int {
	result := make([]int, len(strings.Fields(text)))
	for i := range result {
		result[i] = i
	}
	return result
}
func (f *preparedEmbeddingFixture) Tokenize(_ context.Context, text string) ([]int, error) {
	f.tokenizations++
	return preparedFixtureTokens(text), nil
}
func (f *preparedEmbeddingFixture) Detokenize(_ context.Context, tokens []int) (string, error) {
	f.detokenizations++
	return strings.TrimSpace(strings.Repeat("trim ", len(tokens))), nil
}
func (f *preparedEmbeddingFixture) EmbeddingBatch(context.Context, []string) ([][]float32, int, error) {
	return nil, 0, fmt.Errorf("expected prepared single-item groups")
}
func (f *preparedEmbeddingFixture) Embedding(_ context.Context, text string, _ [][]byte) ([]float32, int, error) {
	f.embedded = append(f.embedded, text)
	if f.rejectLong && len(strings.Fields(text)) > 2 {
		return nil, 0, api.StatusError{StatusCode: 413, ErrorMessage: "input too long"}
	}
	return []float32{3, 4}, len(strings.Fields(text)), nil
}

func TestEmbedHandlerPreparedSingleAvoidsRetokenization(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	writeShowManifestVariant(t, "embedding-prepared", manifest.RunnerGGML, manifest.FormatGGUF, model.ConfigV2{
		ModelFormat: manifest.FormatGGUF, Capabilities: []string{string(model.CapabilityEmbedding)},
	}, map[string]any{"general.architecture": "llama", "llama.context_length": uint32(64)})
	for _, strict := range []bool{false, true} {
		for _, nativeLimit := range []bool{false, true} {
			t.Run(fmt.Sprintf("strict=%v/native-limit=%v", strict, nativeLimit), func(t *testing.T) {
				runner := &preparedEmbeddingFixture{rejectLong: nativeLimit}
				s := &Server{sched: &Scheduler{
					pendingReqCh: make(chan *LlmRequest, 1), finishedReqCh: make(chan *LlmRequest, 1), expiredCh: make(chan *runnerRef, 1),
					unloadedCh: make(chan any, 1), loaded: make(map[string]*runnerRef), getGpuFn: getGpuFn, getSystemInfoFn: getSystemInfoFn, waitForRecovery: 250 * time.Millisecond,
					loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
						req.successCh <- &runnerRef{llama: runner}
						return false
					},
				}}
				go s.sched.Run(t.Context())
				truncate := !strict
				texts := []string{"one", "two", "one"}
				if nativeLimit {
					texts = []string{"one two three four", "one"}
				}
				response := createRequest(t, s.EmbedHandler, api.EmbedRequest{Model: "embedding-prepared", Input: texts, Truncate: &truncate, Options: map[string]any{"num_batch": 3}})
				status := http.StatusOK
				if strict && nativeLimit {
					status = http.StatusRequestEntityTooLarge
				}
				if response.Code != status {
					t.Fatalf("status=%d want=%d body=%s", response.Code, status, response.Body.String())
				}
				wantTokens, wantDetokens := 0, 0
				if nativeLimit && !strict {
					wantTokens, wantDetokens = 1, 1
				}
				if runner.tokenBatches != 1 || runner.tokenizations != wantTokens || runner.detokenizations != wantDetokens {
					t.Fatalf("batch=%d separate-tokenize=%d detokenize=%d; expected1/%d/%d", runner.tokenBatches, runner.tokenizations, runner.detokenizations, wantTokens, wantDetokens)
				}
				if status == http.StatusOK {
					var result api.EmbedResponse
					if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
						t.Fatal(err)
					}
					if len(result.Embeddings) != len(texts) {
						t.Fatalf("outputs=%d", len(result.Embeddings))
					}
					for _, vector := range result.Embeddings {
						if len(vector) != 2 || vector[0] != float32(0.6) || vector[1] != float32(0.8) {
							t.Fatalf("normalized embedding changed: %v", vector)
						}
					}
				}
				if nativeLimit && !strict && runner.embedded[len(runner.embedded)-1] != "trim trim" {
					t.Fatalf("physical limit retry changed: %v", runner.embedded)
				}
			})
		}
	}
}
