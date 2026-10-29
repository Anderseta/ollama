package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
)

func TestModelShowCacheVoicesIsolation(t *testing.T) {
	for _, cloud := range []bool{false, true} {
		t.Run(fmt.Sprintf("cloud=%v", cloud), func(t *testing.T) {
			cache := newModelShowCache()
			response := &api.ShowResponse{Voices: []string{"voice-one", "voice-two"}}
			localKey := modelShowLocalKey{Model: "speech"}
			cloudKey := modelShowCloudKey{Model: "speech"}
			var get func() (*api.ShowResponse, bool)
			if cloud {
				cache.setCloud(cloudKey, response)
				get = func() (*api.ShowResponse, bool) { return cache.getCloud(cloudKey) }
			} else {
				cache.setLocal(localKey, "digest", response)
				get = func() (*api.ShowResponse, bool) { return cache.getLocal(localKey, "digest") }
			}
			response.Voices[0] = "changed before read"
			for range 2 {
				got, ok := get()
				if !ok || !slices.Equal(got.Voices, []string{"voice-one", "voice-two"}) {
					t.Fatalf("cached voices changed: %+v, found=%v", got, ok)
				}
				got.Voices[1] = "changed after read"
			}
		})
	}
}

func TestGenerateLargeResponseContextModes(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_GO_TEMPLATE", "1")
	gin.SetMode(gin.TestMode)
	const count = 1024
	chunk := strings.Repeat("ação ", 32)
	want := strings.Repeat(chunk, count) + "final"
	fixture := &mockRunner{CompletionFn: func(ctx context.Context, req llm.CompletionRequest, emit func(llm.CompletionResponse)) error {
		for range count {
			if err := ctx.Err(); err != nil {
				return err
			}
			emit(llm.CompletionResponse{Content: chunk})
		}
		emit(llm.CompletionResponse{Content: "final", Done: true, DoneReason: llm.DoneReasonStop, EvalCount: count + 1})
		return nil
	}}
	s := newServerWithMockRunner(t, fixture)
	createMinimalGGUFModel(t, s, "copy-retention", gguftest.KV{}, "{{ .Prompt }}", nil)
	for _, raw := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("raw=%v/stream=%v", raw, stream), func(t *testing.T) {
				w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
					Model: "copy-retention", Prompt: "prompt ", Raw: raw, Stream: &stream,
				})
				if w.Code != http.StatusOK {
					t.Fatalf("status=%d: %s", w.Code, w.Body.String())
				}
				dec := json.NewDecoder(w.Body)
				var output strings.Builder
				var final api.GenerateResponse
				for {
					var response api.GenerateResponse
					err := dec.Decode(&response)
					if err == io.EOF {
						break
					}
					if err != nil {
						t.Fatal(err)
					}
					output.WriteString(response.Response)
					final = response
				}
				if output.String() != want || !final.Done || final.EvalCount != count+1 {
					t.Fatalf("output/final changed: bytes=%d, done=%v, count=%d", output.Len(), final.Done, final.EvalCount)
				}
				if raw {
					if len(final.Context) != 0 || fixture.CompletionRequest.Prompt != "prompt " {
						t.Fatal("raw prompt/context behavior changed")
					}
				} else if wantCount := len(strings.Fields(fixture.CompletionRequest.Prompt + want)); len(final.Context) != wantCount {
					t.Fatalf("context tokens=%d, want=%d", len(final.Context), wantCount)
				}
			})
		}
	}
}
