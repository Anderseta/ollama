package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestNativeEmbeddingBatchOrderAndTokenization(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if r.URL.Path == "/tokenize" {
			fmt.Fprint(w, `{"tokens_batch":[[1,2],[3]]}`)
			return
		}
		if len(body["input"].([]any)) != 2 {
			t.Error("not a real array request")
		}
		fmt.Fprint(w, `{"data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}],"usage":{"prompt_tokens":5}}`)
	}))
	defer server.Close()
	address, _ := url.Parse(server.URL)
	port, _ := strconv.Atoi(address.Port())
	runner := &llamaServerRunner{port: port, sem: semaphore.NewWeighted(4), launch: llamaServerLaunchConfig{embedding: true, numParallel: 4, opts: api.Options{Runner: api.Runner{NumBatch: 512}}}}
	tokens, err := runner.TokenizeBatch(t.Context(), []string{"a", "b"})
	if err != nil || len(tokens) != 2 || len(tokens[0]) != 2 {
		t.Fatalf("tokenization: %v", err)
	}
	vectors, count, err := runner.EmbeddingBatch(t.Context(), []string{"a", "b"})
	if err != nil || count != 5 || vectors[0][0] != 1 || vectors[1][0] != 3 {
		t.Fatalf("embedding ordering or usage: %v", err)
	}
}
