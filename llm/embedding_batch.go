package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/ollama/ollama/api"
)

func (s *llamaServerRunner) EmbeddingBatchLimits() (int, int) {
	if !s.launch.embedding || s.launch.numParallel < 2 {
		return 1, 0
	}
	physical := s.launch.opts.NumBatch
	if s.launch.opts.NumUbatch > 0 {
		physical = min(physical, s.launch.opts.NumUbatch)
	}
	return min(s.launch.numParallel, 32), physical
}

func (s *llamaServerRunner) batchJSON(ctx context.Context, route string, input, output any) error {
	data, err := json.Marshal(input)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d%s", s.port, route), bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := s.httpClient().Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode >= 400 {
		body, err := io.ReadAll(io.LimitReader(response.Body, 64*1024))
		if err != nil {
			return err
		}
		status, message := normalizeEmbeddingError(response.StatusCode, body)
		return api.StatusError{StatusCode: status, ErrorMessage: message}
	}
	return json.NewDecoder(response.Body).Decode(output)
}

func (s *llamaServerRunner) TokenizeBatch(ctx context.Context, inputs []string) ([][]int, error) {
	if len(inputs) == 0 || len(inputs) > 64 {
		return nil, fmt.Errorf("tokenization batch must contain 1 to 64 strings")
	}
	var response struct {
		Tokens [][]int `json:"tokens_batch"`
	}
	if err := s.batchJSON(ctx, "/tokenize", map[string]any{"contents": inputs}, &response); err != nil {
		return nil, err
	}
	if len(response.Tokens) != len(inputs) {
		return nil, fmt.Errorf("invalid tokenization batch length")
	}
	return response.Tokens, nil
}

func (s *llamaServerRunner) EmbeddingBatch(ctx context.Context, inputs []string) ([][]float32, int, error) {
	maximum, _ := s.EmbeddingBatchLimits()
	if len(inputs) == 0 || len(inputs) > maximum {
		return nil, 0, fmt.Errorf("embedding batch exceeds %d sequences", maximum)
	}
	if err := s.sem.Acquire(ctx, int64(len(inputs))); err != nil {
		return nil, 0, err
	}
	defer s.sem.Release(int64(len(inputs)))
	request := map[string]any{"input": inputs}
	if s.rawEmbeddings {
		request["embd_normalize"] = -1
	}
	var response struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
		Usage struct {
			Tokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := s.batchJSON(ctx, "/v1/embeddings", request, &response); err != nil {
		return nil, 0, err
	}
	if len(response.Data) != len(inputs) {
		return nil, 0, fmt.Errorf("invalid embedding batch length")
	}
	result := make([][]float32, len(inputs))
	for _, item := range response.Data {
		if item.Index < 0 || item.Index >= len(result) || result[item.Index] != nil || len(item.Embedding) == 0 {
			return nil, 0, fmt.Errorf("invalid embedding batch index")
		}
		result[item.Index] = item.Embedding
	}
	return result, response.Usage.Tokens, nil
}
