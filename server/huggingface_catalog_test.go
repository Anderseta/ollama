package server

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestHuggingFaceCatalogGroupsCompleteVariantsWithoutProjectorOrOtherQuants(t *testing.T) {
	files := []huggingFaceFile{
		huggingFaceTestFile("IQ3_S/model-IQ3_S-00002-of-00002.gguf"),
		huggingFaceTestFile("IQ3_S/model-IQ3_S-00001-of-00002.gguf"),
		huggingFaceTestFile("other-Q4_K_M.gguf"),
		huggingFaceTestFile("broken-Q8_0-00001-of-00002.gguf"),
		huggingFaceTestFile("mmproj-BF16.gguf"),
		huggingFaceTestFile("mmproj-F16.gguf"),
		{Filename: "model.safetensors"},
	}
	catalog := huggingFaceVariants("owner/model", huggingFaceRepository{SHA: strings.Repeat("a", 40), Files: files})
	if len(catalog.Variants) != 3 || len(catalog.Projectors) != 2 || catalog.UnsupportedFiles != 1 {
		t.Fatalf("unexpected catalog: %+v", catalog)
	}
	if first := catalog.Variants[0]; first.Filename != files[1].Filename || first.Shards != 2 || first.Size != 200 || !first.Complete || first.Quantization != "IQ3_S" {
		t.Fatalf("unexpected split variant: %+v", first)
	}
	if incomplete := catalog.Variants[1]; incomplete.Complete || incomplete.Error == "" {
		t.Fatalf("missing shard was listed as complete: %+v", incomplete)
	}
	projector := "mmproj-F16.gguf"
	selected, err := selectHuggingFaceGGUF(files, "same-tag", huggingFaceSelection{File: files[1].Filename, Projector: &projector})
	if err != nil {
		t.Fatal(err)
	}
	if len(selected) != 3 || selected[2].Filename != projector {
		t.Fatalf("wrong exact selection: %+v", selected)
	}
	none := "none"
	selected, err = selectHuggingFaceGGUF(files, "same-tag", huggingFaceSelection{File: files[1].Filename, Projector: &none})
	if err != nil || len(selected) != 2 {
		t.Fatalf("explicit text-only selection: %+v, %v", selected, err)
	}
}

func TestHuggingFaceExactSelectionRejectsUnverifiedFilesAndUnsafeSelectors(t *testing.T) {
	files := []huggingFaceFile{huggingFaceTestFile("sub/model-Q4_K_M.gguf"), huggingFaceTestFile("mmproj-F16.gguf")}
	for _, selection := range []huggingFaceSelection{
		{File: "../model.gguf"}, {File: "sub\\model.gguf"}, {File: "config.json"},
		{File: "missing.gguf"}, {File: files[0].Filename, Revision: "main"},
		{File: files[0].Filename, Projector: func() *string { value := "missing-projector.gguf"; return &value }()},
	} {
		if _, err := selectHuggingFaceGGUF(files, "latest", selection); err == nil {
			t.Errorf("invalid selector accepted: %+v", selection)
		}
	}
	files[0].LFS.SHA256 = "invalid"
	if _, err := selectHuggingFaceGGUF(files, "latest", huggingFaceSelection{File: files[0].Filename}); err == nil {
		t.Fatal("invalid digest accepted")
	}
}

func TestHuggingFaceCatalogProjectorValidationAndLFSSizeFallback(t *testing.T) {
	file := huggingFaceTestFile("model-Q4_K_M.gguf")
	file.Size = 0
	projector := huggingFaceTestFile("mmproj-BF16.gguf")
	projector.LFS.SHA256 = "invalid"
	catalog := huggingFaceVariants("owner/model", huggingFaceRepository{Files: []huggingFaceFile{file, projector}})
	if len(catalog.Variants) != 1 || !catalog.Variants[0].Complete || catalog.Variants[0].Size != file.LFS.Size {
		t.Fatalf("valid LFS size omitted: %+v", catalog)
	}
	if len(catalog.Projectors) != 1 || catalog.Projectors[0].Complete == nil || *catalog.Projectors[0].Complete || catalog.Projectors[0].Error == "" {
		t.Fatalf("unverified projector marked available: %+v", catalog.Projectors)
	}
	file.Size = 50
	catalog = huggingFaceVariants("owner/model", huggingFaceRepository{Files: []huggingFaceFile{file}})
	if catalog.Variants[0].Complete {
		t.Fatal("mismatched size was normalized instead of rejected")
	}
}

type huggingFaceRoundTrip func(*http.Request) (*http.Response, error)

func (fn huggingFaceRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return fn(r) }

func mockHuggingFaceHTTP(t *testing.T, fn func(http.ResponseWriter, *http.Request)) {
	t.Helper()
	previous := http.DefaultTransport
	previousCache := huggingFaceMetadata
	huggingFaceMetadata = newHuggingFaceMetadataCache()
	http.DefaultTransport = huggingFaceRoundTrip(func(r *http.Request) (*http.Response, error) {
		recorder := httptest.NewRecorder()
		fn(recorder, r)
		return recorder.Result(), nil
	})
	t.Cleanup(func() { http.DefaultTransport = previous; huggingFaceMetadata = previousCache })
}

func TestHuggingFaceCatalogEndpointsBoundQueriesAndPreserveRevision(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sha := strings.Repeat("b", 40)
	requests := 0
	mockHuggingFaceHTTP(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Host != "huggingface.co" || r.URL.Scheme != "https" {
			t.Fatalf("unexpected metadata host: %v", r.URL)
		}
		if r.URL.Path == "/api/models" {
			if r.URL.Query().Get("search") != "Qwen GGUF" || r.URL.Query().Get("filter") != "gguf" || r.URL.Query().Get("limit") != "20" {
				t.Fatalf("unbounded/wrong search: %v", r.URL)
			}
			_, _ = io.WriteString(w, `[{"id":"owner/model","downloads":123,"likes":4,"private":false,"gated":"manual"}]`)
			return
		}
		if r.URL.Path != "/api/models/owner/model" || r.URL.Query().Get("blobs") != "true" {
			t.Fatalf("wrong metadata URL: %v", r.URL)
		}
		_ = json.NewEncoder(w).Encode(huggingFaceRepository{SHA: sha, Files: []huggingFaceFile{huggingFaceTestFile("model-Q4_K_M.gguf")}})
	})
	s := &Server{}
	router := gin.New()
	router.GET("/search", s.HuggingFaceSearchHandler)
	router.GET("/files", s.HuggingFaceFilesHandler)
	for _, route := range []string{"/search?q=Qwen+GGUF", "/files?repository=owner/model"} {
		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, route, nil))
		if recorder.Code != 200 {
			t.Fatalf("catalog endpoint failed: %d %s", recorder.Code, recorder.Body)
		}
		if strings.HasPrefix(route, "/files") && !strings.Contains(recorder.Body.String(), sha) {
			t.Fatal("revision omitted")
		}
		if strings.HasPrefix(route, "/search") && !strings.Contains(recorder.Body.String(), `"gated":true`) {
			t.Fatal("gated flag omitted")
		}
	}
	for _, route := range []string{"/search?q=x", "/search?q=" + strings.Repeat("x", 201), "/files?repository=../private", "/files?repository=https://evil.invalid/model"} {
		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, route, nil))
		if recorder.Code != 400 {
			t.Fatalf("unsafe query accepted: %s", route)
		}
	}
	if requests != 2 {
		t.Fatalf("invalid queries reached network: %d requests", requests)
	}
}

func TestHuggingFaceMetadataPinsCommitAndRejectsOversizedResponses(t *testing.T) {
	sha := strings.Repeat("c", 40)
	name := model.ParseName("huggingface.co/owner/model")
	mockHuggingFaceHTTP(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/model/revision/"+sha {
			t.Fatalf("selection not pinned: %v", r.URL)
		}
		_ = json.NewEncoder(w).Encode(huggingFaceRepository{SHA: sha})
	})
	if _, err := readHuggingFaceRepository(t.Context(), name, &registryOptions{HuggingFace: huggingFaceSelection{Revision: sha}}); err != nil {
		t.Fatal(err)
	}
	http.DefaultTransport = huggingFaceRoundTrip(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"sha":"` + strings.Repeat("d", 40) + `","siblings":[]}`))}, nil
	})
	if _, err := readHuggingFaceRepository(t.Context(), name, &registryOptions{HuggingFace: huggingFaceSelection{Revision: sha}}); err == nil || !strings.Contains(err.Error(), "revision changed") {
		t.Fatalf("different revision was accepted: %v", err)
	}
	http.DefaultTransport = huggingFaceRoundTrip(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(strings.Repeat(" ", (8<<20)+1)))}, nil
	})
	if _, err := readHuggingFaceRepository(t.Context(), name, &registryOptions{}); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("unbounded response accepted: %v", err)
	}
}

func TestHuggingFacePullHandlerRejectsSelectionForRegistryBeforeNetwork(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	router := gin.New()
	router.POST("/pull", s.PullHandler)
	for _, req := range []api.PullRequest{
		{Model: "qwen:latest", HuggingFaceFile: "model.gguf"},
		{Model: "huggingface.co/owner/model:Q4_K_M", HuggingFaceFile: "../model.gguf"},
	} {
		body, _ := json.Marshal(req)
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodPost, "/pull", bytes.NewReader(body))
		request.Header.Set("Content-Type", "application/json")
		router.ServeHTTP(recorder, request)
		if recorder.Code != 400 {
			t.Fatalf("invalid selection accepted: %d %s", recorder.Code, recorder.Body)
		}
	}
}

func TestHuggingFaceCatalogCancellationReachesMetadataRequest(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	mockHuggingFaceHTTP(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Context().Err() != context.Canceled {
			t.Fatal("cancellation was discarded")
		}
		w.WriteHeader(503)
	})
	target, _ := url.Parse("https://huggingface.co/api/models")
	if err := readHuggingFaceJSON(ctx, target, &registryOptions{}, new(any)); err == nil {
		t.Fatal("cancelled request succeeded")
	}
}

func TestHuggingFaceFilesEndpointPinsRevisionAndCachesEachRevisionSeparately(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sha := strings.Repeat("a", 40)
	requests := map[string]int{}
	mockHuggingFaceHTTP(t, func(w http.ResponseWriter, r *http.Request) {
		requests[r.URL.Path]++
		value := strings.Repeat("b", 40)
		if strings.HasSuffix(r.URL.Path, "/revision/"+sha) {
			value = sha
		}
		_ = json.NewEncoder(w).Encode(huggingFaceRepository{SHA: value, Files: []huggingFaceFile{huggingFaceTestFile("model-Q4_K_M.gguf")}})
	})
	router := gin.New()
	s := &Server{}
	router.GET("/files", s.HuggingFaceFilesHandler)
	for _, revision := range []string{sha, sha, "", ""} {
		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/files?repository=owner/model&revision="+revision, nil))
		if recorder.Code != 200 || revision != "" && !strings.Contains(recorder.Body.String(), sha) {
			t.Fatalf("pinned catalog failed: %d %s", recorder.Code, recorder.Body)
		}
	}
	if requests["/api/models/owner/model/revision/"+sha] != 1 || requests["/api/models/owner/model"] != 1 {
		t.Fatalf("metadata repeated or revisions shared: %+v", requests)
	}
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/files?repository=owner/model&revision=main", nil))
	if recorder.Code != 400 {
		t.Fatal("moving branch accepted as pinned commit")
	}
}
