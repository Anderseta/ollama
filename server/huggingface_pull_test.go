package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func huggingFaceTestFile(name string) huggingFaceFile {
	file := huggingFaceFile{Filename: name, Size: 100}
	file.LFS.SHA256 = strings.Repeat("a", 64)
	file.LFS.Size = file.Size
	return file
}

func TestHuggingFaceSelectShardsAndVision(t *testing.T) {
	files := []huggingFaceFile{
		huggingFaceTestFile("model-IQ3_S-00002-of-00002.gguf"),
		huggingFaceTestFile("mmproj-model-BF16.gguf"),
		huggingFaceTestFile("model-Q4_K_M.gguf"),
		huggingFaceTestFile("model-IQ3_S-00001-of-00002.gguf"),
	}
	for _, tag := range []string{"IQ3_S", "iq3_s", "model-IQ3_S-00001-of-00002.gguf"} {
		selected, err := selectHuggingFaceGGUF(files, tag)
		if err != nil {
			t.Fatal(err)
		}
		var names []string
		for _, file := range selected {
			names = append(names, file.Filename)
		}
		want := []string{"model-IQ3_S-00001-of-00002.gguf", "model-IQ3_S-00002-of-00002.gguf", "mmproj-model-BF16.gguf"}
		if !slices.Equal(names, want) {
			t.Fatalf("selected %q, want %q", names, want)
		}
	}
}

func TestHuggingFaceRejectsMissingShardsUnsafePathsAndAmbiguousQuant(t *testing.T) {
	for name, files := range map[string][]huggingFaceFile{
		"missing":       {huggingFaceTestFile("model-IQ3_S-00001-of-00002.gguf")},
		"duplicate":     {huggingFaceTestFile("model-IQ3_S-00001-of-00002.gguf"), huggingFaceTestFile("model-IQ3_S-00001-of-00002.gguf")},
		"traversal":     {huggingFaceTestFile("../model-IQ3_S.gguf")},
		"windows path":  {huggingFaceTestFile(`folder\model-IQ3_S.gguf`)},
		"ambiguous":     {huggingFaceTestFile("first-IQ3_S.gguf"), huggingFaceTestFile("second-IQ3_S.gguf")},
		"unknown quant": {huggingFaceTestFile("model-Q4_K_M.gguf")},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := selectHuggingFaceGGUF(files, "IQ3_S"); err == nil {
				t.Fatal("invalid or ambiguous files accepted")
			}
		})
	}
	file := huggingFaceTestFile("model-IQ3_S.gguf")
	file.LFS.SHA256 = "invalid"
	if _, err := selectHuggingFaceGGUF([]huggingFaceFile{file}, "IQ3_S"); err == nil {
		t.Fatal("unverified file accepted")
	}
}

func TestHuggingFacePullRegistersBothShardsWithoutMergingAndReusesBlobs(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	var files []huggingFaceFile
	data := make(map[string][]byte)
	for index, tensor := range []string{"blk.0.attn_q.weight", "blk.1.attn_q.weight"} {
		digest := createSplitGGUFShard(t, uint32(index), 2, 2, tensor)
		blob, err := manifest.BlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		content, err := os.ReadFile(blob)
		if err != nil {
			t.Fatal(err)
		}
		filename := []string{"subfolder/model-IQ3_S-00001-of-00002.gguf", "subfolder/model-IQ3_S-00002-of-00002.gguf"}[index]
		file := huggingFaceTestFile(filename)
		file.Size = int64(len(content))
		file.LFS.Size = file.Size
		file.LFS.SHA256 = strings.TrimPrefix(digest, "sha256:")
		files = append(files, file)
		data[digest] = content
		if err := os.Remove(blob); err != nil {
			t.Fatal(err)
		}
	}
	var bodies atomic.Int32
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		digest := r.URL.Path[strings.LastIndex(r.URL.Path, "/")+1:]
		content, ok := data[digest]
		if !ok {
			http.NotFound(w, r)
			return
		}
		if r.Method == http.MethodGet {
			bodies.Add(1)
		}
		http.ServeContent(w, r, "model.gguf", time.Unix(1, 0), bytes.NewReader(content))
	}))
	defer cdn.Close()
	var pinnedRequests atomic.Int32
	revision := strings.Repeat("b", 40)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/models/") {
			if r.URL.Query().Get("blobs") != "true" {
				t.Error("SHA-256 metadata not requested")
			}
			if strings.Contains(r.URL.Path, "/revision/") {
				if !strings.HasSuffix(r.URL.Path, "/revision/"+revision) {
					t.Error("wrong pinned revision requested")
				}
				pinnedRequests.Add(1)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"sha": revision, "siblings": files})
			return
		}
		if strings.Contains(r.URL.Path, "/manifests/") {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "This repository only contains sharded GGUF files."})
			return
		}
		digest := r.URL.Path[strings.LastIndex(r.URL.Path, "/")+1:]
		if _, ok := data[digest]; ok {
			http.Redirect(w, r, cdn.URL+"/data/"+digest, http.StatusTemporaryRedirect)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()
	testMakeRequestDialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		if strings.HasPrefix(address, "huggingface.co:") {
			address = server.Listener.Addr().String()
		}
		return (&net.Dialer{}).DialContext(ctx, network, address)
	}
	t.Cleanup(func() { testMakeRequestDialContext = nil })
	name := model.ParseName("http://huggingface.co/owner/" + strings.Repeat("a", 96) + ":IQ3_S")
	cancelled, cancel := context.WithCancel(t.Context())
	none := "none"
	err := PullModel(cancelled, name.String(), "llamacpp", &registryOptions{Insecure: true, HuggingFace: huggingFaceSelection{File: files[0].Filename, Projector: &none}}, func(response api.ProgressResponse) {
		if response.FileIndex == 2 && strings.HasPrefix(response.Status, "downloading Hugging Face file") {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("download did not stop between shards: %v", err)
	}
	firstBlob, _ := manifest.BlobsPath("sha256:" + files[0].LFS.SHA256)
	secondBlob, _ := manifest.BlobsPath("sha256:" + files[1].LFS.SHA256)
	secondData := data["sha256:"+files[1].LFS.SHA256]
	half := len(secondData) / 2
	if err := os.WriteFile(secondBlob+"-partial", secondData[:half], 0644); err != nil {
		t.Fatal(err)
	}
	part, err := json.Marshal(jsonBlobDownloadPart{N: 0, Size: int64(len(secondData)), Completed: int64(half)})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(secondBlob+"-partial-0", part, 0644); err != nil {
		t.Fatal(err)
	}
	orphan, _ := manifest.BlobsPath("sha256:" + strings.Repeat("f", 64))
	if err := os.WriteFile(orphan, []byte("unused"), 0644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-48 * time.Hour)
	for _, path := range []string{firstBlob, secondBlob + "-partial", secondBlob + "-partial-0", orphan} {
		if err := os.Chtimes(path, old, old); err != nil {
			t.Fatal(err)
		}
	}
	if err := PruneLayers(); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{firstBlob, secondBlob + "-partial", secondBlob + "-partial-0"} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("startup prune discarded resumable download: %s: %v", path, err)
		}
	}
	if _, err := os.Stat(orphan); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("normal orphan was not pruned: %v", err)
	}
	for attempt := range 2 {
		options := &registryOptions{Insecure: true}
		if attempt == 1 {
			none := "none"
			options.HuggingFace = huggingFaceSelection{File: files[0].Filename, Projector: &none, Revision: revision}
		}
		var last int64
		seenSecond := false
		progress := func(response api.ProgressResponse) {
			if response.OverallTotal == 0 {
				return
			}
			if response.OverallTotal != files[0].Size+files[1].Size {
				t.Errorf("wrong overall total: %+v", response)
			}
			if response.OverallCompleted < last || response.OverallCompleted > response.OverallTotal {
				t.Errorf("aggregate progress regressed: %+v, previous %d", response, last)
			}
			last = response.OverallCompleted
			if response.FileIndex == 2 && response.FileCount == 2 {
				seenSecond = true
			}
		}
		if err := PullModel(t.Context(), name.String(), "llamacpp", options, progress); err != nil {
			t.Fatal(err)
		}
		if !seenSecond || last != files[0].Size+files[1].Size {
			t.Fatal("incomplete aggregate progress")
		}
		if pins, err := manifest.DownloadPinnedBlobs(); err != nil || len(pins) != 0 {
			t.Fatalf("completed model retained download checkpoint: %+v %v", pins, err)
		}
	}
	if pinnedRequests.Load() != 1 {
		t.Fatal("exact selection did not pin the catalog revision")
	}
	if bodies.Load() != 2 {
		t.Fatalf("blob GETs=%d, want 2 with cache reuse", bodies.Load())
	}
	created, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, layer := range created.Layers {
		if layer.MediaType != "application/vnd.ollama.image.model" {
			continue
		}
		count++
		blob, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			t.Fatal(err)
		}
		content, err := os.ReadFile(blob)
		if err != nil || !bytes.Equal(content, data[layer.Digest]) {
			t.Fatal("GGUF bytes changed during import", err)
		}
	}
	if count != 2 {
		t.Fatalf("model has %d shards, want 2", count)
	}
}
