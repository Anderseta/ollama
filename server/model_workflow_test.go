package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

type modelHashFixtureReader struct {
	cancel context.CancelFunc
	err    error
	reads  int
}

func (r *modelHashFixtureReader) Read(buffer []byte) (int, error) {
	r.reads++
	if r.err != nil {
		return 0, r.err
	}
	clear(buffer)
	r.cancel()
	return len(buffer), nil
}

func TestModelBlobHashCancelsDuringReadAndPropagatesIOErrors(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	reader := &modelHashFixtureReader{cancel: cancel}
	digest, count, err := hashModelBlob(ctx, reader)
	if !errors.Is(err, context.Canceled) || reader.reads != 1 || count > 1024*1024 || digest != "" {
		t.Fatalf("hash ignored cancellation: digest=%s bytes=%d error=%v reads=%d", digest, count, err, reader.reads)
	}
	expected := errors.New("fixture disk read failed")
	if _, _, err := GetSHA256Digest(&modelHashFixtureReader{err: expected}); !errors.Is(err, expected) {
		t.Fatalf("disk read error was discarded: %v", err)
	}
	if err := verifyBlobContext(ctx, "sha256:"+strings.Repeat("a", 64)); !errors.Is(err, context.Canceled) {
		t.Fatalf("verification performed filesystem work after cancellation: %v", err)
	}
	digest, count, err = hashModelBlob(t.Context(), bytes.NewBufferString("abc"))
	if err != nil || digest != "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" || count != 3 {
		t.Fatalf("hash changed: %s %d %v", digest, count, err)
	}
}

type cancelDuringBlobHash struct {
	context.Context
	cancel context.CancelFunc
	checks int
}

func (ctx *cancelDuringBlobHash) Err() error {
	ctx.checks++
	if ctx.checks == 4 {
		ctx.cancel()
	}
	return ctx.Context.Err()
}

func TestCancelledBlobVerificationRemainsUntrustedAcrossResume(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	data := bytes.Repeat([]byte("source model bytes"), 128*1024)
	digest, _, err := GetSHA256Digest(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	path, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	corrupted := bytes.Repeat([]byte("X"), len(data))
	if err := os.WriteFile(path, corrupted, 0644); err != nil {
		t.Fatal(err)
	}
	if err := markBlobVerificationPending(digest); err != nil {
		t.Fatal(err)
	}
	base, cancel := context.WithCancel(t.Context())
	ctx := &cancelDuringBlobHash{Context: base, cancel: cancel}
	if err := verifyBlobContext(ctx, digest); !errors.Is(err, context.Canceled) {
		t.Fatalf("verification did not stop: %v", err)
	}
	if info, err := os.Stat(path); err != nil || info.Size() != int64(len(data)) {
		t.Fatalf("cancelled verification discarded downloaded bytes: %+v %v", info, err)
	}
	options := downloadOpts{n: model.ParseName("huggingface.co/owner/model"), digest: digest, regOpts: &registryOptions{}, fn: func(api.ProgressResponse) {}}
	if cached, err := downloadBlob(t.Context(), options); err != nil || cached {
		t.Fatalf("unverified cached blob trusted on resume: %v %v", cached, err)
	}
	if err := verifyBlobContext(t.Context(), digest); !errors.Is(err, errDigestMismatch) {
		t.Fatalf("resumed verification skipped corrupt bytes: %v", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	if err := verifyBlobContext(t.Context(), digest); err != nil {
		t.Fatal(err)
	}
	if pending, err := blobVerificationPending(digest); err != nil || pending {
		t.Fatalf("successful verification not remembered: %v %v", pending, err)
	}
	if cached, err := downloadBlob(t.Context(), options); err != nil || !cached {
		t.Fatalf("verified blob unnecessarily redownloaded/rehashed: %v %v", cached, err)
	}
}

func TestCancelledPullDoesNotPublishModelAfterDownload(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	ctx, cancel := context.WithCancel(t.Context())
	name := model.ParseName("owner/model")
	err := pullWithTransfer(ctx, name, nil, []byte(`{"schemaVersion":2,"layers":[]}`), &registryOptions{}, func(progress api.ProgressResponse) {
		if progress.Status == "writing manifest" {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled model was published: %v", err)
	}
	if _, err := manifest.ReadManifestData(name); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("cancelled publication left manifest: %v", err)
	}
}

func TestHuggingFaceMetadataCacheCoalescesAndKeepsOtherWaitersAlive(t *testing.T) {
	cache := newHuggingFaceMetadataCache()
	ctx, cancel := context.WithCancel(t.Context())
	started, release := make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	load := func(ctx context.Context) ([]byte, error) {
		calls.Add(1)
		close(started)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-release:
			return []byte("same metadata"), nil
		}
	}
	first := make(chan error, 1)
	go func() { _, err := cache.get(ctx, "repo@sha", time.Minute, load); first <- err }()
	<-started
	second := make(chan error, 1)
	go func() {
		value, err := cache.get(t.Context(), "repo@sha", time.Minute, load)
		if err == nil && string(value) != "same metadata" {
			err = fmt.Errorf("wrong shared result: %s", value)
		}
		second <- err
	}()
	deadline := time.Now().Add(time.Second)
	for {
		cache.mu.Lock()
		joined := cache.requests["repo@sha"].waiters == 2
		cache.mu.Unlock()
		if joined {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("waiter did not join")
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	if err := <-first; !errors.Is(err, context.Canceled) {
		t.Fatalf("waiter did not cancel: %v", err)
	}
	close(release)
	if err := <-second; err != nil {
		t.Fatal(err)
	}
	value, err := cache.get(t.Context(), "repo@sha", time.Minute, func(context.Context) ([]byte, error) { t.Error("cached response reloaded"); return nil, io.EOF })
	if err != nil || string(value) != "same metadata" || calls.Load() != 1 {
		t.Fatalf("coalescing/cache failed: %s %v %d", value, err, calls.Load())
	}
}

func TestHuggingFaceMetadataCacheCancelsAbandonedFetchAndNeverCachesFailures(t *testing.T) {
	cache := newHuggingFaceMetadataCache()
	ctx, cancel := context.WithCancel(t.Context())
	started, stopped := make(chan struct{}), make(chan struct{})
	result := make(chan error, 1)
	go func() {
		_, err := cache.get(ctx, "repo", time.Minute, func(ctx context.Context) ([]byte, error) {
			close(started)
			<-ctx.Done()
			close(stopped)
			return nil, ctx.Err()
		})
		result <- err
	}()
	<-started
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("abandoned network work continued")
	}
	value, err := cache.get(t.Context(), "repo", time.Minute, func(context.Context) ([]byte, error) { return []byte("fresh"), nil })
	if err != nil || string(value) != "fresh" {
		t.Fatalf("cancelled request poisoned retry: %s %v", value, err)
	}
	_, err = cache.get(t.Context(), "failure", time.Minute, func(context.Context) ([]byte, error) { return nil, io.EOF })
	if !errors.Is(err, io.EOF) {
		t.Fatal(err)
	}
	if _, err = cache.get(t.Context(), "failure", time.Minute, func(context.Context) ([]byte, error) { return []byte("retry"), nil }); err != nil {
		t.Fatal(err)
	}
}

func TestHuggingFaceMetadataCacheBoundsStorageAndSeparatesRevisions(t *testing.T) {
	cache := newHuggingFaceMetadataCache()
	for i := range 40 {
		key := fmt.Sprintf("repo@%d", i)
		_, err := cache.get(t.Context(), key, time.Minute, func(context.Context) ([]byte, error) { return make([]byte, 512*1024), nil })
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(cache.entries) > 32 || cache.bytes > huggingFaceCacheBytes {
		t.Fatalf("unbounded cache: %d %d", len(cache.entries), cache.bytes)
	}
	cache.mu.Lock()
	entry := cache.entries["repo@39"]
	entry.expires = time.Now().Add(-time.Second)
	cache.entries["repo@39"] = entry
	cache.mu.Unlock()
	value, err := cache.get(t.Context(), "repo@39", time.Minute, func(context.Context) ([]byte, error) { return []byte("updated"), nil })
	if err != nil || string(value) != "updated" {
		t.Fatalf("expired metadata not refreshed: %s %v", value, err)
	}
}

func TestHuggingFaceMetadataCacheLimitsConcurrentNetworkWork(t *testing.T) {
	cache := newHuggingFaceMetadataCache()
	entered, release, results := make(chan struct{}, 8), make(chan struct{}), make(chan error, 8)
	for i := range 8 {
		go func() {
			_, err := cache.get(t.Context(), fmt.Sprint(i), time.Minute, func(ctx context.Context) ([]byte, error) {
				entered <- struct{}{}
				select {
				case <-release:
					return []byte("metadata"), nil
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			})
			results <- err
		}()
	}
	for range 8 {
		select {
		case <-entered:
		case <-time.After(time.Second):
			t.Fatal("metadata request did not start")
		}
	}
	_, err := cache.get(t.Context(), "overflow", time.Minute, func(context.Context) ([]byte, error) { t.Error("overflow reached network"); return nil, nil })
	if !errors.Is(err, errHuggingFaceCatalogBusy) {
		t.Fatalf("metadata work not bounded: %v", err)
	}
	close(release)
	for range 8 {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
}

func TestHuggingFaceUppercaseSplitFilesAndSplitProjectorsArePreflighted(t *testing.T) {
	files := []huggingFaceFile{huggingFaceTestFile("model-Q4_K_M-00001-of-00002.GGUF"), huggingFaceTestFile("model-Q4_K_M-00002-of-00002.GGUF"), huggingFaceTestFile("mmproj-F16-00001-of-00002.gguf"), huggingFaceTestFile("mmproj-F16-00002-of-00002.gguf")}
	catalog := huggingFaceVariants("owner/model", huggingFaceRepository{Files: files})
	if len(catalog.Variants) != 1 || !catalog.Variants[0].Complete || catalog.Variants[0].Shards != 2 {
		t.Fatalf("uppercase model shards not grouped: %+v", catalog.Variants)
	}
	if len(catalog.Projectors) != 1 || *catalog.Projectors[0].Complete || catalog.Projectors[0].Shards != 2 || !strings.Contains(catalog.Projectors[0].Error, "split vision projectors") {
		t.Fatalf("unsupported projector passed preflight: %+v", catalog.Projectors)
	}
	projector := files[2].Filename
	if _, err := selectHuggingFaceGGUF(files, "Q4_K_M", huggingFaceSelection{File: files[0].Filename, Projector: &projector}); err == nil || !strings.Contains(err.Error(), "split vision projectors") {
		t.Fatalf("unsupported projector was downloaded: %v", err)
	}
	none := "none"
	if selected, err := selectHuggingFaceGGUF(files, "Q4_K_M", huggingFaceSelection{File: files[0].Filename, Projector: &none}); err != nil || len(selected) != 2 {
		t.Fatalf("text model blocked by optional projector: %+v %v", selected, err)
	}
}
