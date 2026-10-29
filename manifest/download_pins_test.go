package manifest

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

func TestDownloadPinRetainsUnregisteredBlobsAndCanReplaceSelection(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	first, err := NewLayer(strings.NewReader("first selected weights"), MediaTypeImageModel)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewLayer(strings.NewReader("second selected weights"), MediaTypeImageModel)
	if err != nil {
		t.Fatal(err)
	}
	name := model.ParseName("huggingface.co/Owner/Model:Q4_K_M")
	if err := PinDownload(name, []string{first.Digest, first.Digest}); err != nil {
		t.Fatal(err)
	}
	if removed, err := RemoveUnreferencedBlobs(first.Digest); err != nil || len(removed) != 0 {
		t.Fatalf("pending weights removed: %+v %v", removed, err)
	}
	if err := PinDownload(name, []string{second.Digest}); err != nil {
		t.Fatal(err)
	}
	if removed, err := RemoveUnreferencedBlobs(first.Digest, second.Digest); err != nil || len(removed) != 1 || removed[0] != first.Digest {
		t.Fatalf("selection pin not replaced: %+v %v", removed, err)
	}
	if err := ReleaseDownloadPin(model.ParseName(strings.ToLower(name.String()))); err != nil {
		t.Fatal(err)
	}
	if pins, err := DownloadPinnedBlobs(); err != nil || len(pins) != 0 {
		t.Fatalf("case alias left pending download: %+v %v", pins, err)
	}
}

func TestRemoveNamedReleasesOwnPinWhileKeepingSharedModelWeights(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	weights, err := NewLayer(strings.NewReader("shared weights"), MediaTypeImageModel)
	if err != nil {
		t.Fatal(err)
	}
	config, err := NewLayer(strings.NewReader(`{}`), MediaTypeImageConfig)
	if err != nil {
		t.Fatal(err)
	}
	first, second := model.ParseName("first"), model.ParseName("second")
	for _, name := range []model.Name{first, second} {
		if err := WriteManifest(name, config, []Layer{weights}); err != nil {
			t.Fatal(err)
		}
	}
	if err := PinDownload(first, []string{weights.Digest}); err != nil {
		t.Fatal(err)
	}
	if _, err := RemoveNamed(first); err != nil {
		t.Fatal(err)
	}
	if pins, err := DownloadPinnedBlobs(); err != nil || len(pins) != 0 {
		t.Fatalf("deleted model kept workflow pin: %+v %v", pins, err)
	}
	path, _ := BlobsPath(weights.Digest)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("shared model lost weights: %v", err)
	}
	if _, err := RemoveNamed(second); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("last reference retained weights: %v", err)
	}
}

func TestInvalidDownloadPinsCannotRetainArbitraryPathsOrHidePruneErrors(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := model.ParseName("owner/model")
	if err := PinDownload(name, []string{"../../arbitrary-path"}); err == nil {
		t.Fatal("invalid digest accepted")
	}
	if err := PinDownload(name, []string{"sha256:" + strings.Repeat("a", 64)}); err != nil {
		t.Fatal(err)
	}
	path := downloadPinPath(name)
	if !strings.HasPrefix(path, filepath.Join(envconfig.Models(), "metadata")) {
		t.Fatal("download metadata escaped model store")
	}
	if err := os.WriteFile(path, []byte(`{"model":"owner/model","digests":["../../bad"]}`), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := DownloadPinnedBlobs(); err == nil {
		t.Fatal("invalid pin silently trusted")
	}
	if _, err := RemoveUnreferencedBlobs("sha256:" + strings.Repeat("a", 64)); err == nil {
		t.Fatal("pruning proceeded with corrupt retention metadata")
	}
}
