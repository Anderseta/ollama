package llm

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/ollama/ollama/ml"
)

var tuningPayloadDigests sync.Map

type tuningPayloadDigest struct {
	infos []os.FileInfo
	id    string
}

// A policy is local administrator state, never a path supplied by a model or
// API request. Missing identity, cache, or native support preserves dispatch.
func setCUDATuningEnvironment(env map[string]string, exe string, gpus []ml.DeviceInfo, gpuLibs []string) {
	env["GGML_CUDA_TUNING_CACHE"] = ""
	env["GGML_CUDA_TUNING_FINGERPRINT"] = ""
	directory := os.Getenv("WILDLLAMA_CUDA_TUNING_DIR")
	if directory == "" || !filepath.IsAbs(directory) || len(gpus) != 1 || !strings.EqualFold(gpus[0].Library, "cuda") {
		return
	}
	var backend string
	for _, directory := range gpuLibs {
		if candidate := findLlamaServerGPUBackend(directory); strings.Contains(strings.ToLower(filepath.Base(candidate)), "ggml-cuda") {
			backend = candidate
			break
		}
	}
	if backend == "" {
		return
	}
	payload, err := cudaTuningPayloadID(exe, backend)
	if err != nil {
		slog.Debug("CUDA tuning policy unavailable", "error", err)
		return
	}
	fingerprint, ok := gpus[0].CapabilityFingerprint(payload)
	if !ok {
		return
	}
	policy := filepath.Join(directory, fingerprint+".policy")
	info, err := os.Stat(policy)
	if err != nil || !info.Mode().IsRegular() || info.Size() > 64<<10 {
		slog.Debug("CUDA tuning policy not calibrated", "fingerprint", fingerprint, "file", policy)
		return
	}
	env["GGML_CUDA_TUNING_CACHE"] = policy
	env["GGML_CUDA_TUNING_FINGERPRINT"] = fingerprint
}

// Hash every runtime library in the two selected payload directories, not only
// the small launcher executable. Memoization avoids rereading CUDA libraries
// on each model reload; replacing any file invalidates the process-local entry.
func cudaTuningPayloadID(exe, backend string) (string, error) {
	files := []string{exe}
	for _, directory := range []string{filepath.Dir(exe), filepath.Dir(backend)} {
		entries, err := os.ReadDir(directory)
		if err != nil {
			return "", err
		}
		for _, entry := range entries {
			name := strings.ToLower(entry.Name())
			if !entry.IsDir() && (strings.HasSuffix(name, ".dll") || strings.Contains(name, ".so") || strings.HasSuffix(name, ".dylib")) {
				files = append(files, filepath.Join(directory, entry.Name()))
			}
		}
	}
	files = append(files, backend)
	slices.Sort(files)
	files = slices.Compact(files)
	var key strings.Builder
	infos := make([]os.FileInfo, len(files))
	for i, path := range files {
		info, err := os.Stat(path)
		if err != nil || !info.Mode().IsRegular() {
			return "", fmt.Errorf("native tuning payload is unavailable: %s", path)
		}
		infos[i] = info
		fmt.Fprintf(&key, "%q:%d:%d;", path, info.Size(), info.ModTime().UnixNano())
	}
	if cached, ok := tuningPayloadDigests.Load(key.String()); ok {
		previous := cached.(tuningPayloadDigest)
		same := len(previous.infos) == len(infos)
		for i := range infos {
			same = same && os.SameFile(previous.infos[i], infos[i])
		}
		if same {
			return previous.id, nil
		}
	}
	hash := sha256.New()
	for i, path := range files {
		file, err := os.Open(path)
		if err != nil {
			return "", err
		}
		before, err := file.Stat()
		if err == nil && !os.SameFile(infos[i], before) {
			file.Close()
			return "", fmt.Errorf("native tuning payload replaced while hashing: %s", path)
		}
		if err == nil {
			fmt.Fprintf(hash, "%q:%d:", filepath.Base(path), before.Size())
			_, err = io.Copy(hash, file)
		}
		after, statErr := file.Stat()
		closeErr := file.Close()
		if err != nil || statErr != nil || closeErr != nil {
			return "", fmt.Errorf("cannot hash native tuning payload: %s", path)
		}
		if before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
			return "", fmt.Errorf("native tuning payload changed while hashing: %s", path)
		}
	}
	for i, path := range files {
		info, err := os.Stat(path)
		if err != nil || !os.SameFile(infos[i], info) || info.Size() != infos[i].Size() || !info.ModTime().Equal(infos[i].ModTime()) {
			return "", fmt.Errorf("native tuning payload changed while hashing: %s", path)
		}
	}
	payload := hex.EncodeToString(hash.Sum(nil))
	tuningPayloadDigests.Store(key.String(), tuningPayloadDigest{infos, payload})
	return payload, nil
}
