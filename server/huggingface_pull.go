package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"regexp"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

type huggingFaceFile struct {
	Filename string `json:"rfilename"`
	Size     int64  `json:"size"`
	LFS      struct {
		SHA256 string `json:"sha256"`
		Size   int64  `json:"size"`
	} `json:"lfs"`
}

func selectHuggingFaceGGUF(files []huggingFaceFile, tag string) ([]huggingFaceFile, error) {
	groups := make(map[string][]huggingFaceFile)
	var projectors []huggingFaceFile
	for _, file := range files {
		if !strings.HasSuffix(strings.ToLower(file.Filename), ".gguf") {
			continue
		}
		if err := validateCreateFilePath(file.Filename); err != nil || strings.Contains(file.Filename, "\\") {
			return nil, fmt.Errorf("invalid Hugging Face GGUF path: %q", file.Filename)
		}
		if strings.HasPrefix(strings.ToLower(path.Base(file.Filename)), "mmproj") {
			projectors = append(projectors, file)
			continue
		}
		key := file.Filename
		if prefix, _, _, split := splitGGUFName(file.Filename); split {
			key = path.Join(path.Dir(file.Filename), prefix)
		}
		groups[key] = append(groups[key], file)
	}
	var selected []huggingFaceFile
	quant := regexp.MustCompile(`(?i)(?:^|[-_.])` + regexp.QuoteMeta(tag) + `(?:[-_.]|$)`)
	for key, group := range groups {
		match := quant.MatchString(key)
		for _, file := range group {
			match = match || strings.EqualFold(file.Filename, tag)
		}
		if strings.EqualFold(tag, "latest") {
			match = len(groups) == 1 || strings.Contains(strings.ToUpper(key), "Q4_K_M")
		}
		if !match {
			continue
		}
		if selected != nil {
			return nil, errors.New("multiple GGUF variants match this tag; specify the GGUF filename")
		}
		selected = group
	}
	if len(selected) == 0 {
		return nil, fmt.Errorf("no Hugging Face GGUF files match %q", tag)
	}
	slices.SortFunc(selected, func(a, b huggingFaceFile) int { return strings.Compare(a.Filename, b.Filename) })
	for position, file := range selected {
		if _, index, count, split := splitGGUFName(file.Filename); split {
			if int(count) != len(selected) || int(index) != position {
				return nil, fmt.Errorf("%w: missing or duplicate Hugging Face shard", errInvalidSplitGGUF)
			}
		}
	}
	if len(projectors) > 1 {
		return nil, errors.New("multiple Hugging Face vision projectors; import the selected files explicitly")
	}
	selected = append(selected, projectors...)
	if len(selected) > maxCreateFiles {
		return nil, errors.New("too many Hugging Face GGUF files")
	}
	for _, file := range selected {
		if file.Size <= 0 || file.Size != file.LFS.Size || manifest.ValidateDigest("sha256:"+file.LFS.SHA256) != nil {
			return nil, fmt.Errorf("missing or inconsistent SHA-256/size for %q", file.Filename)
		}
	}
	return selected, nil
}

func pullHuggingFaceGGUF(ctx context.Context, name model.Name, runner string, options *registryOptions, fn func(api.ProgressResponse)) error {
	if runner != "" && runner != "llamacpp" {
		return fmt.Errorf("Hugging Face GGUF requires the llamacpp runner, not %q", runner)
	}
	fn(api.ProgressResponse{Status: "resolving Hugging Face GGUF shards"})
	requestURL := name.BaseURL().JoinPath("api/models", name.Namespace, name.Model)
	requestURL.RawQuery = "blobs=true"
	response, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, nil, nil, options)
	if err != nil {
		return fmt.Errorf("read Hugging Face file list: %w", err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, (8<<20)+1))
	if err != nil {
		return err
	}
	if len(data) > 8<<20 {
		return errors.New("Hugging Face file list is too large")
	}
	var repository struct {
		Files []huggingFaceFile `json:"siblings"`
	}
	if err := json.Unmarshal(data, &repository); err != nil {
		return err
	}
	files, err := selectHuggingFaceGGUF(repository.Files, name.Tag)
	if err != nil {
		return err
	}
	oldDigests, _ := manifest.ReferencedBlobDigestsForName(name)
	local := make(map[string]string, len(files))
	for _, file := range files {
		digest := "sha256:" + file.LFS.SHA256
		cached, err := downloadBlob(ctx, downloadOpts{n: name, digest: digest, regOpts: options, fn: fn})
		if err != nil {
			return err
		}
		blob, err := manifest.BlobsPath(digest)
		if err != nil {
			return err
		}
		info, err := os.Stat(blob)
		if err != nil {
			return err
		}
		if info.Size() != file.Size {
			return fmt.Errorf("unexpected size for Hugging Face GGUF %q", file.Filename)
		}
		if !cached {
			fn(api.ProgressResponse{Status: "verifying sha256 digest", Digest: digest})
			if err := verifyBlob(digest); err != nil {
				if errors.Is(err, errDigestMismatch) {
					_ = os.Remove(blob)
					removeGGUFMetadata(digest)
				}
				return err
			}
		}
		local[file.Filename] = digest
	}
	layers, err := convertModelFromFiles(ctx, local, fn)
	if err != nil {
		return err
	}
	if err := createModel(ctx, api.CreateRequest{Model: name.String(), Files: local}, name, layers, new(model.ConfigV2), fn); err != nil {
		return err
	}
	if !envconfig.NoPrune() && len(oldDigests) > 0 {
		removed, err := manifest.RemoveUnreferencedBlobs(oldDigests...)
		removeGGUFMetadata(removed...)
		if err != nil {
			return err
		}
	}
	fn(api.ProgressResponse{Status: "success"})
	return nil
}
