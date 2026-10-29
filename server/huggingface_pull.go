package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
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

func normalizeHuggingFaceFile(file huggingFaceFile) huggingFaceFile {
	if file.Size == 0 && file.LFS.Size > 0 {
		file.Size = file.LFS.Size
	}
	return file
}

func validateHuggingFaceFile(file huggingFaceFile) error {
	if file.Size <= 0 || file.Size != file.LFS.Size || manifest.ValidateDigest("sha256:"+file.LFS.SHA256) != nil {
		return fmt.Errorf("missing or inconsistent SHA-256/size for %q", file.Filename)
	}
	return nil
}

func huggingFaceFileGroup(filename string) string {
	if prefix, _, _, split := splitGGUFName(filename); split {
		return path.Join(path.Dir(filename), prefix)
	}
	return filename
}

func validateHuggingFaceProjector(files []huggingFaceFile) error {
	for _, file := range files {
		if err := validateHuggingFaceFile(file); err != nil {
			return err
		}
		if _, _, count, split := splitGGUFName(file.Filename); split && count > 1 {
			return errors.New("split vision projectors are not supported; choose a single-file mmproj GGUF")
		}
	}
	return nil
}

func selectHuggingFaceGGUF(files []huggingFaceFile, tag string, choices ...huggingFaceSelection) ([]huggingFaceFile, error) {
	var choice huggingFaceSelection
	if len(choices) > 0 {
		choice = choices[0]
	}
	if err := choice.validate(); err != nil {
		return nil, err
	}
	groups := make(map[string][]huggingFaceFile)
	projectorGroups := map[string][]huggingFaceFile{}
	for _, file := range files {
		file = normalizeHuggingFaceFile(file)
		if !strings.HasSuffix(strings.ToLower(file.Filename), ".gguf") {
			continue
		}
		if err := validateCreateFilePath(file.Filename); err != nil || strings.Contains(file.Filename, "\\") {
			return nil, fmt.Errorf("invalid Hugging Face GGUF path: %q", file.Filename)
		}
		if strings.HasPrefix(strings.ToLower(path.Base(file.Filename)), "mmproj") {
			key := huggingFaceFileGroup(file.Filename)
			projectorGroups[key] = append(projectorGroups[key], file)
			continue
		}
		key := huggingFaceFileGroup(file.Filename)
		groups[key] = append(groups[key], file)
	}
	var selected []huggingFaceFile
	var quant *regexp.Regexp
	if choice.File == "" {
		quant = regexp.MustCompile(`(?i)(?:^|[-_.])` + regexp.QuoteMeta(tag) + `(?:[-_.]|$)`)
	}
	for key, group := range groups {
		match := quant != nil && quant.MatchString(key)
		for _, file := range group {
			match = match || strings.EqualFold(file.Filename, tag)
		}
		if choice.File != "" {
			match = slices.ContainsFunc(group, func(file huggingFaceFile) bool { return file.Filename == choice.File })
		} else if strings.EqualFold(tag, "latest") {
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
	var projectors []huggingFaceFile
	if choice.Projector != nil {
		projector := *choice.Projector
		if projector == "none" || projector == "" {
			projectors = nil
		} else {
			for _, group := range projectorGroups {
				if slices.ContainsFunc(group, func(file huggingFaceFile) bool { return file.Filename == projector }) {
					projectors = group
					break
				}
			}
			if len(projectors) == 0 {
				return nil, fmt.Errorf("Hugging Face vision projector not found: %q", projector)
			}
		}
	} else if len(projectorGroups) > 1 {
		return nil, errors.New("multiple Hugging Face vision projectors; import the selected files explicitly")
	} else {
		for _, group := range projectorGroups {
			projectors = group
		}
	}
	if err := validateHuggingFaceProjector(projectors); err != nil {
		return nil, err
	}
	selected = append(selected, projectors...)
	if len(selected) > maxCreateFiles {
		return nil, errors.New("too many Hugging Face GGUF files")
	}
	for _, file := range selected {
		if err := validateHuggingFaceFile(file); err != nil {
			return nil, err
		}
	}
	return selected, nil
}

func pullHuggingFaceGGUF(ctx context.Context, name model.Name, runner string, options *registryOptions, fn func(api.ProgressResponse)) error {
	if name.ProtocolScheme == "http" && !options.Insecure {
		return errInsecureProtocol
	}
	if runner != "" && runner != "llamacpp" {
		return fmt.Errorf("Hugging Face GGUF requires the llamacpp runner, not %q", runner)
	}
	fn(api.ProgressResponse{Status: "resolving Hugging Face GGUF shards"})
	repository, err := readHuggingFaceRepository(ctx, name, options)
	if err != nil {
		return err
	}
	files, err := selectHuggingFaceGGUF(repository.Files, name.Tag, options.HuggingFace)
	if err != nil {
		return err
	}
	oldDigests, _ := manifest.ReferencedBlobDigestsForName(name)
	local := make(map[string]string, len(files))
	var total, completed int64
	for _, file := range files {
		if file.Size > math.MaxInt64-total {
			return errors.New("Hugging Face model size exceeds supported range")
		}
		total += file.Size
	}
	digests := make([]string, 0, len(files))
	for _, file := range files {
		digests = append(digests, "sha256:"+file.LFS.SHA256)
	}
	if err := manifest.PinDownload(name, digests); err != nil {
		return err
	}
	for index, file := range files {
		report := func(progress api.ProgressResponse) {
			progress.OverallTotal = total
			progress.OverallCompleted = completed + max(0, min(progress.Completed, file.Size))
			progress.File, progress.FileIndex, progress.FileCount = file.Filename, index+1, len(files)
			fn(progress)
		}
		report(api.ProgressResponse{Status: fmt.Sprintf("downloading Hugging Face file %d/%d: %s", index+1, len(files), file.Filename)})
		digest := "sha256:" + file.LFS.SHA256
		cached, err := downloadBlob(ctx, downloadOpts{n: name, digest: digest, regOpts: options, fn: report})
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
			report(api.ProgressResponse{Status: "verifying sha256 digest", Digest: digest, Total: file.Size, Completed: file.Size})
			if err := verifyBlobContext(ctx, digest); err != nil {
				if errors.Is(err, errDigestMismatch) {
					_ = os.Remove(blob)
					removeGGUFMetadata(digest)
				}
				return err
			}
		}
		local[file.Filename] = digest
		completed += file.Size
	}
	finishedDownload := func(progress api.ProgressResponse) {
		progress.OverallTotal, progress.OverallCompleted = total, completed
		fn(progress)
	}
	finishedDownload(api.ProgressResponse{Status: "importing downloaded Hugging Face model"})
	layers, err := convertModelFromFiles(ctx, local, finishedDownload)
	if err != nil {
		return err
	}
	if err := createModel(ctx, api.CreateRequest{Model: name.String(), Files: local}, name, layers, new(model.ConfigV2), finishedDownload); err != nil {
		return err
	}
	if err := manifest.ReleaseDownloadPin(name); err != nil {
		slog.Warn("could not release completed Hugging Face download", "model", name.String(), "error", err)
	}
	if !envconfig.NoPrune() && len(oldDigests) > 0 {
		removed, err := manifest.RemoveUnreferencedBlobs(oldDigests...)
		removeGGUFMetadata(removed...)
		if err != nil {
			return err
		}
	}
	finishedDownload(api.ProgressResponse{Status: "success"})
	return nil
}
