package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/types/model"
)

type huggingFaceSelection struct {
	File      string
	Projector *string
	Revision  string
}

func (s huggingFaceSelection) explicit() bool {
	return s.File != "" || s.Projector != nil || s.Revision != ""
}

var huggingFaceRevision = regexp.MustCompile(`^[0-9a-fA-F]{40,64}$`)
var huggingFaceRepositoryID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)
var huggingFaceQuantization = regexp.MustCompile(`(?i)(?:^|[-_.])(IQ[1-4]_[A-Z0-9_]+|Q[0-9](?:_[A-Z0-9_]+)?|TQ[12]_0|BF16|FP16|F16|F32|MXFP4)(?:[-.]|$)`)

func isHuggingFaceName(name model.Name) bool {
	return strings.EqualFold(name.Host, "huggingface.co") || strings.EqualFold(name.Host, "hf.co")
}

func (s huggingFaceSelection) validate() error {
	for _, filename := range []string{s.File, valueOrEmpty(s.Projector)} {
		if filename == "" || filename == "none" {
			continue
		}
		if err := validateCreateFilePath(filename); err != nil || strings.Contains(filename, "\\") || !strings.HasSuffix(strings.ToLower(filename), ".gguf") {
			return fmt.Errorf("invalid Hugging Face GGUF selection: %q", filename)
		}
	}
	if s.Revision != "" && !huggingFaceRevision.MatchString(s.Revision) {
		return errors.New("Hugging Face revision must be a commit SHA")
	}
	return nil
}

func valueOrEmpty(value *string) string {
	if value != nil {
		return *value
	}
	return ""
}

type huggingFaceRepository struct {
	SHA   string            `json:"sha"`
	Files []huggingFaceFile `json:"siblings"`
}

func readHuggingFaceJSON(ctx context.Context, target *url.URL, options *registryOptions, value any) error {
	response, err := makeRequestWithRetry(ctx, http.MethodGet, target, nil, nil, options)
	if err != nil {
		return fmt.Errorf("Hugging Face: %w", err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, (8<<20)+1))
	if err != nil {
		return err
	}
	if len(data) > 8<<20 {
		return errors.New("Hugging Face file list is too large")
	}
	if err := json.Unmarshal(data, value); err != nil {
		return fmt.Errorf("invalid Hugging Face response: %w", err)
	}
	return nil
}

func readHuggingFaceRepository(ctx context.Context, name model.Name, options *registryOptions) (huggingFaceRepository, error) {
	var repository huggingFaceRepository
	if err := options.HuggingFace.validate(); err != nil {
		return repository, err
	}
	target := name.BaseURL().JoinPath("api/models", name.Namespace, name.Model)
	if revision := options.HuggingFace.Revision; revision != "" {
		target = target.JoinPath("revision", revision)
	}
	target.RawQuery = "blobs=true"
	err := readHuggingFaceJSON(ctx, target, options, &repository)
	if err == nil && options.HuggingFace.Revision != "" && repository.SHA != "" && !strings.EqualFold(repository.SHA, options.HuggingFace.Revision) {
		err = errors.New("Hugging Face repository revision changed; refresh available files")
	}
	return repository, err
}

type huggingFaceCatalogFile struct {
	Filename string                   `json:"filename"`
	Size     int64                    `json:"size"`
	Complete *bool                    `json:"complete,omitempty"`
	Error    string                   `json:"error,omitempty"`
	Shards   int                      `json:"shards,omitempty"`
	Files    []huggingFaceCatalogFile `json:"files,omitempty"`
}

type huggingFaceVariant struct {
	Filename     string                   `json:"filename"`
	Quantization string                   `json:"quantization"`
	Files        []huggingFaceCatalogFile `json:"files"`
	Size         int64                    `json:"size"`
	Shards       int                      `json:"shards"`
	Complete     bool                     `json:"complete"`
	Error        string                   `json:"error,omitempty"`
}

type huggingFaceCatalog struct {
	Repository       string                   `json:"repository"`
	Revision         string                   `json:"revision"`
	Variants         []huggingFaceVariant     `json:"variants"`
	Projectors       []huggingFaceCatalogFile `json:"projectors"`
	UnsupportedFiles int                      `json:"unsupported_files"`
}

func huggingFaceVariants(repository string, source huggingFaceRepository) huggingFaceCatalog {
	result := huggingFaceCatalog{Repository: repository, Revision: source.SHA, Variants: []huggingFaceVariant{}, Projectors: []huggingFaceCatalogFile{}}
	groups := map[string][]huggingFaceFile{}
	projectorGroups := map[string][]huggingFaceFile{}
	for _, file := range source.Files {
		file = normalizeHuggingFaceFile(file)
		if !strings.HasSuffix(strings.ToLower(file.Filename), ".gguf") {
			result.UnsupportedFiles++
			continue
		}
		if err := validateCreateFilePath(file.Filename); err != nil || strings.Contains(file.Filename, "\\") {
			continue
		}
		if strings.HasPrefix(strings.ToLower(path.Base(file.Filename)), "mmproj") {
			key := huggingFaceFileGroup(file.Filename)
			projectorGroups[key] = append(projectorGroups[key], file)
			continue
		}
		key := huggingFaceFileGroup(file.Filename)
		groups[key] = append(groups[key], file)
	}
	for _, files := range projectorGroups {
		slices.SortFunc(files, func(a, b huggingFaceFile) int { return strings.Compare(a.Filename, b.Filename) })
		err := validateHuggingFaceProjector(files)
		complete := err == nil
		projector := huggingFaceCatalogFile{Filename: files[0].Filename, Complete: &complete, Shards: len(files)}
		for _, file := range files {
			projector.Size += file.Size
			projector.Files = append(projector.Files, huggingFaceCatalogFile{Filename: file.Filename, Size: file.Size})
		}
		if err != nil {
			projector.Error = err.Error()
		}
		result.Projectors = append(result.Projectors, projector)
	}
	none := "none"
	for key, files := range groups {
		slices.SortFunc(files, func(a, b huggingFaceFile) int { return strings.Compare(a.Filename, b.Filename) })
		variant := huggingFaceVariant{Filename: files[0].Filename, Quantization: "GGUF", Shards: len(files), Files: []huggingFaceCatalogFile{}}
		if matches := huggingFaceQuantization.FindStringSubmatch(path.Base(key)); len(matches) > 1 {
			variant.Quantization = strings.ToUpper(matches[1])
		}
		for _, file := range files {
			variant.Files = append(variant.Files, huggingFaceCatalogFile{Filename: file.Filename, Size: file.Size})
			variant.Size += file.Size
		}
		_, err := selectHuggingFaceGGUF(files, "", huggingFaceSelection{File: variant.Filename, Projector: &none})
		variant.Complete = err == nil
		if err != nil {
			variant.Error = err.Error()
		}
		result.Variants = append(result.Variants, variant)
	}
	slices.SortFunc(result.Variants, func(a, b huggingFaceVariant) int { return strings.Compare(a.Filename, b.Filename) })
	slices.SortFunc(result.Projectors, func(a, b huggingFaceCatalogFile) int { return strings.Compare(a.Filename, b.Filename) })
	return result
}

func huggingFaceCatalogError(c *gin.Context, err error) {
	status := http.StatusBadGateway
	if errors.Is(err, os.ErrNotExist) {
		status = http.StatusNotFound
	}
	if errors.Is(err, errUnauthorized) || strings.Contains(err.Error(), "403:") {
		status = http.StatusForbidden
	}
	if errors.Is(err, context.DeadlineExceeded) {
		status = http.StatusGatewayTimeout
	}
	if errors.Is(err, errHuggingFaceCatalogBusy) {
		status = http.StatusTooManyRequests
		c.Header("Retry-After", "1")
	}
	c.JSON(status, gin.H{"error": err.Error()})
}

func (s *Server) HuggingFaceFilesHandler(c *gin.Context) {
	repository := strings.TrimSpace(c.Query("repository"))
	revision := strings.TrimSpace(c.Query("revision"))
	if len(repository) > 256 || !huggingFaceRepositoryID.MatchString(repository) || strings.Contains(repository, "..") {
		c.JSON(http.StatusBadRequest, gin.H{"error": "use a Hugging Face repository in owner/model format"})
		return
	}
	if revision != "" && !huggingFaceRevision.MatchString(revision) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Hugging Face revision must be a commit SHA"})
		return
	}
	ctx, cancel := context.WithTimeout(c.Request.Context(), 20*time.Second)
	defer cancel()
	ttl := time.Minute
	if revision != "" {
		ttl = 10 * time.Minute
	}
	data, err := huggingFaceMetadata.get(ctx, "files:"+repository+"@"+revision, ttl, func(ctx context.Context) ([]byte, error) {
		source, err := readHuggingFaceRepository(ctx, model.ParseName("huggingface.co/"+repository), &registryOptions{HuggingFace: huggingFaceSelection{Revision: revision}})
		if err != nil {
			return nil, err
		}
		return json.Marshal(huggingFaceVariants(repository, source))
	})
	if err != nil {
		huggingFaceCatalogError(c, err)
		return
	}
	c.Data(http.StatusOK, "application/json; charset=utf-8", data)
}

func (s *Server) HuggingFaceSearchHandler(c *gin.Context) {
	query := strings.TrimSpace(c.Query("q"))
	if len(query) < 2 || len(query) > 200 || strings.ContainsAny(query, "\x00\r\n") {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Hugging Face search requires 2 to 200 characters"})
		return
	}
	target, _ := url.Parse("https://huggingface.co/api/models")
	target.RawQuery = url.Values{"search": {query}, "filter": {"gguf"}, "sort": {"downloads"}, "direction": {"-1"}, "limit": {"20"}}.Encode()
	type searchResult struct {
		ID        string          `json:"id"`
		Downloads int64           `json:"downloads"`
		Likes     int64           `json:"likes"`
		Private   bool            `json:"private"`
		Gated     json.RawMessage `json:"gated"`
	}
	ctx, cancel := context.WithTimeout(c.Request.Context(), 15*time.Second)
	defer cancel()
	data, err := huggingFaceMetadata.get(ctx, "search:"+query, 45*time.Second, func(ctx context.Context) ([]byte, error) {
		var source []searchResult
		if err := readHuggingFaceJSON(ctx, target, &registryOptions{}, &source); err != nil {
			return nil, err
		}
		models := []gin.H{}
		for _, item := range source {
			if !huggingFaceRepositoryID.MatchString(item.ID) {
				continue
			}
			gated := string(item.Gated)
			models = append(models, gin.H{"id": item.ID, "downloads": item.Downloads, "likes": item.Likes, "private": item.Private, "gated": gated != "" && gated != "false" && gated != "null"})
		}
		return json.Marshal(gin.H{"models": models, "source": "huggingface"})
	})
	if err != nil {
		huggingFaceCatalogError(c, err)
		return
	}
	c.Data(http.StatusOK, "application/json; charset=utf-8", data)
}
