package manifest

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

const maxDownloadPinBytes = 256 * 1024

type downloadPin struct {
	Model   string   `json:"model"`
	Digests []string `json:"digests"`
}

func downloadPinPath(name model.Name) string {
	key := sha256.Sum256([]byte(strings.ToLower(name.String())))
	return filepath.Join(envconfig.Models(), "metadata", fmt.Sprintf("download-pin-%x.json", key))
}

func PinDownload(name model.Name, digests []string) error {
	if !name.IsFullyQualified() || !name.IsValid() || len(digests) > 4096 {
		return errors.New("invalid model download pin")
	}
	seen := map[string]struct{}{}
	pin := downloadPin{Model: name.String(), Digests: []string{}}
	for _, digest := range digests {
		if err := ValidateDigest(digest); err != nil {
			return err
		}
		if _, ok := seen[digest]; !ok {
			pin.Digests = append(pin.Digests, digest)
			seen[digest] = struct{}{}
		}
	}
	data, err := json.Marshal(pin)
	if err != nil {
		return err
	}
	if len(data) > maxDownloadPinBytes {
		return errors.New("model download pin is too large")
	}
	manifestStoreMu.Lock()
	defer manifestStoreMu.Unlock()
	path := downloadPinPath(name)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".download-pin-*")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	if _, err := file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(file.Name(), path)
}

func ReleaseDownloadPin(name model.Name) error {
	manifestStoreMu.Lock()
	defer manifestStoreMu.Unlock()
	err := os.Remove(downloadPinPath(name))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

func DownloadPinnedBlobs() (map[string]struct{}, error) {
	manifestStoreMu.Lock()
	defer manifestStoreMu.Unlock()
	return downloadPinnedBlobsLocked()
}

func downloadPinnedBlobsLocked() (map[string]struct{}, error) {
	result := map[string]struct{}{}
	directory, err := os.Open(filepath.Join(envconfig.Models(), "metadata"))
	if errors.Is(err, os.ErrNotExist) {
		return result, nil
	}
	if err != nil {
		return nil, err
	}
	defer directory.Close()
	count, total := 0, 0
	for {
		entries, err := directory.ReadDir(128)
		if err != nil && !errors.Is(err, io.EOF) {
			return nil, err
		}
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasPrefix(entry.Name(), "download-pin-") || !strings.HasSuffix(entry.Name(), ".json") {
				continue
			}
			count++
			if count > 512 {
				return nil, errors.New("too many pending model downloads to prune safely")
			}
			path := filepath.Join(directory.Name(), entry.Name())
			file, err := os.Open(path)
			if err != nil {
				return nil, err
			}
			data, readErr := io.ReadAll(io.LimitReader(file, maxDownloadPinBytes+1))
			file.Close()
			if readErr != nil {
				return nil, readErr
			}
			total += len(data)
			if len(data) > maxDownloadPinBytes || total > 4*1024*1024 {
				return nil, errors.New("pending model download metadata exceeds safe prune limit")
			}
			var pin downloadPin
			if err := json.Unmarshal(data, &pin); err != nil {
				return nil, fmt.Errorf("invalid pending model download: %w", err)
			}
			name := model.ParseName(pin.Model)
			if !name.IsFullyQualified() || !name.IsValid() || filepath.Base(downloadPinPath(name)) != entry.Name() || len(pin.Digests) > 4096 {
				return nil, errors.New("invalid pending model download reference")
			}
			for _, digest := range pin.Digests {
				if err := ValidateDigest(digest); err != nil {
					return nil, err
				}
				result[digest] = struct{}{}
			}
		}
		if errors.Is(err, io.EOF) {
			return result, nil
		}
	}
}
