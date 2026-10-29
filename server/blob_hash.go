package server

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

var modelHashBuffers = sync.Pool{New: func() any { buffer := make([]byte, 1024*1024); return &buffer }}

func blobVerificationMarker(digest string) (string, error) {
	path, err := ggufMetadataPath(digest)
	return path + ".verify-pending", err
}

func markBlobVerificationPending(digest string) error {
	path, err := blobVerificationMarker(digest)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	return file.Close()
}

func blobVerificationPending(digest string) (bool, error) {
	path, err := blobVerificationMarker(digest)
	if err != nil {
		return false, err
	}
	_, err = os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	return err == nil, err
}

func clearBlobVerificationPending(digest string) error {
	path, err := blobVerificationMarker(digest)
	if err != nil {
		return err
	}
	err = os.Remove(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

type modelBlobReader struct {
	context context.Context
	source  io.Reader
}

func (r modelBlobReader) Read(buffer []byte) (int, error) {
	if err := r.context.Err(); err != nil {
		return 0, err
	}
	n, err := r.source.Read(buffer)
	if cancelled := r.context.Err(); cancelled != nil {
		return n, cancelled
	}
	return n, err
}

func hashModelBlob(ctx context.Context, source io.Reader) (string, int64, error) {
	if err := ctx.Err(); err != nil {
		return "", 0, err
	}
	h := sha256.New()
	buffer := modelHashBuffers.Get().(*[]byte)
	defer modelHashBuffers.Put(buffer)
	n, err := io.CopyBuffer(h, modelBlobReader{context: ctx, source: source}, *buffer)
	if err != nil {
		return "", n, err
	}
	return fmt.Sprintf("sha256:%x", h.Sum(nil)), n, nil
}
