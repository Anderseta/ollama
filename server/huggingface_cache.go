package server

import (
	"context"
	"errors"
	"sync"
	"time"
)

const huggingFaceCacheBytes = 16 * 1024 * 1024

var errHuggingFaceCatalogBusy = errors.New("Hugging Face metadata queue is full; try again shortly")
var huggingFaceMetadata = newHuggingFaceMetadataCache()

type huggingFaceCachedResponse struct {
	data    []byte
	expires time.Time
	used    time.Time
}

type huggingFaceMetadataRequest struct {
	done    chan struct{}
	cancel  context.CancelFunc
	waiters int
	data    []byte
	err     error
}

type huggingFaceMetadataCache struct {
	mu       sync.Mutex
	entries  map[string]huggingFaceCachedResponse
	requests map[string]*huggingFaceMetadataRequest
	bytes    int
}

func newHuggingFaceMetadataCache() *huggingFaceMetadataCache {
	return &huggingFaceMetadataCache{entries: map[string]huggingFaceCachedResponse{}, requests: map[string]*huggingFaceMetadataRequest{}}
}

func (cache *huggingFaceMetadataCache) get(ctx context.Context, key string, ttl time.Duration, load func(context.Context) ([]byte, error)) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	cache.mu.Lock()
	now := time.Now()
	for name, entry := range cache.entries {
		if !now.Before(entry.expires) {
			delete(cache.entries, name)
			cache.bytes -= len(entry.data)
		}
	}
	if entry, ok := cache.entries[key]; ok {
		entry.used = now
		cache.entries[key] = entry
		cache.mu.Unlock()
		return entry.data, nil
	}
	request := cache.requests[key]
	if request != nil && request.waiters == 0 {
		cache.mu.Unlock()
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-request.done:
			return cache.get(ctx, key, ttl, load)
		}
	}
	if request == nil {
		if len(cache.requests) >= 8 {
			cache.mu.Unlock()
			return nil, errHuggingFaceCatalogBusy
		}
		shared, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		request = &huggingFaceMetadataRequest{done: make(chan struct{}), cancel: cancel}
		cache.requests[key] = request
		go cache.load(shared, key, ttl, request, load)
	}
	request.waiters++
	cache.mu.Unlock()
	defer func() {
		cache.mu.Lock()
		defer cache.mu.Unlock()
		request.waiters--
		if request.waiters == 0 {
			request.cancel()
		}
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-request.done:
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return request.data, request.err
	}
}

func (cache *huggingFaceMetadataCache) load(ctx context.Context, key string, ttl time.Duration, request *huggingFaceMetadataRequest, load func(context.Context) ([]byte, error)) {
	data, err := load(ctx)
	valid := err == nil && ctx.Err() == nil
	request.cancel()
	cache.mu.Lock()
	defer cache.mu.Unlock()
	request.data, request.err = data, err
	if valid && len(data) <= 2*1024*1024 {
		for len(cache.entries) >= 32 || cache.bytes+len(data) > huggingFaceCacheBytes {
			var oldest string
			var at time.Time
			for name, entry := range cache.entries {
				if oldest == "" || entry.used.Before(at) {
					oldest, at = name, entry.used
				}
			}
			if oldest == "" {
				break
			}
			cache.bytes -= len(cache.entries[oldest].data)
			delete(cache.entries, oldest)
		}
		now := time.Now()
		cache.entries[key] = huggingFaceCachedResponse{data: data, expires: now.Add(ttl), used: now}
		cache.bytes += len(data)
	}
	delete(cache.requests, key)
	close(request.done)
}
