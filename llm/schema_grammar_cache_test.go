package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func assertSchemaCacheAccounting(t *testing.T, cache *schemaGrammarCache) {
	t.Helper()
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if cache.order.Len() != len(cache.entries) || len(cache.entries) > cache.maxEntries {
		t.Fatalf("cache has %d list entries and %d map entries, limit %d", cache.order.Len(), len(cache.entries), cache.maxEntries)
	}
	bytes := 0
	for e := cache.order.Front(); e != nil; e = e.Next() {
		value := e.Value.(*schemaGrammar)
		if cache.entries[value.schema] != e {
			t.Fatal("cache index does not match its LRU list")
		}
		bytes += len(value.schema) + len(value.grammar)
	}
	if bytes != cache.bytes || bytes > cache.maxBytes {
		t.Fatalf("cache payload bytes = %d, accounted = %d, limit = %d", bytes, cache.bytes, cache.maxBytes)
	}
}

func TestSchemaGrammarCacheByteBudget(t *testing.T) {
	cache := newSchemaGrammarCache(64, 16)
	cache.put("a", "aaaa") // five bytes
	cache.put("b", "bbbb") // five bytes
	cache.put("c", "cccc") // five bytes
	if grammar, ok := cache.get("a"); !ok || grammar != "aaaa" {
		t.Fatal("cache hit lost its value")
	}
	cache.put("d", "dddddd") // seven bytes, requiring two oldest evictions
	for _, schema := range []string{"b", "c"} {
		if _, ok := cache.get(schema); ok {
			t.Fatalf("old entry %q was not evicted for the byte budget", schema)
		}
	}
	for _, schema := range []string{"a", "d"} {
		if _, ok := cache.get(schema); !ok {
			t.Fatalf("recent entry %q was evicted", schema)
		}
	}
	assertSchemaCacheAccounting(t, cache)
	if cache.bytes != 12 {
		t.Fatalf("payload bytes = %d, want 12", cache.bytes)
	}

	// A duplicate completion must not replace or charge an existing immutable
	// conversion twice, even if its new value would exceed the whole budget.
	cache.put("a", strings.Repeat("ignored", 10))
	if grammar, ok := cache.get("a"); !ok || grammar != "aaaa" {
		t.Fatal("duplicate insertion replaced the first conversion")
	}
	if cache.bytes != 12 {
		t.Fatalf("duplicate insertion changed payload bytes to %d", cache.bytes)
	}
	cache.put("oversized", strings.Repeat("large", 10))
	assertSchemaCacheAccounting(t, cache)
	if len(cache.entries) != 2 {
		t.Fatal("oversized conversion displaced useful cached entries")
	}
}

func TestSchemaGrammarCacheEntryBudget(t *testing.T) {
	cache := newSchemaGrammarCache(2, 1024)
	cache.put("a", "aaa")
	cache.put("b", "bbb")
	cache.get("a")
	cache.put("c", "ccc")
	if _, ok := cache.get("b"); ok {
		t.Fatal("entry limit did not evict the least recently used entry")
	}
	assertSchemaCacheAccounting(t, cache)
}

func TestSchemaGrammarCacheUncachedConversion(t *testing.T) {
	schema := `{"type":"string"}`
	grammar := `root ::= "complete-original-grammar"`
	var conversions atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conversions.Add(1)
		var body struct {
			Schema json.RawMessage `json:"json_schema"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || string(body.Schema) != schema {
			t.Errorf("native conversion request did not retain schema: %s, error %v", body.Schema, err)
		}
		json.NewEncoder(w).Encode(map[string]string{"generation_settings/grammar": grammar})
	}))
	defer server.Close()
	runner := &llamaServerRunner{port: server.Listener.Addr().(*net.TCPAddr).Port}
	cache := newSchemaGrammarCache(64, len(schema)+len(grammar)-1)
	for range 2 {
		got, err := cache.getOrConvert(schema, func() (string, error) {
			return runner.convertSchemaGrammar(t.Context(), json.RawMessage(schema))
		})
		if err != nil || got != grammar {
			t.Fatalf("oversized conversion changed: %q, %v", got, err)
		}
	}
	if conversions.Load() != 2 || len(cache.entries) != 0 || cache.bytes != 0 {
		t.Fatalf("oversized conversion retained: calls %d, entries %d, bytes %d", conversions.Load(), len(cache.entries), cache.bytes)
	}
	assertSchemaCacheAccounting(t, cache)
}

func TestSchemaGrammarCacheConversionErrors(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
	}{
		{"native_error", http.StatusBadRequest, `{"error":"invalid schema"}`},
		{"invalid_json", http.StatusOK, "{"},
		{"missing_grammar", http.StatusOK, "{}"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if calls.Add(1) == 1 {
					w.WriteHeader(tc.status)
					fmt.Fprint(w, tc.body)
					return
				}
				json.NewEncoder(w).Encode(map[string]string{"generation_settings/grammar": "root ::= \"ok\""})
			}))
			defer server.Close()
			runner := &llamaServerRunner{port: server.Listener.Addr().(*net.TCPAddr).Port}
			cache := newSchemaGrammarCache(64, 1024)
			key := `{"type":"object"}`
			convert := func() (string, error) { return runner.convertSchemaGrammar(t.Context(), json.RawMessage(key)) }
			if _, err := cache.getOrConvert(key, convert); err == nil {
				t.Fatal("failed native conversion did not return an error")
			}
			if len(cache.entries) != 0 || cache.bytes != 0 {
				t.Fatal("failed conversion was cached")
			}
			for range 2 {
				if got, err := cache.getOrConvert(key, convert); err != nil || got != `root ::= "ok"` {
					t.Fatalf("recovered conversion = %q, %v", got, err)
				}
			}
			if calls.Load() != 2 {
				t.Fatalf("native conversion called %d times, want failed call then one successful cached call", calls.Load())
			}
			assertSchemaCacheAccounting(t, cache)
		})
	}
	cache := newSchemaGrammarCache(2, 16)
	failure := errors.New("request canceled")
	if _, err := cache.getOrConvert("schema", func() (string, error) { return "", failure }); !errors.Is(err, failure) {
		t.Fatalf("conversion error changed: %v", err)
	}
}

func TestSchemaGrammarCacheConcurrentInsertion(t *testing.T) {
	cache := newSchemaGrammarCache(8, 256)
	const workers = 32
	arrived := make(chan struct{}, workers)
	release := make(chan struct{})
	var wait sync.WaitGroup
	for range workers {
		wait.Add(1)
		go func() {
			defer wait.Done()
			value, err := cache.getOrConvert("same schema", func() (string, error) {
				arrived <- struct{}{}
				<-release
				return "same grammar", nil
			})
			if err != nil || value != "same grammar" {
				t.Errorf("concurrent conversion = %q, %v", value, err)
			}
		}()
	}
	for range workers {
		select {
		case <-arrived:
		case <-t.Context().Done():
			close(release)
			wait.Wait()
			t.Fatal("native conversion held the cache lock")
		}
	}
	close(release)
	wait.Wait()
	if len(cache.entries) != 1 || cache.bytes != len("same schema")+len("same grammar") {
		t.Fatal("concurrent conversions charged the same schema repeatedly")
	}
	// Exercise eviction and lookup concurrently as well as duplicate publication.
	for worker := range workers {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for step := range 32 {
				key := fmt.Sprintf("schema-%d", (worker+step)%16)
				cache.put(key, "grammar-"+key)
				if value, ok := cache.get(key); ok && value != "grammar-"+key {
					t.Errorf("cache value changed for %q: %q", key, value)
				}
			}
		}()
	}
	wait.Wait()
	assertSchemaCacheAccounting(t, cache)
}
