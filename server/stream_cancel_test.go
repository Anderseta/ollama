package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"golang.org/x/sync/semaphore"
)

type cancellationRoute struct {
	name    string
	path    string
	request func(bool) any
	native  bool
}

var cancellationRoutes = []cancellationRoute{
	{"native-chat", "/api/chat", func(stream bool) any {
		return api.ChatRequest{Model: "cancellation-model", Messages: []api.Message{{Role: "user", Content: "hello"}}, Stream: &stream}
	}, true},
	{"rendered-chat", "/api/chat", func(stream bool) any {
		return api.ChatRequest{Model: "cancellation-model", Messages: []api.Message{{Role: "user", Content: "hello"}}, Stream: &stream}
	}, false},
	{"generate", "/api/generate", func(stream bool) any {
		return api.GenerateRequest{Model: "cancellation-model", Prompt: "hello", Raw: true, Stream: &stream}
	}, false},
}

// A cancelled inference can still deliver a queued token before returning its
// error. That callback must finish before its deferred runner lease can release.
func cancellationServer(t *testing.T, route cancellationRoute) (*Server, <-chan struct{}, func(), <-chan struct{}) {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	t.Setenv("OLLAMA_GO_TEMPLATE", "1")
	gin.SetMode(gin.TestMode)

	slot := semaphore.NewWeighted(1)
	started := make(chan struct{})
	late := make(chan struct{})
	finished := make(chan struct{})
	var unblock sync.Once
	releaseLate := func() { unblock.Do(func() { close(late) }) }
	t.Cleanup(releaseLate)
	var calls atomic.Int32
	run := func(ctx context.Context, emit func(string, bool)) error {
		if err := slot.Acquire(ctx, 1); err != nil {
			return err
		}
		first := calls.Add(1) == 1
		defer func() {
			slot.Release(1)
			if first {
				close(finished)
			}
		}()
		if !first {
			emit("second request", true)
			return nil
		}
		close(started)
		emit("first token", false)
		<-ctx.Done()
		<-late
		emit("queued after cancellation", false)
		return ctx.Err()
	}
	fixture := &mockRunner{}
	fixture.ChatFn = func(ctx context.Context, _ llm.ChatRequest, callback func(llm.ChatResponse)) error {
		return run(ctx, func(content string, done bool) {
			callback(llm.ChatResponse{Message: api.Message{Role: "assistant", Content: content}, Done: done, DoneReason: llm.DoneReasonStop})
		})
	}
	fixture.CompletionFn = func(ctx context.Context, _ llm.CompletionRequest, callback func(llm.CompletionResponse)) error {
		return run(ctx, func(content string, done bool) {
			callback(llm.CompletionResponse{Content: content, Done: done, DoneReason: llm.DoneReasonStop})
		})
	}
	s := newServerWithMockRunner(t, fixture)
	tmpl := "{{ range .Messages }}{{ .Content }}{{ end }}"
	kv := gguftest.KV{}
	if route.native {
		tmpl = ""
		kv["tokenizer.chat_template"] = "{{ messages[0]['content'] }}"
	}
	createMinimalGGUFModel(t, s, "cancellation-model", kv, tmpl, nil)
	return s, started, releaseLate, finished
}

func waitStreamEvent(t *testing.T, event <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-event:
	case <-time.After(3 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func routeHandler(s *Server, route cancellationRoute) gin.HandlerFunc {
	if route.path == "/api/generate" {
		return s.GenerateHandler
	}
	return s.ChatHandler
}

func TestInferenceHTTPDisconnectReleasesRunner(t *testing.T) {
	for _, route := range cancellationRoutes {
		t.Run(route.name, func(t *testing.T) {
			s, started, releaseLate, finished := cancellationServer(t, route)
			returned := make(chan struct{}, 2)
			router := gin.New()
			router.POST(route.path, func(c *gin.Context) {
				defer func() { returned <- struct{}{} }()
				routeHandler(s, route)(c)
			})
			httpServer := httptest.NewServer(router)
			t.Cleanup(func() {
				releaseLate()
				httpServer.Close()
			})
			client := &http.Client{Timeout: 3 * time.Second}
			request := func(stream bool) *http.Response {
				t.Helper()
				body, err := json.Marshal(route.request(stream))
				if err != nil {
					t.Fatal(err)
				}
				response, err := client.Post(httpServer.URL+route.path, "application/json", bytes.NewReader(body))
				if err != nil {
					t.Fatal(err)
				}
				return response
			}
			response := request(true)
			t.Cleanup(func() { response.Body.Close() })
			line, err := bufio.NewReader(response.Body).ReadString('\n')
			if err != nil || !strings.Contains(line, "first token") {
				t.Fatalf("first response %q, %v", line, err)
			}
			waitStreamEvent(t, started, "first inference")
			response.Body.Close()
			waitStreamEvent(t, returned, "HTTP handler after client disconnect before next token")
			releaseLate()
			waitStreamEvent(t, finished, "queued callback, error and runner lease release")

			second := request(false)
			defer second.Body.Close()
			body, err := io.ReadAll(second.Body)
			if err != nil || second.StatusCode != http.StatusOK || !bytes.Contains(body, []byte("second request")) {
				t.Fatalf("next inference could not use released runner: status=%d body=%s err=%v", second.StatusCode, body, err)
			}
		})
	}
}

type brokenStreamWriter struct {
	header http.Header
}

func (w *brokenStreamWriter) Header() http.Header     { return w.header }
func (*brokenStreamWriter) WriteHeader(int)           {}
func (*brokenStreamWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
func (*brokenStreamWriter) Flush()                    {}
func (*brokenStreamWriter) CloseNotify() <-chan bool  { return make(chan bool) }

func TestInferenceWriterFailureCancelsRunner(t *testing.T) {
	for _, route := range cancellationRoutes {
		t.Run(route.name, func(t *testing.T) {
			s, started, releaseLate, finished := cancellationServer(t, route)
			body, err := json.Marshal(route.request(true))
			if err != nil {
				t.Fatal(err)
			}
			// Keep the parent request live: the handler itself must stop inference
			// on a write failure, rather than relying on net/http cancellation.
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			request := httptest.NewRequest(http.MethodPost, route.path, bytes.NewReader(body)).WithContext(ctx)
			request.Header.Set("Content-Type", "application/json")
			c, _ := gin.CreateTestContext(&brokenStreamWriter{header: make(http.Header)})
			c.Request = request
			returned := make(chan struct{})
			go func() {
				defer close(returned)
				routeHandler(s, route)(c)
			}()
			waitStreamEvent(t, started, "writer-failure inference")
			waitStreamEvent(t, returned, "handler write failure")
			releaseLate()
			waitStreamEvent(t, finished, "writer-failure runner release")
			if ctx.Err() != nil {
				t.Fatal("test parent context should still be live")
			}
			w := createRequest(t, routeHandler(s, route), route.request(false))
			if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "second request") {
				t.Fatalf("next request blocked or failed: %d %s", w.Code, w.Body.String())
			}
		})
	}
}

func TestStreamResponseCancellationWhileWaiting(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	c, _ := gin.CreateTestContext(NewRecorder())
	c.Request = httptest.NewRequest(http.MethodGet, "/", nil).WithContext(ctx)
	finished := make(chan struct{})
	ch := make(chan any)
	defer close(ch)
	go func() {
		defer close(finished)
		streamResponse(c, ch)
	}()
	cancel()
	waitStreamEvent(t, finished, "stream cancellation while no producer token exists")
}

func TestInferenceParserErrorSurvivesChildCancellation(t *testing.T) {
	for _, generate := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			name := "chat"
			if generate {
				name = "generate"
			}
			if stream {
				name += "-stream"
			}
			t.Run(name, func(t *testing.T) {
				t.Setenv("OLLAMA_MODELS", t.TempDir())
				t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
				t.Setenv("OLLAMA_GO_TEMPLATE", "1")
				cancelled := make(chan error, 1)
				fixture := &mockRunner{CompletionFn: func(ctx context.Context, _ llm.CompletionRequest, callback func(llm.CompletionResponse)) error {
					callback(llm.CompletionResponse{Content: "<tool_call>gr", Done: true, DoneReason: llm.DoneReasonStop})
					cancelled <- ctx.Err()
					return ctx.Err()
				}}
				s := newServerWithMockRunner(t, fixture)
				createMinimalGGUFModel(t, s, "parser-base", nil, "{{ range .Messages }}{{ .Content }}{{ end }}", nil)
				w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: "parser-error", From: "parser-base", Parser: "glm-4.7", Stream: new(bool)})
				if w.Code != http.StatusOK {
					t.Fatalf("create parser model: %d %s", w.Code, w.Body.String())
				}
				think := &api.ThinkValue{Value: false}
				if generate {
					w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "parser-error", Prompt: "hello", Think: think, Stream: &stream})
				} else {
					w = createRequest(t, s.ChatHandler, api.ChatRequest{Model: "parser-error", Messages: []api.Message{{Role: "user", Content: "hello"}}, Think: think, Stream: &stream})
				}
				if w.Code != http.StatusInternalServerError || !strings.Contains(w.Body.String(), "incomplete GLM tool call") {
					t.Fatalf("parser error was lost after stopping inference: %d %s", w.Code, w.Body.String())
				}
				select {
				case err := <-cancelled:
					if err != context.Canceled {
						t.Fatalf("parser did not cancel inference: %v", err)
					}
				case <-time.After(3 * time.Second):
					t.Fatal("runner did not return after parser failure")
				}
			})
		}
	}
}
