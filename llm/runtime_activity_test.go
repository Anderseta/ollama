package llm

import (
	"encoding/json"
	"io"
	"strings"
	"sync"
	"testing"
)

func TestRuntimeActivityFramingAndSnapshots(t *testing.T) {
	w := NewStatusWriter(io.Discard)
	parts := []string{
		"wildllama_activity {\"kind\":\"phase\",\"phase\":\"load",
		"ing\",\"progress\":0.25,\"stage\":\"text_model\"}\n",
		"wildllama_activity {\"kind\":\"memory\",\"stage\":\"ram\",\"memory\":{\"ram_bytes\":4096,\"ram_limit_bytes\":8192,\"ssd_bytes\":1024}}\n" +
			"wildllama_activity {\"kind\":\"memory\",\"stage\":\"cuda\",\"memory\":{\"vram_bytes\":1234,\"vram_limit_bytes\":9000}}\n",
	}
	for _, part := range parts {
		_, _ = w.Write([]byte(part))
	}
	first := w.ActivitySnapshot()
	if first.Phase != "loading" || first.Progress == nil || *first.Progress != .25 || first.Memory.RAMBytes != 4096 || first.Memory.VRAMBytes != 1234 {
		t.Fatalf("bad snapshot: %+v", first)
	}
	_, _ = w.Write([]byte("wildllama_activity {\"kind\":\"phase\",\"phase\":\"prefill\",\"progress\":0,\"prompt_processed\":0,\"prompt_total\":100}\n"))
	_, _ = w.Write([]byte("wildllama_activity {\"kind\":\"operation\",\"name\":\"reading_weights\",\"layer\":3}\n"))
	second := w.ActivitySnapshot()
	if second.Phase != "prefill" || second.Stage != "" || second.Operation.Name != "reading_weights" || len(second.History) != 2 {
		t.Fatalf("phase state: %+v", second)
	}
	first.Memory.RAMBytes = 999
	first.History[0].Phase = "failed"
	if second.Memory.RAMBytes != 4096 || second.History[0].Phase != "loading" {
		t.Fatal("snapshots share mutable state")
	}
	_, _ = w.Write([]byte("wildllama_activity {\"kind\":\"phase\",\"phase\":\"generating\",\"generated_tokens\":5,\"tokens_per_second\":2.5}\n"))
	_, _ = w.Write([]byte("wildllama_activity {\"kind\":\"phase\",\"phase\":\"ready\"}\n"))
	_, _ = w.Write([]byte("wildllama_activity {\"kind\":\"operation\",\"name\":\"compute\"}\n"))
	if state := w.ActivitySnapshot(); state.Operation != nil || state.Phase != "ready" {
		t.Fatal("completed work remained active")
	}
}

func TestRuntimeActivityIgnoresMalformedAndPrivateLogs(t *testing.T) {
	w := NewStatusWriter(nil)
	for _, line := range []string{
		"private prompt: wildllama_activity {\"kind\":\"phase\",\"phase\":\"loading\"}\n",
		"wildllama_activity {bad json}\n",
		"wildllama_activity {\"kind\":\"phase\",\"phase\":\"made_up\"}\n",
		"wildllama_activity {\"kind\":\"operation\",\"name\":\"unknown\"}\n",
		strings.Repeat("x", 100000),
		"\n",
	} {
		_, _ = w.Write([]byte(line))
	}
	if state := w.ActivitySnapshot(); state.Detailed || state.Phase != "" || len(w.activityTail) > 32768 {
		t.Fatal("malformed record affected activity")
	}
	_, _ = w.Write([]byte("wildllama_activity {\"kind\":\"phase\",\"phase\":\"loading\",\"progress\":4,\"stage\":\"private model path\",\"prompt\":\"private prompt\"}\n"))
	state := w.ActivitySnapshot()
	encoded, _ := json.Marshal(state)
	if state.Progress != nil || state.Stage != "" || strings.Contains(string(encoded), "private") {
		t.Fatalf("unexpected data exposure: %s", encoded)
	}
}

func TestRuntimeActivityConcurrentPollingAndBoundedHistory(t *testing.T) {
	w := NewStatusWriter(nil)
	var group sync.WaitGroup
	group.Add(2)
	go func() {
		defer group.Done()
		for range 1000 {
			_, _ = w.Write([]byte("wildllama_activity {\"kind\":\"phase\",\"phase\":\"prefill\"}\nwildllama_activity {\"kind\":\"phase\",\"phase\":\"ready\"}\n"))
		}
	}()
	go func() {
		defer group.Done()
		for range 1000 {
			_ = w.ActivitySnapshot()
		}
	}()
	group.Wait()
	if len(w.ActivitySnapshot().History) != 24 {
		t.Fatal("history is not bounded")
	}
}
