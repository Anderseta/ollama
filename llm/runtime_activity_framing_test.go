package llm

import (
	"bytes"
	"strings"
	"testing"
)

func TestRuntimeActivityDiscardsEntireOversizedLine(t *testing.T) {
	for _, prefix := range []string{strings.Repeat("x", 32769), strings.Repeat("x", 32700)} {
		w := NewStatusWriter(nil)
		w.observeActivity([]byte(prefix))
		w.observeActivity([]byte(strings.Repeat("y", 100)))
		// This is still the suffix of the oversized line, not a new record.
		w.observeActivity([]byte("wildllama_activity {\"kind\":\"memory\",\"stage\":\"cuda\",\"memory\":{\"vram_bytes\":1}}\n"))
		if _, observed := w.managedVRAM(); observed {
			t.Fatal("suffix of an oversized line changed memory admission observations")
		}
		w.observeActivity([]byte("wildllama_activity {\"kind\":\"memory\",\"stage\":\"cuda\",\"memory\":{\"vram_bytes\":1234}}\n"))
		if value, observed := w.managedVRAM(); !observed || value != 1234 {
			t.Fatal("valid record after oversized line was not recovered")
		}
		if len(w.activityTail) > 32768 {
			t.Fatal("line buffer exceeded its bound")
		}
	}
}

func BenchmarkRuntimeActivityLogFraming(b *testing.B) {
	for _, test := range []struct{ name, line string }{
		{"ordinary", "llama_model_loader: - kv 12: model.attention.head_count u32 = 32\n"},
		{"activity", "wildllama_activity {\"kind\":\"memory\",\"stage\":\"cuda\",\"memory\":{\"vram_bytes\":1234,\"vram_limit_bytes\":9000}}\n"},
	} {
		b.Run(test.name, func(b *testing.B) {
			chunk := bytes.Repeat([]byte(test.line), 64)
			w := NewStatusWriter(nil)
			b.SetBytes(int64(len(chunk)))
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				w.observeActivity(chunk)
			}
		})
	}
}

func TestRuntimeActivitySnapshotValuesAreIndependent(t *testing.T) {
	w := NewStatusWriter(nil)
	w.observeActivity([]byte("wildllama_activity {\"kind\":\"phase\",\"phase\":\"generating\",\"progress\":0.5,\"prompt_processed\":8,\"prompt_total\":10,\"generated_tokens\":3,\"tokens_per_second\":2.5}\n"))
	w.observeActivity([]byte("wildllama_activity {\"kind\":\"operation\",\"name\":\"compute\",\"layer\":4}\n"))
	first := w.ActivitySnapshot()
	*first.Progress = 1
	*first.PromptProcessed = 99
	*first.PromptTotal = 99
	*first.GeneratedTokens = 99
	*first.TokensPerSecond = 99
	*first.Operation.Layer = 99
	next := w.ActivitySnapshot()
	if *next.Progress != 0.5 || *next.PromptProcessed != 8 || *next.PromptTotal != 10 || *next.GeneratedTokens != 3 || *next.TokensPerSecond != 2.5 || *next.Operation.Layer != 4 {
		t.Fatal("poller mutated the runner's activity through a snapshot")
	}
}
