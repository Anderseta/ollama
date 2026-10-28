package llm

import (
	"bytes"
	"encoding/json"
	"math"
	"time"
)

// RuntimeActivity is an observed runner snapshot, shared by its active slots.
// It intentionally contains no prompts, responses, file paths or raw log lines.
type RuntimeActivity struct {
	SpeculativeDraftTokens    *int  `json:"speculative_draft_tokens,omitempty"`
	SpeculativeAcceptedTokens *int  `json:"speculative_accepted_tokens,omitempty"`
	SpeculativeObservedAt     int64 `json:"speculative_observed_at,omitempty"`
	speculativeGeneration     uint64
	Phase                     string            `json:"phase"`
	PhaseStartedAt            int64             `json:"phase_started_at"`
	UpdatedAt                 int64             `json:"updated_at"`
	Detailed                  bool              `json:"detailed"`
	Stage                     string            `json:"stage,omitempty"`
	Progress                  *float64          `json:"progress,omitempty"`
	PromptProcessed           *uint64           `json:"prompt_processed,omitempty"`
	PromptTotal               *uint64           `json:"prompt_total,omitempty"`
	GeneratedTokens           *uint64           `json:"generated_tokens,omitempty"`
	TokensPerSecond           *float64          `json:"tokens_per_second,omitempty"`
	Operation                 *RuntimeOperation `json:"operation,omitempty"`
	Memory                    *RuntimeMemory    `json:"memory,omitempty"`
	History                   []RuntimePhase    `json:"history"`
}

type RuntimePhase struct {
	Phase string `json:"phase"`
	At    int64  `json:"at"`
}
type RuntimeOperation struct {
	Name      string `json:"name"`
	Layer     *int32 `json:"layer,omitempty"`
	UpdatedAt int64  `json:"updated_at"`
}
type RuntimeMemory struct {
	RAMBytes       uint64 `json:"ram_bytes"`
	RAMLimitBytes  uint64 `json:"ram_limit_bytes"`
	VRAMBytes      uint64 `json:"vram_bytes"`
	VRAMLimitBytes uint64 `json:"vram_limit_bytes"`
	SSDBytes       uint64 `json:"ssd_bytes"`
	H2DBytes       uint64 `json:"h2d_bytes"`
	UpdatedAt      int64  `json:"updated_at"`
	cudaObserved   bool
}

type runtimeEvent struct {
	Kind            string         `json:"kind"`
	Phase           string         `json:"phase"`
	Stage           string         `json:"stage"`
	Progress        *float64       `json:"progress"`
	PromptProcessed *uint64        `json:"prompt_processed"`
	PromptTotal     *uint64        `json:"prompt_total"`
	GeneratedTokens *uint64        `json:"generated_tokens"`
	TokensPerSecond *float64       `json:"tokens_per_second"`
	Name            string         `json:"name"`
	Layer           *int32         `json:"layer"`
	Memory          *RuntimeMemory `json:"memory"`
}

func runtimePhaseAllowed(phase string) bool {
	switch phase {
	case "starting", "loading", "context", "warming", "ready", "prefill", "generating", "sleeping", "failed":
		return true
	}
	return false
}

func (a *RuntimeActivity) setPhase(phase string, now int64) {
	if !runtimePhaseAllowed(phase) {
		return
	}
	if a.Phase != phase {
		a.Phase, a.PhaseStartedAt = phase, now
		a.Progress, a.TokensPerSecond = nil, nil
		a.Stage = ""
		if phase == "prefill" || phase == "loading" {
			a.PromptProcessed, a.PromptTotal, a.GeneratedTokens = nil, nil, nil
			a.SpeculativeDraftTokens, a.SpeculativeAcceptedTokens, a.SpeculativeObservedAt = nil, nil, 0
		}
		a.History = append(a.History, RuntimePhase{Phase: phase, At: now})
		if len(a.History) > 24 {
			a.History = a.History[len(a.History)-24:]
		}
	}
	if phase == "ready" || phase == "failed" || phase == "sleeping" {
		a.Operation = nil
	}
	a.UpdatedAt = now
}

func (w *StatusWriter) beginSpeculativeGeneration() uint64 {
	if w == nil {
		return 0
	}
	w.activityMu.Lock()
	defer w.activityMu.Unlock()
	a := &w.activity
	a.speculativeGeneration++
	a.SpeculativeDraftTokens, a.SpeculativeAcceptedTokens, a.SpeculativeObservedAt = nil, nil, 0
	return a.speculativeGeneration
}

func (w *StatusWriter) observeSpeculativeGeneration(generation uint64, draft, accepted *int) {
	if w == nil || draft == nil || accepted == nil {
		return
	}
	w.activityMu.Lock()
	defer w.activityMu.Unlock()
	a := &w.activity
	if a.speculativeGeneration != generation {
		return
	}
	a.SpeculativeDraftTokens, a.SpeculativeAcceptedTokens = activityValueCopy(draft), activityValueCopy(accepted)
	a.SpeculativeObservedAt = time.Now().UnixMilli()
}

func (w *StatusWriter) observeActivity(b []byte) {
	w.activityMu.Lock()
	defer w.activityMu.Unlock()
	// Subprocess writes can split JSON, combine records or emit a long dot line.
	// Bound incomplete data independently of how much text a runner logs.
	const maxLine = 32 * 1024
	for len(b) > 0 {
		i := bytes.IndexByte(b, '\n')
		if i < 0 {
			if !w.activityDiscardingLine && len(w.activityTail)+len(b) <= maxLine {
				w.activityTail = append(w.activityTail, b...)
			} else {
				w.activityTail = w.activityTail[:0]
				w.activityDiscardingLine = true
			}
			break
		}
		if !w.activityDiscardingLine && len(w.activityTail)+i <= maxLine {
			if len(w.activityTail) == 0 {
				w.observeActivityLine(b[:i], time.Now().UnixMilli())
			} else {
				w.activityTail = append(w.activityTail, b[:i]...)
				w.observeActivityLine(w.activityTail, time.Now().UnixMilli())
			}
		}
		w.activityTail = w.activityTail[:0]
		w.activityDiscardingLine = false
		b = b[i+1:]
	}
}

func (w *StatusWriter) observeActivityLine(line []byte, now int64) {
	a := &w.activity
	const marker = "wildllama_activity "
	if bytes.HasPrefix(line, []byte(marker)) {
		var event runtimeEvent
		if json.Unmarshal(line[len(marker):], &event) != nil {
			return
		}
		switch event.Kind {
		case "phase":
			if !runtimePhaseAllowed(event.Phase) {
				return
			}
			a.setPhase(event.Phase, now)
			if event.Stage == "text_model" || event.Stage == "mmproj_model" || event.Stage == "spec_model" {
				a.Stage = event.Stage
			}
			if event.Progress != nil && !math.IsNaN(*event.Progress) && *event.Progress >= 0 && *event.Progress <= 1 {
				a.Progress = event.Progress
			}
			if event.PromptProcessed != nil {
				a.PromptProcessed = event.PromptProcessed
			}
			if event.PromptTotal != nil {
				a.PromptTotal = event.PromptTotal
			}
			if event.GeneratedTokens != nil {
				a.GeneratedTokens = event.GeneratedTokens
			}
			if event.TokensPerSecond != nil && !math.IsNaN(*event.TokensPerSecond) && !math.IsInf(*event.TokensPerSecond, 0) && *event.TokensPerSecond >= 0 {
				a.TokensPerSecond = event.TokensPerSecond
			}
		case "operation":
			if a.Phase == "ready" || a.Phase == "sleeping" || a.Phase == "failed" {
				return
			}
			switch event.Name {
			case "compute", "routing", "reading_weights", "uploading_weights":
				a.Operation = &RuntimeOperation{Name: event.Name, Layer: event.Layer, UpdatedAt: now}
			default:
				return
			}
		case "memory":
			if event.Memory == nil {
				return
			}
			event.Memory.UpdatedAt = now
			// RAM/SSD counters and CUDA accounting are emitted independently.
			if a.Memory == nil {
				a.Memory = &RuntimeMemory{}
			}
			if event.Stage == "ram" {
				a.Memory.RAMBytes, a.Memory.RAMLimitBytes = event.Memory.RAMBytes, event.Memory.RAMLimitBytes
				a.Memory.SSDBytes, a.Memory.H2DBytes = event.Memory.SSDBytes, event.Memory.H2DBytes
			} else if event.Stage == "cuda" {
				a.Memory.VRAMBytes, a.Memory.VRAMLimitBytes = event.Memory.VRAMBytes, event.Memory.VRAMLimitBytes
				a.Memory.cudaObserved = true
			} else {
				return
			}
			a.Memory.UpdatedAt = now
		default:
			return
		}
		a.Detailed, a.UpdatedAt = true, now
		return
	}
	// Older native payloads still report useful milestones. These are concrete
	// log messages, never inferred from a timer or from increasing RAM usage.
	switch {
	case bytes.Contains(line, []byte("load_tensors:")) && bytes.Contains(line, []byte("loading model tensors")):
		a.setPhase("loading", now)
	case bytes.Contains(line, []byte("llama_context:")) && bytes.Contains(line, []byte("constructing llama_context")):
		a.setPhase("context", now)
	case bytes.Contains(line, []byte("warmup")) && bytes.Contains(line, []byte("running")):
		a.setPhase("warming", now)
	}
}

func (w *StatusWriter) setActivityPhase(phase string) {
	if w == nil {
		return
	}
	w.activityMu.Lock()
	defer w.activityMu.Unlock()
	w.activity.setPhase(phase, time.Now().UnixMilli())
}

func (w *StatusWriter) ActivitySnapshot() RuntimeActivity {
	if w == nil {
		return RuntimeActivity{}
	}
	w.activityMu.Lock()
	defer w.activityMu.Unlock()
	value := w.activity
	value.Progress = activityValueCopy(value.Progress)
	value.PromptProcessed = activityValueCopy(value.PromptProcessed)
	value.PromptTotal = activityValueCopy(value.PromptTotal)
	value.GeneratedTokens = activityValueCopy(value.GeneratedTokens)
	value.TokensPerSecond = activityValueCopy(value.TokensPerSecond)
	value.SpeculativeDraftTokens = activityValueCopy(value.SpeculativeDraftTokens)
	value.SpeculativeAcceptedTokens = activityValueCopy(value.SpeculativeAcceptedTokens)
	value.History = append([]RuntimePhase{}, value.History...)
	if value.Memory != nil {
		memory := *value.Memory
		value.Memory = &memory
	}
	if value.Operation != nil {
		operation := *value.Operation
		operation.Layer = activityValueCopy(operation.Layer)
		value.Operation = &operation
	}
	return value
}

func activityValueCopy[T any](value *T) *T {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

// managedVRAM reports successful live allocations in the native CUDA allocator.
// This includes expert caches and graph pools, but excludes the driver and other
// processes. An observed zero quota still provides a valid allocation count.
func (w *StatusWriter) managedVRAM() (uint64, bool) {
	if w == nil {
		return 0, false
	}
	w.activityMu.Lock()
	defer w.activityMu.Unlock()
	if w.activity.Memory == nil || !w.activity.Memory.cudaObserved {
		return 0, false
	}
	return w.activity.Memory.VRAMBytes, true
}

// Optional so non-llama.cpp backends retain their existing interface.
type RuntimeActivityReporter interface{ RuntimeActivity() RuntimeActivity }

func (s *llamaServerRunner) RuntimeActivity() RuntimeActivity { return s.status.ActivitySnapshot() }
