package llm

import (
	"os"
	"os/exec"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestRunnerEnvironmentClearDoesNotPassEmptyNativeArgument(t *testing.T) {
	t.Setenv("LLAMA_ARG_CACHE_RAM", "4096")
	cmd := &exec.Cmd{}
	SetupLlamaServerCommandEnv(cmd, "llama-server", nil, map[string]string{
		"LLAMA_ARG_CACHE_RAM": "", "LLAMA_ARG_MOE_RAM_LIMIT_MIB": "", "CUDA_VISIBLE_DEVICES": "",
	})
	foundCUDA := false
	for _, entry := range cmd.Env {
		key, value, _ := strings.Cut(entry, "=")
		if key == "LLAMA_ARG_CACHE_RAM" || key == "LLAMA_ARG_MOE_RAM_LIMIT_MIB" {
			t.Fatalf("empty override reached native argument parser: %q", entry)
		}
		if key == "CUDA_VISIBLE_DEVICES" && value == "" {
			foundCUDA = true
		}
	}
	if !foundCUDA || os.Getenv("LLAMA_ARG_CACHE_RAM") != "4096" {
		t.Fatal("GPU visibility or parent environment changed")
	}
}

func TestLogicalAndPhysicalBatchAreIndependent(t *testing.T) {
	opts := api.DefaultOptions()
	opts.NumBatch = 2048
	opts.NumUbatch = 512
	got := appendBatchArgs(nil, opts, false, 1)
	if !reflect.DeepEqual(got, []string{"-b", "2048", "-ub", "512"}) {
		t.Fatalf("batch args %v", got)
	}
	_, allowance := LlamaServerBatch(loadTestGGUF(t, map[string]any{"general.architecture": "llama"}), opts, true)
	if allowance != 0 {
		t.Fatalf("logical batch incorrectly inflated physical scratch memory by %d", allowance)
	}
}

func TestExplicitSpeculationTypeOverridesDetectedDraft(t *testing.T) {
	opts := api.DefaultOptions()
	opts.SpeculativeType = "none"
	got := appendDraftArgs(nil, draftTypeMTP, "draft.gguf", opts)
	if !reflect.DeepEqual(got, []string{"--spec-type", "none"}) {
		t.Fatalf("draft not disabled: %v", got)
	}
	opts.SpeculativeType = "ngram-simple"
	got = appendDraftArgs(nil, draftTypeMTP, "draft.gguf", opts)
	if !reflect.DeepEqual(got, []string{"--spec-type", "ngram-simple", "--spec-draft-n-max", "4"}) {
		t.Fatalf("ngram should not load an extra model: %v", got)
	}
}
