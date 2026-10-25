package cmd

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

func TestGPUExecutionCLIOverridesEnvironment(t *testing.T) {
	for _, key := range []string{"OLLAMA_GPU_EXCLUSIVE", "OLLAMA_GPU_DIAGNOSTICS", "OLLAMA_GPU_VRAM_LIMIT_MIB", "LLAMA_ARG_GPU_EXCLUSIVE", "LLAMA_ARG_GPU_DIAGNOSTICS", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB"} {
		t.Setenv(key, "")
	}
	t.Setenv("OLLAMA_GPU_EXCLUSIVE", "true")
	command := &cobra.Command{}
	command.Flags().Bool("gpu-exclusive", false, "")
	command.Flags().Bool("gpu-diagnostics", false, "")
	command.Flags().Int("gpu-vram-limit-mib", 0, "")
	if err := command.Flags().Parse([]string{"--gpu-exclusive=false", "--gpu-diagnostics", "--gpu-vram-limit-mib=12288"}); err != nil {
		t.Fatal(err)
	}
	if err := configureGPUExecution(command); err != nil {
		t.Fatal(err)
	}
	cfg, err := api.ResolveGPUExecution(api.DefaultOptions())
	if err != nil || cfg.Exclusive || !cfg.Diagnostics || cfg.VRAMLimitMiB != 12288 {
		t.Fatalf("wrong effective CLI policy %+v %v", cfg, err)
	}
	if err := command.Flags().Set("gpu-vram-limit-mib", "-1"); err != nil {
		t.Fatal(err)
	}
	if err := configureGPUExecution(command); err == nil {
		t.Fatal("negative budget accepted before startup")
	}
}
