package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestMOERuntimeServeFlagAndPrecedence(t *testing.T) {
	for _, value := range []string{"true", "false"} {
		t.Run(value, func(t *testing.T) {
			for _, key := range []string{"OLLAMA_GPU_EXCLUSIVE", "OLLAMA_GPU_DIAGNOSTICS", "OLLAMA_GPU_VRAM_LIMIT_MIB", "LLAMA_ARG_GPU_EXCLUSIVE", "LLAMA_ARG_GPU_DIAGNOSTICS", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB", "OLLAMA_MOE_RUNTIME"} {
				t.Setenv(key, "")
			}
			t.Setenv("OLLAMA_MOE_RUNTIME", "true")
			command, _, err := NewCLI().Find([]string{"serve"})
			if err != nil {
				t.Fatal(err)
			}
			flag := command.Flags().Lookup("moe-runtime")
			if flag == nil || flag.DefValue != "false" {
				t.Fatal("serve runtime flag is missing or defaults to enabled")
			}
			if err := command.Flags().Parse([]string{"--moe-runtime=" + value}); err != nil {
				t.Fatal(err)
			}
			if err := configureGPUExecution(command); err != nil {
				t.Fatal(err)
			}
			cfg, err := api.ResolveGPUExecution(api.DefaultOptions())
			if err != nil || cfg.Runtime != (value == "true") {
				t.Fatalf("serve runtime flag lost: %+v %v", cfg, err)
			}
			var help bytes.Buffer
			command.SetOut(&help)
			if err := command.Help(); err != nil || !strings.Contains(help.String(), "--moe-runtime") {
				t.Fatalf("serve help omitted runtime option: %v", err)
			}
		})
	}
}
