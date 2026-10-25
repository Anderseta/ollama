package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestKVCacheInitialTokensServeFlag(t *testing.T) {
	for _, value := range []string{"0", "4096", "255"} {
		t.Run(value, func(t *testing.T) {
			for _, key := range []string{"OLLAMA_GPU_EXCLUSIVE", "LLAMA_ARG_GPU_EXCLUSIVE", "OLLAMA_GPU_DIAGNOSTICS", "LLAMA_ARG_GPU_DIAGNOSTICS", "OLLAMA_GPU_VRAM_LIMIT_MIB", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB", "OLLAMA_MOE_RUNTIME"} {
				t.Setenv(key, "")
			}
			t.Setenv("OLLAMA_KV_CACHE_INITIAL_TOKENS", "8192")
			command, _, err := NewCLI().Find([]string{"serve"})
			if err != nil {
				t.Fatal(err)
			}
			flag := command.Flags().Lookup("kv-cache-initial-tokens")
			if flag == nil || flag.DefValue != "0" {
				t.Fatal("serve KV commitment flag missing or not default-off")
			}
			if err := command.Flags().Parse([]string{"--kv-cache-initial-tokens=" + value}); err != nil {
				t.Fatal(err)
			}
			err = configureGPUExecution(command)
			if value == "255" {
				if err == nil {
					t.Fatal("invalid KV commitment reached server startup")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			got, err := api.ResolveKVCacheInitialTokens(api.DefaultOptions())
			if err != nil || (value == "0" && got != 0) || (value == "4096" && got != 4096) {
				t.Fatalf("serve flag lost to environment: %d %v", got, err)
			}
			var help bytes.Buffer
			command.SetOut(&help)
			if err := command.Help(); err != nil || !strings.Contains(help.String(), "--kv-cache-initial-tokens") {
				t.Fatal("serve help omitted experimental KV control")
			}
		})
	}
}
