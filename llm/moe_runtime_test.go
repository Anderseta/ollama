package llm

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestMOERuntimeChildEnvironmentOverridesInheritance(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		name, parent, want := "disabled", "1", "0"
		if enabled {
			name, parent, want = "enabled", "0", "1"
		}
		t.Run(name, func(t *testing.T) {
			t.Setenv("WILDLLAMA_MOE_RUNTIME", parent)
			environment := map[string]string{
				"WILDLLAMA_MOE_RUNTIME":         parent,
				"WILDLLAMA_MOE_DYNAMIC_PREFILL": "1",
			}
			setMOERuntimeEnvironment(environment, enabled)
			command := &exec.Cmd{}
			SetupLlamaServerCommandEnv(command, "llama-server", nil, environment)
			found := 0
			for _, entry := range command.Env {
				key, value, _ := strings.Cut(entry, "=")
				if strings.EqualFold(key, "WILDLLAMA_MOE_RUNTIME") {
					found++
					if value != want {
						t.Fatalf("child inherited contradictory runtime: %s", entry)
					}
				}
			}
			if found != 1 || os.Getenv("WILDLLAMA_MOE_RUNTIME") != parent {
				t.Fatal("runtime child override duplicated entries or modified parent environment")
			}
			if environment["WILDLLAMA_MOE_DYNAMIC_PREFILL"] != "1" {
				t.Fatal("runtime option changed the separate prefill feature")
			}
		})
	}
}
