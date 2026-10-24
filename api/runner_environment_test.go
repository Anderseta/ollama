package api

import (
	"encoding/json"
	"testing"
)

func TestRunnerEnvironmentValidation(t *testing.T) {
	for _, input := range []string{
		`{"runner_env":{"PATH":"evil"}}`,
		`{"runner_env":{"LLAMA_ARG_MODEL":"other.gguf"}}`,
		`{"runner_env":{"OLLAMA_NUM_PARALLEL":"0"}}`,
		`{"runner_env":{"LLAMA_ARG_FIT_TARGET":"4096,-1"}}`,
		`{"runner_env":{"LLAMA_ARG_MOE_RAM_LIMIT_MIB":"2097152"}}`,
		`{"runner_env":{"LLAMA_ARG_CPU_MOE":"true\n--model=other"}}`,
		`{"runner_env":{"LLAMA_ARG_UBATCH":128}}`,
		`{"num_ubatch":-1}`, `{"num_ubatch":1.5}`,
	} {
		var values map[string]any
		if err := json.Unmarshal([]byte(input), &values); err != nil {
			t.Fatal(err)
		}
		opts := DefaultOptions()
		if err := opts.FromMap(values); err == nil {
			t.Errorf("accepted %s", input)
		}
	}
	values := map[string]string{"LLAMA_ARG_CACHE_RAM": "8192", "LLAMA_ARG_FIT_TARGET": "4096,1024", "LLAMA_ARG_UBATCH": "128", "OLLAMA_FLASH_ATTENTION": "1"}
	opts := DefaultOptions()
	if err := opts.FromMap(map[string]any{"runner_env": values, "num_ubatch": float64(128)}); err != nil {
		t.Fatal(err)
	}
	values["LLAMA_ARG_CACHE_RAM"] = "1"
	if opts.RunnerEnv["LLAMA_ARG_CACHE_RAM"] != "8192" || opts.NumUbatch != 128 {
		t.Fatal("options alias input or lost physical batch")
	}
}
