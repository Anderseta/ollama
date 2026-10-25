package api

import (
	"fmt"
	"strconv"
	"strings"
)

// RunnerEnvironment contains only model-runner settings, never process paths or credentials.
func ParseRunnerEnvironment(value any) (map[string]string, error) {
	result := make(map[string]string)
	switch values := value.(type) {
	case map[string]string:
		for key, value := range values {
			result[key] = value
		}
	case map[string]any:
		for key, value := range values {
			text, ok := value.(string)
			if !ok {
				return nil, fmt.Errorf("runner_env %q must be a string", key)
			}
			result[key] = text
		}
	default:
		return nil, fmt.Errorf("runner_env must be an object of strings")
	}
	for key, value := range result {
		if len(value) > 2048 || strings.ContainsAny(value, "\x00\r\n") {
			return nil, fmt.Errorf("invalid runner_env value for %q", key)
		}
		var choices string
		switch key {
		case "LLAMA_ARG_LOAD_MODE":
			choices = "|auto|mmap|none|dio"
		case "LLAMA_ARG_LAZY_MODE", "LLAMA_ARG_FIT", "LLAMA_ARG_FLASH_ATTN":
			choices = "|auto|on|off"
		case "LLAMA_ARG_CPU_MOE", "LLAMA_ARG_NO_WARMUP", "OLLAMA_FLASH_ATTENTION", "LLAMA_ARG_GPU_EXCLUSIVE", "LLAMA_ARG_GPU_DIAGNOSTICS", "WILDLLAMA_MOE_RUNTIME", "GGML_CUDA_DENSE_OPT":
			choices = "|0|1"
		case "OLLAMA_KV_CACHE_TYPE", "LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V":
			choices = "|f16|bf16|q8_0|q4_0|q4_1|q5_0|q5_1|iq4_nl"
		case "LLAMA_ARG_N_GPU_LAYERS":
			if value == "auto" || value == "all" {
				continue
			}
		case "LLAMA_ARG_FIT_TARGET":
			if value == "" {
				continue
			}
			for _, item := range strings.Split(value, ",") {
				if err := runnerInteger(key, item, 0, 2097152); err != nil {
					return nil, err
				}
			}
			continue
		case "LLAMA_ARG_OVERRIDE_TENSOR":
			continue
		case "LLAMA_ARG_N_CPU_MOE", "LLAMA_ARG_MOE_CACHE_MIB", "LLAMA_ARG_CACHE_RAM", "LLAMA_ARG_MOE_RAM_LIMIT_MIB", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB":
		case "LLAMA_ARG_BATCH", "LLAMA_ARG_UBATCH":
			if value != "" {
				if err := runnerInteger(key, value, 1, 2097152); err != nil {
					return nil, err
				}
			}
			continue
		case "OLLAMA_NUM_PARALLEL":
			if value != "" {
				if err := runnerInteger(key, value, 1, 64); err != nil {
					return nil, err
				}
			}
			continue
		default:
			return nil, fmt.Errorf("unsupported runner_env key %q", key)
		}
		if choices != "" {
			valid := false
			for _, choice := range strings.Split(choices, "|") {
				valid = valid || value == choice
			}
			if !valid {
				return nil, fmt.Errorf("invalid runner_env %s: %q", key, value)
			}
		} else if value != "" {
			maximum := int64(2097152)
			if key == "LLAMA_ARG_MOE_RAM_LIMIT_MIB" || key == "LLAMA_ARG_GPU_VRAM_LIMIT_MIB" {
				maximum--
			}
			if err := runnerInteger(key, value, 0, maximum); err != nil {
				return nil, err
			}
		}
	}
	if len(result) == 0 {
		return nil, nil
	}
	return result, nil
}

func runnerInteger(key, value string, minimum, maximum int64) error {
	n, err := strconv.ParseInt(value, 10, 64)
	if err != nil || n < minimum || n > maximum {
		return fmt.Errorf("runner_env %s must be an integer between %d and %d", key, minimum, maximum)
	}
	return nil
}
