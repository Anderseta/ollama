package llm

import (
	"fmt"
	"regexp"
	"strconv"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

var moeRAMWeightLimitRegex = regexp.MustCompile(`moe_ram: host weight budget = (\d+) bytes`)

func appendMoeRAMLimitArgs(args []string, opts api.Options, experts uint64) ([]string, error) {
	raw := envconfig.Var("OLLAMA_MOE_RAM_LIMIT_MIB")
	if raw == "" {
		raw = envconfig.Var("LLAMA_ARG_MOE_RAM_LIMIT_MIB")
	}
	if opts.MoeRAMLimitMiB != nil {
		raw = strconv.Itoa(*opts.MoeRAMLimitMiB)
	}
	if raw == "" {
		return args, nil
	}
	value, err := strconv.ParseUint(raw, 10, 32)
	if err != nil || value > 2097151 {
		return nil, fmt.Errorf("MoE RAM weight limit must be an integer between 0 and 2097151 MiB")
	}
	// A server default applies only to MoE; explicit model settings are checked by the motor.
	if experts == 0 && opts.MoeRAMLimitMiB == nil {
		value = 0
	}
	return append(args, "--moe-ram-limit-mib", strconv.FormatUint(value, 10)), nil
}
