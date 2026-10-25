package cmd

import (
	"os"
	"strconv"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

func configureGPUExecution(cmd *cobra.Command) error {
	if cmd.Flags().Changed("moe-runtime") {
		enabled, err := cmd.Flags().GetBool("moe-runtime")
		if err != nil {
			return err
		}
		if err := os.Setenv("OLLAMA_MOE_RUNTIME", strconv.FormatBool(enabled)); err != nil {
			return err
		}
	}
	if cmd.Flags().Changed("gpu-diagnostics") {
		enabled, err := cmd.Flags().GetBool("gpu-diagnostics")
		if err != nil {
			return err
		}
		if err := os.Setenv("OLLAMA_GPU_DIAGNOSTICS", strconv.FormatBool(enabled)); err != nil {
			return err
		}
	}
	if cmd.Flags().Changed("gpu-exclusive") {
		enabled, err := cmd.Flags().GetBool("gpu-exclusive")
		if err != nil {
			return err
		}
		if err := os.Setenv("OLLAMA_GPU_EXCLUSIVE", strconv.FormatBool(enabled)); err != nil {
			return err
		}
	}
	if cmd.Flags().Changed("gpu-vram-limit-mib") {
		limit, err := cmd.Flags().GetInt("gpu-vram-limit-mib")
		if err != nil {
			return err
		}
		if err := os.Setenv("OLLAMA_GPU_VRAM_LIMIT_MIB", strconv.Itoa(limit)); err != nil {
			return err
		}
	}
	_, err := api.ResolveGPUExecution(api.DefaultOptions())
	return err
}
