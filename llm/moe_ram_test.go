package llm

import (
	"io"
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestMoeRAMLimitArguments(t *testing.T) {
	zero, override, invalid := 0, 2048, -1
	for _, tt := range []struct {
		name, server, native string
		model                *int
		experts              uint64
		want                 []string
		failure              bool
	}{
		{name: "default", experts: 512},
		{name: "server", server: "10240", experts: 512, want: []string{"--moe-ram-limit-mib", "10240"}},
		{name: "native environment", native: "4096", experts: 512, want: []string{"--moe-ram-limit-mib", "4096"}},
		{name: "model override", server: "10240", model: &override, experts: 512, want: []string{"--moe-ram-limit-mib", "2048"}},
		{name: "model disable", server: "10240", model: &zero, experts: 512, want: []string{"--moe-ram-limit-mib", "0"}},
		{name: "dense default", server: "10240", want: []string{"--moe-ram-limit-mib", "0"}},
		{name: "invalid server", server: "10GiB", experts: 512, failure: true},
		{name: "negative model", model: &invalid, experts: 512, failure: true},
		{name: "overflow", server: "2147483648", experts: 512, failure: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_MOE_RAM_LIMIT_MIB", tt.server)
			t.Setenv("LLAMA_ARG_MOE_RAM_LIMIT_MIB", tt.native)
			opts := api.DefaultOptions()
			opts.MoeRAMLimitMiB = tt.model
			got, err := appendMoeRAMLimitArgs(nil, opts, tt.experts)
			if (err != nil) != tt.failure {
				t.Fatalf("error=%v failure=%v", err, tt.failure)
			}
			if !tt.failure && !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v want %v", got, tt.want)
			}
			if opts.NumCtx != api.DefaultOptions().NumCtx {
				t.Fatal("weight budget changed context")
			}
		})
	}
}

func TestMoeRAMMemoryAccounting(t *testing.T) {
	const mib = uint64(1024 * 1024)
	runner := &llamaServerRunner{}
	w := &memoryParsingWriter{inner: io.Discard, runner: runner}
	for _, line := range []string{
		"moe_ram: host weight budget = 10737418240 bytes\n",
		"load_tensors: CPU_Mapped model buffer size = 110000.00 MiB\n",
		"load_tensors: CUDA0 model buffer size = 7000.00 MiB\n",
		"llama_kv_cache: CUDA0 KV buffer size = 2000.00 MiB\n",
		"llama_context: CPU compute buffer size = 500.00 MiB\n",
		"load_tensors: offloaded 49/49 layers to GPU\n",
	} {
		if _, err := w.Write([]byte(line)); err != nil {
			t.Fatal(err)
		}
	}
	total, vram := runner.MemorySize()
	if total != (10240+7000+2000+500)*mib || vram != 9000*mib {
		t.Fatalf("memory accounting lost weight cache or non-weight buffers: total=%d vram=%d", total, vram)
	}
}

func TestMoeRAMWarmupConfiguration(t *testing.T) {
	for _, tt := range []struct {
		raw           string
		flag, failure bool
	}{
		{"", false, false}, {"1", true, false}, {"true", true, false},
		{"0", false, false}, {"false", false, false}, {"invalid", false, true},
	} {
		t.Run(tt.raw, func(t *testing.T) {
			t.Setenv("LLAMA_ARG_NO_WARMUP", tt.raw)
			got, err := appendWarmupArgs(nil)
			if (err != nil) != tt.failure {
				t.Fatalf("error=%v failure=%v", err, tt.failure)
			}
			if tt.failure {
				return
			}
			if len(got) != 0 && (len(got) != 1 || got[0] != "--no-warmup") {
				t.Fatalf("unexpected flags %v", got)
			}
			if (len(got) > 0) != tt.flag {
				t.Fatalf("flags=%v expected no-warmup=%v", got, tt.flag)
			}
		})
	}
}
