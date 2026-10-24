package api

import (
	"encoding/json"
	"math"
	"testing"
)

func TestMoeRAMOptionValidation(t *testing.T) {
	for _, tt := range []struct {
		name    string
		value   any
		failure bool
	}{
		{"inherit", nil, false}, {"disabled", float64(0), false},
		{"request", float64(10240), false}, {"modelfile", int64(10240), false},
		{"maximum", float64(2097151), false}, {"negative", float64(-1), true},
		{"fraction", 10.5, true}, {"overflow", float64(2097152), true},
		{"not finite", math.NaN(), true}, {"units", "10GiB", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			opts := DefaultOptions()
			err := opts.FromMap(map[string]any{"moe_ram_limit_mib": tt.value, "num_ctx": float64(131072)})
			if (err != nil) != tt.failure {
				t.Fatalf("error=%v expected failure=%v", err, tt.failure)
			}
			if tt.failure {
				return
			}
			if opts.NumCtx != 131072 {
				t.Fatal("weight budget altered context")
			}
			data, err := json.Marshal(opts)
			if err != nil {
				t.Fatal(err)
			}
			var decoded Options
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatal(err)
			}
			if (decoded.MoeRAMLimitMiB == nil) != (tt.value == nil) {
				t.Fatal("lost distinction between inherit and zero")
			}
			if tt.value != nil && *decoded.MoeRAMLimitMiB != *opts.MoeRAMLimitMiB {
				t.Fatal("round trip altered limit")
			}
		})
	}
}
