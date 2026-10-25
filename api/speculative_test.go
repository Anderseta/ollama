package api

import "testing"

func TestOptionsSpeculativeType(t *testing.T) {
	for _, mode := range []string{"", "auto", "none", "draft-mtp", "draft-eagle3", "ngram-simple", "ngram-mod", "ngram-cache"} {
		t.Run("allowed_"+mode, func(t *testing.T) {
			var opts Options
			if err := opts.FromMap(map[string]any{"speculative_type": mode}); err != nil {
				t.Fatalf("FromMap rejected %q: %v", mode, err)
			}
			if opts.SpeculativeType != mode {
				t.Fatalf("SpeculativeType = %q, want %q", opts.SpeculativeType, mode)
			}
		})
	}

	for _, mode := range []string{"unknown", "draft-simple", "draft-dflash", "draft-dspark", "DRAFT-MTP", "draft-mtp,ngram-simple"} {
		t.Run("rejected_"+mode, func(t *testing.T) {
			var opts Options
			if err := opts.FromMap(map[string]any{"speculative_type": mode}); err == nil {
				t.Fatalf("FromMap accepted unsupported mode %q", mode)
			}
		})
	}

	var opts Options
	if err := opts.FromMap(map[string]any{"speculative_type": 1}); err == nil {
		t.Fatal("FromMap accepted a non-string speculative type")
	}
}
