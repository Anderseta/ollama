package api

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestReasoningBudgetValidationAndExplicitZero(t *testing.T) {
	for _, value := range []any{int64(-1), float64(0), int64(256), float64(2097152)} {
		opts := DefaultOptions()
		if err := opts.FromMap(map[string]any{"reasoning_budget": value}); err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(opts)
		if err != nil || opts.ReasoningBudget == nil || !strings.Contains(string(encoded), `"reasoning_budget":`) {
			t.Fatalf("budget omitted: %s %v", encoded, err)
		}
	}
	for _, value := range []any{float64(-2), float64(0.5), math.Inf(1), math.NaN(), float64(2097153), true, "256"} {
		opts := DefaultOptions()
		if opts.FromMap(map[string]any{"reasoning_budget": value}) == nil {
			t.Fatalf("accepted invalid budget %v", value)
		}
	}
	encoded, _ := json.Marshal(DefaultOptions())
	if strings.Contains(string(encoded), `"reasoning_budget"`) {
		t.Fatal("default request acquired a thinking limit")
	}
}
