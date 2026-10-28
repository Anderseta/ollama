package api

import (
	"encoding/json"
	"testing"
)

func TestSpeculativeMetricsOmissionAndZero(t *testing.T) {
	for _, present := range []bool{false, true} {
		metrics := Metrics{}
		if present {
			draft, accepted := 4, 0
			metrics.SpeculativeDraftTokens = &draft
			metrics.SpeculativeAcceptedTokens = &accepted
		}
		encoded, err := json.Marshal(ChatResponse{Metrics: metrics})
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err = json.Unmarshal(encoded, &fields); err != nil {
			t.Fatal(err)
		}
		value, exists := fields["speculative_accepted_tokens"]
		if exists != present || present && string(value) != "0" {
			t.Fatalf("missing versus known-zero was lost: %s", encoded)
		}
	}
}
