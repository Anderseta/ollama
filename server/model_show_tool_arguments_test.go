package server

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestModelShowCacheToolArgumentsIsolation(t *testing.T) {
	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal([]byte(`{"first":{"items":[{"value":"original"}]},"second":2}`), &args); err != nil {
		t.Fatal(err)
	}
	response := &api.ShowResponse{Messages: []api.Message{{
		Role:      "assistant",
		ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "example", Arguments: args}}},
	}}}
	cache := newModelShowCache()
	key := modelShowLocalKey{Model: "example"}
	cache.setLocal(key, "digest", response)
	mutate := func(arguments *api.ToolCallFunctionArguments) {
		arguments.Set("second", 99)
		arguments.Set("third", "new")
		first, _ := arguments.Get("first")
		items := first.(map[string]any)["items"].([]any)
		items[0].(map[string]any)["value"] = "changed"
	}
	mutate(&response.Messages[0].ToolCalls[0].Function.Arguments)
	for range 2 {
		got, ok := cache.getLocal(key, "digest")
		if !ok {
			t.Fatal("cache entry disappeared")
		}
		arguments := &got.Messages[0].ToolCalls[0].Function.Arguments
		if want := `{"first":{"items":[{"value":"original"}]},"second":2}`; arguments.String() != want {
			t.Fatalf("cached tool arguments/order = %s, want %s", arguments.String(), want)
		}
		mutate(arguments)
	}
}
