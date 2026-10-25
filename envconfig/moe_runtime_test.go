package envconfig

import "testing"

func TestMOERuntimeDefaultsAndRegistration(t *testing.T) {
	t.Setenv("OLLAMA_MOE_RUNTIME", "")
	if MOERuntime() {
		t.Fatal("experimental runtime defaults to enabled")
	}
	t.Setenv("OLLAMA_MOE_RUNTIME", "1")
	if !MOERuntime() {
		t.Fatal("runtime environment flag did not enable")
	}
	if variable, ok := AsMap()["OLLAMA_MOE_RUNTIME"]; !ok || variable.Value != "1" {
		t.Fatal("runtime flag missing from server environment reporting")
	}
}
