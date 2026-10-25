package api

import "testing"

func TestDenseCUDAOptionsRequireExplicitBoolean(t *testing.T) {
	for _, value := range []string{"", "0", "1"} {
		options, err := ParseRunnerEnvironment(map[string]string{"GGML_CUDA_DENSE_OPT": value})
		if err != nil || options["GGML_CUDA_DENSE_OPT"] != value {
			t.Fatalf("explicit comparison value lost: %q %v", value, err)
		}
	}
	if _, err := ParseRunnerEnvironment(map[string]string{"GGML_CUDA_DENSE_OPT": "auto"}); err == nil {
		t.Fatal("unsupported dense mode accepted")
	}
}
