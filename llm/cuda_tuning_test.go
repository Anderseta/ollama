package llm

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/ml"
)

func tuningTestPayload(t *testing.T) (string, string) {
	t.Helper()
	root := t.TempDir()
	backendDir := filepath.Join(root, "cuda")
	if err := os.Mkdir(backendDir, 0700); err != nil {
		t.Fatal(err)
	}
	exe, backend := filepath.Join(root, "llama-server.exe"), filepath.Join(backendDir, "ggml-cuda.dll")
	for path, content := range map[string]string{
		exe: "launcher", backend: "CUDA kernel payload", filepath.Join(root, "llama-server-impl.dll"): "server implementation",
	} {
		if err := os.WriteFile(path, []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
	}
	return exe, backend
}

func TestCUDATuningPayloadIdentity(t *testing.T) {
	exe, backend := tuningTestPayload(t)
	first, err := cudaTuningPayloadID(exe, backend)
	if err != nil {
		t.Fatal(err)
	}
	second, err := cudaTuningPayloadID(exe, backend)
	if err != nil || first != second {
		t.Fatalf("unchanged payload identity: %q %q %v", first, second, err)
	}
	implementation := filepath.Join(filepath.Dir(exe), "llama-server-impl.dll")
	if err := os.WriteFile(implementation, []byte("updated server implementation"), 0600); err != nil {
		t.Fatal(err)
	}
	third, err := cudaTuningPayloadID(exe, backend)
	if err != nil || third == first {
		t.Fatalf("implementation DLL must invalidate policy: %q %q %v", first, third, err)
	}
	if err := os.Remove(backend); err != nil {
		t.Fatal(err)
	}
	if _, err := cudaTuningPayloadID(exe, backend); err == nil {
		t.Fatal("missing backend reused a memoized identity")
	}
}

func TestCUDATuningEnvironment(t *testing.T) {
	exe, backend := tuningTestPayload(t)
	device := ml.DeviceInfo{
		DeviceID: ml.DeviceID{Library: "cuda", ID: "0"}, PhysicalUUID: "test-gpu",
		TotalMemory: 16 << 30, ComputeMajor: 12, NVIDIADriverVersion: "621.01",
	}
	libs := []string{filepath.Dir(backend)}
	env := map[string]string{"GGML_CUDA_TUNING_CACHE": "inherited", "GGML_CUDA_TUNING_FINGERPRINT": "inherited"}
	t.Setenv("WILDLLAMA_CUDA_TUNING_DIR", "")
	setCUDATuningEnvironment(env, exe, []ml.DeviceInfo{device}, libs)
	if env["GGML_CUDA_TUNING_CACHE"] != "" || env["GGML_CUDA_TUNING_FINGERPRINT"] != "" {
		t.Fatal("unconfigured runner inherited an unvalidated policy")
	}
	directory := t.TempDir()
	t.Setenv("WILDLLAMA_CUDA_TUNING_DIR", directory)
	payload, err := cudaTuningPayloadID(exe, backend)
	if err != nil {
		t.Fatal(err)
	}
	fingerprint, ok := device.CapabilityFingerprint(payload)
	if !ok {
		t.Fatal("test device must have a complete identity")
	}
	policy := filepath.Join(directory, fingerprint+".policy")
	setCUDATuningEnvironment(env, exe, []ml.DeviceInfo{device}, libs)
	if env["GGML_CUDA_TUNING_CACHE"] != "" {
		t.Fatal("missing policy must retain normal dispatch")
	}
	if err := os.WriteFile(policy, []byte("native parser validates the policy contents"), 0600); err != nil {
		t.Fatal(err)
	}
	setCUDATuningEnvironment(env, exe, []ml.DeviceInfo{device}, libs)
	if env["GGML_CUDA_TUNING_CACHE"] != policy || env["GGML_CUDA_TUNING_FINGERPRINT"] != fingerprint {
		t.Fatal("matching local policy was not supplied to the native validator")
	}
	device.NVIDIADriverVersion = "621.02"
	setCUDATuningEnvironment(env, exe, []ml.DeviceInfo{device}, libs)
	if env["GGML_CUDA_TUNING_CACHE"] != "" {
		t.Fatal("driver update reused an old policy")
	}
	device.NVIDIADriverVersion = "621.01"
	for _, devices := range [][]ml.DeviceInfo{nil, {device, device}, {{DeviceID: ml.DeviceID{Library: "ROCm"}}}, {{DeviceID: device.DeviceID, TotalMemory: 16 << 30}}} {
		setCUDATuningEnvironment(env, exe, devices, libs)
		if env["GGML_CUDA_TUNING_CACHE"] != "" {
			t.Fatal("unsupported or unidentified hardware reused a CUDA policy")
		}
	}
	t.Setenv("WILDLLAMA_CUDA_TUNING_DIR", "relative-cache")
	setCUDATuningEnvironment(env, exe, []ml.DeviceInfo{device}, libs)
	if env["GGML_CUDA_TUNING_CACHE"] != "" {
		t.Fatal("relative policy path was accepted")
	}
}
