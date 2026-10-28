package ml

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
)

// DeviceCapabilities describes optional hardware limits reported by a backend.
// It deliberately makes no kernel selection decisions. Zero numeric values and
// absent capability groups mean unknown, so backends with partial detection
// retain their existing conservative behavior.
type DeviceCapabilities struct {
	Multiprocessors           int    `json:"multiprocessors,omitempty"`
	WarpSize                  int    `json:"warp_size,omitempty"`
	SharedMemoryPerBlock      uint64 `json:"shared_memory_per_block,omitempty"`
	SharedMemoryPerBlockOptin uint64 `json:"shared_memory_per_block_optin,omitempty"`
	MemoryBusWidthBits        int    `json:"memory_bus_width_bits,omitempty"`
	MemoryClockRateKHz        int    `json:"memory_clock_rate_khz,omitempty"`
	VirtualMemoryKnown        bool   `json:"virtual_memory_known,omitempty"`
	VirtualMemoryManagement   bool   `json:"virtual_memory_management,omitempty"`
	VirtualMemoryGranularity  uint64 `json:"virtual_memory_granularity,omitempty"`
}

// CapabilityFingerprint returns an opaque, versioned identity for future
// measured policies. payloadID must identify the exact native backend build,
// rather than just the Go server version. A caller must additionally key any
// observation by operation, quantization, shape/strides and relevant options.
//
// Device ordinals, names, free memory and instantaneous clocks are excluded:
// visibility filters and unrelated workloads must not change the identity.
// No tuning or persistence is enabled by computing this fingerprint.
func (d DeviceInfo) CapabilityFingerprint(payloadID string) (string, bool) {
	backend := strings.ToLower(strings.TrimSpace(d.Library))
	identity := strings.ToLower(strings.TrimSpace(d.PhysicalUUID))
	identityKind := "uuid"
	if identity == "" {
		identity = strings.ToLower(strings.TrimSpace(d.PCIID))
		identityKind = "pci"
	}
	if strings.TrimSpace(payloadID) == "" || backend == "" || identity == "" || d.TotalMemory == 0 {
		return "", false
	}
	driver := strings.TrimSpace(d.NVIDIADriverVersion)
	if backend == "cuda" {
		// A CUDA compatibility version alone cannot distinguish NVIDIA driver
		// patch releases. Missing optional NVML data is a safe cache miss.
		if driver == "" || d.ComputeMajor <= 0 {
			return "", false
		}
	} else {
		if d.DriverMajor <= 0 {
			return "", false
		}
		driver = d.Driver()
	}
	key := struct {
		Schema       int
		Payload      string
		Backend      string
		IdentityKind string
		Identity     string
		Driver       string
		Compute      string
		GFXTarget    string
		TotalMemory  uint64
		Integrated   bool
		Capabilities *DeviceCapabilities
	}{1, payloadID, backend, identityKind, identity, driver, d.Compute(), d.GFXTarget, d.TotalMemory, d.Integrated, d.Capabilities}
	data, err := json.Marshal(key)
	if err != nil {
		return "", false
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), true
}
