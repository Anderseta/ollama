package ml

import (
	"encoding/json"
	"strings"
	"testing"
)

func fingerprintTestDevice() DeviceInfo {
	return DeviceInfo{
		DeviceID: DeviceID{Library: "CUDA", ID: "0"},
		PCIID:    "0000:01:00.0", PhysicalUUID: "test-physical-uuid",
		TotalMemory: 16 << 30, FreeMemory: 12 << 30,
		ComputeMajor: 12, DriverMajor: 13, DriverMinor: 1,
		NVIDIADriverVersion: "621.01",
		Capabilities:        &DeviceCapabilities{Multiprocessors: 36, WarpSize: 32},
	}
}

func TestCapabilityFingerprintStableAcrossVisibilityAndLoad(t *testing.T) {
	a := fingerprintTestDevice()
	b := a
	b.ID, b.FilterID, b.Name, b.Description, b.PCIID = "3", "7", "CUDA3", "Different label", "0000:09:00.0"
	b.FreeMemory = 1 << 30
	got, ok := a.CapabilityFingerprint("native-payload-sha256")
	if !ok || len(got) != 64 {
		t.Fatalf("invalid fingerprint %q (valid=%v)", got, ok)
	}
	if next, ok := b.CapabilityFingerprint("native-payload-sha256"); !ok || got != next {
		t.Fatalf("visibility/load changed the physical UUID fingerprint: %q != %q", got, next)
	}
}

func TestCapabilityFingerprintInvalidatesMeasuredPolicies(t *testing.T) {
	base := fingerprintTestDevice()
	want, _ := base.CapabilityFingerprint("payload-A")
	tests := []struct {
		name string
		edit func(*DeviceInfo)
	}{
		{"driver patch", func(d *DeviceInfo) { d.NVIDIADriverVersion = "621.02" }},
		{"physical device", func(d *DeviceInfo) { d.PhysicalUUID = "other-physical-uuid" }},
		{"backend", func(d *DeviceInfo) { d.Library = "Vulkan" }},
		{"compute", func(d *DeviceInfo) { d.ComputeMinor++ }},
		{"memory", func(d *DeviceInfo) { d.TotalMemory *= 2 }},
		{"capabilities", func(d *DeviceInfo) { d.Capabilities = &DeviceCapabilities{Multiprocessors: 32, WarpSize: 32} }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			device := base
			tt.edit(&device)
			if got, ok := device.CapabilityFingerprint("payload-A"); !ok || got == want {
				t.Fatalf("changed hardware/runtime retained old fingerprint: %q (valid=%v)", got, ok)
			}
		})
	}
	if got, _ := base.CapabilityFingerprint("payload-B"); got == want {
		t.Fatal("a different backend payload retained the old fingerprint")
	}
}

func TestCapabilityFingerprintRequiresReliableIdentityAndDriver(t *testing.T) {
	base := fingerprintTestDevice()
	base.PhysicalUUID = ""
	if _, ok := base.CapabilityFingerprint("payload-A"); !ok {
		t.Fatal("PCI identity fallback was rejected")
	}
	changedPCI := base
	changedPCI.PCIID = "0000:02:00.0"
	a, _ := base.CapabilityFingerprint("payload-A")
	b, _ := changedPCI.CapabilityFingerprint("payload-A")
	if a == b {
		t.Fatal("different PCI devices shared the fallback identity")
	}
	base.NVIDIADriverVersion = ""
	if _, ok := base.CapabilityFingerprint("payload-A"); ok {
		t.Fatal("a CUDA compatibility version was accepted as a full kernel driver version")
	}
	base = fingerprintTestDevice()
	base.PhysicalUUID, base.PCIID = "", ""
	if _, ok := base.CapabilityFingerprint("payload-A"); ok {
		t.Fatal("a device ordinal was accepted as stable physical identity")
	}
	if _, ok := fingerprintTestDevice().CapabilityFingerprint(""); ok {
		t.Fatal("a missing native payload identity was accepted")
	}
	if _, ok := (DeviceInfo{DeviceID: DeviceID{Library: "Metal"}}).CapabilityFingerprint("payload-A"); ok {
		t.Fatal("partial unsupported detection must leave default policy intact")
	}
}

func TestDeviceCapabilitiesDoNotExposePhysicalUUID(t *testing.T) {
	device := fingerprintTestDevice()
	data, err := json.Marshal(device)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), device.PhysicalUUID) || strings.Contains(string(data), "physical_uuid") {
		t.Fatalf("physical UUID leaked to public device JSON: %s", data)
	}
	if !strings.Contains(string(data), `"multiprocessors":36`) || !strings.Contains(string(data), `"nvidia_driver_version":"621.01"`) {
		t.Fatalf("public capability metadata missing: %s", data)
	}
}
