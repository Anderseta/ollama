package discover

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/ollama/ollama/ml"
)

func TestCUDAAllocationPropLayout(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("64-bit CUDA driver allocation ABI")
	}
	var prop cudaAllocationProp
	if unsafe.Sizeof(prop) != 32 || unsafe.Offsetof(prop.LocationID) != 12 ||
		unsafe.Offsetof(prop.Win32HandleMeta) != 16 || unsafe.Offsetof(prop.AllocationFlags) != 24 {
		t.Fatalf("CUmemAllocationProp layout mismatch: size=%d metadata=%d flags=%d", unsafe.Sizeof(prop), unsafe.Offsetof(prop.Win32HandleMeta), unsafe.Offsetof(prop.AllocationFlags))
	}
}

func TestCUDAProbeCapabilitiesOptionalAttributes(t *testing.T) {
	attrs := map[int]int{16: 36, 10: 32, 8: 49152, 97: 101376, 36: 14000000, 37: 128, 102: 1}
	got := cudaProbeCapabilities(func(attr int) (int, bool) { value, ok := attrs[attr]; return value, ok })
	want := &ml.DeviceCapabilities{Multiprocessors: 36, WarpSize: 32, SharedMemoryPerBlock: 49152,
		SharedMemoryPerBlockOptin: 101376, MemoryClockRateKHz: 14000000, MemoryBusWidthBits: 128,
		VirtualMemoryKnown: true, VirtualMemoryManagement: true}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("capabilities = %+v, want %+v", got, want)
	}
	missing := cudaProbeCapabilities(func(int) (int, bool) { return 0, false })
	if !reflect.DeepEqual(missing, &ml.DeviceCapabilities{}) {
		t.Fatalf("unavailable driver attributes acquired capabilities: %+v", missing)
	}
	unsupported := cudaProbeCapabilities(func(attr int) (int, bool) {
		if attr == 102 {
			return 0, true
		}
		return -1, true
	})
	if !reflect.DeepEqual(unsupported, &ml.DeviceCapabilities{VirtualMemoryKnown: true}) {
		t.Fatalf("unsupported VMM or negative attributes misreported: %+v", unsupported)
	}
}

func TestCUDAUUIDString(t *testing.T) {
	if got := cudaUUIDString([16]byte{}); got != "" {
		t.Fatalf("zero UUID = %q, want unknown", got)
	}
	uuid := [16]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	if got := cudaUUIDString(uuid); got != "00010203-0405-0607-0809-0a0b0c0d0e0f" {
		t.Fatalf("UUID bytes changed: %q", got)
	}
}

func TestNativeCapabilitiesMergeAndPublicDevice(t *testing.T) {
	capabilities := &ml.DeviceCapabilities{Multiprocessors: 36, WarpSize: 32, VirtualMemoryKnown: true,
		VirtualMemoryManagement: true, VirtualMemoryGranularity: 65536}
	base := []nativeProbeDevice{
		{Library: "CUDA", Index: 0, IndexMatchesBackend: true, DeviceID: "0000:02:00.0"},
		{Library: "CUDA", Index: 1, IndexMatchesBackend: true, DeviceID: "0000:01:00.0"},
	}
	// Supplement order differs. PCI identity, rather than physical ordinal,
	// determines which backend-visible device receives the capabilities.
	supplement := []nativeProbeDevice{{Library: "CUDA", Index: 0, DeviceID: "0000:01:00.0",
		PhysicalUUID: "private-device-uuid", NVIDIADriverVersion: "621.01", NVIDIADriverMajor: 621,
		CUDADriverMajor: 13, CUDADriverMinor: 1, ComputeMajor: 12, Capabilities: capabilities}}
	devices := mergeNativeProbeDevices(base, supplement)
	if devices[0].Capabilities != nil || !reflect.DeepEqual(devices[1].Capabilities, capabilities) {
		t.Fatalf("capabilities attached to wrong filtered device: %+v", devices)
	}
	// Metadata must survive the isolated subprocess JSON transport.
	data, err := json.Marshal(devices)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &devices); err != nil {
		t.Fatal(err)
	}
	output := `system_info: n_threads = 4 | CUDA : ARCHS = 1200 |
Available devices:
  CUDA0: Example GPU (16000 MiB, 15000 MiB free)
  CUDA1: Example GPU (16000 MiB, 15000 MiB free)
`
	public := parseLlamaServerDevicesWithNative(output, "", []string{"/lib/ollama", "/lib/ollama/cuda_v13"}, devices)
	if len(public) != 2 || public[0].Capabilities != nil || !reflect.DeepEqual(public[1].Capabilities, capabilities) {
		t.Fatalf("native capabilities lost or reassigned: %+v", public)
	}
	if public[1].PhysicalUUID != "private-device-uuid" || public[1].NVIDIADriverVersion != "621.01" || public[1].Driver() != "13.1" {
		t.Fatalf("UUID/full driver metadata lost: %+v", public[1])
	}
	data, err = json.Marshal(public)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "private-device-uuid") {
		t.Fatalf("internal UUID exposed to public API: %s", data)
	}
}

func TestNativeProbeUUIDPreventsWrongMerge(t *testing.T) {
	a := nativeProbeDevice{Library: "CUDA", Index: 0, IndexMatchesBackend: true, DeviceID: "0000:01:00.0", PhysicalUUID: "partition-a"}
	b := a
	b.PhysicalUUID = "partition-b"
	if sameNativeProbeDevice(a, b) {
		t.Fatal("different UUIDs were merged despite a shared PCI bus (for example MIG partitions)")
	}
	b = a
	b.Index, b.DeviceID = 2, "0000:09:00.0"
	if !sameNativeProbeDevice(a, b) {
		t.Fatal("stable UUID identity was lost after visible ordinal/PCI reassignment")
	}
}

func TestGGMLBackendDevPropsLayout(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("GGML probe layout assertions are for 64-bit builds")
	}

	var props ggmlBackendDevProps
	if got, want := unsafe.Sizeof(props), uintptr(56); got != want {
		t.Fatalf("ggmlBackendDevProps size = %d, want %d", got, want)
	}
	checks := []struct {
		name string
		got  uintptr
		want uintptr
	}{
		{"Name", unsafe.Offsetof(props.Name), 0},
		{"Description", unsafe.Offsetof(props.Description), 8},
		{"MemoryFree", unsafe.Offsetof(props.MemoryFree), 16},
		{"MemoryTotal", unsafe.Offsetof(props.MemoryTotal), 24},
		{"Type", unsafe.Offsetof(props.Type), 32},
		{"DeviceID", unsafe.Offsetof(props.DeviceID), 40},
		{"Caps", unsafe.Offsetof(props.Caps), 48},
	}
	for _, tt := range checks {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.want {
				t.Fatalf("offset = %d, want %d", tt.got, tt.want)
			}
		})
	}
	if got, want := unsafe.Sizeof(ggmlBackendDevCaps{}), uintptr(4); got != want {
		t.Fatalf("ggmlBackendDevCaps size = %d, want %d", got, want)
	}
}

func TestParseLlamaServerDevicesUsesNativeCUDAComputeCapability(t *testing.T) {
	output := `system_info: n_threads = 4 | CUDA : ARCHS = 750,800 |
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 6GB (6063 MiB, 5900 MiB free)
`
	devices := parseLlamaServerDevicesWithNative(output, "", []string{"/lib/ollama", "/lib/ollama/cuda_v13"}, []nativeProbeDevice{{
		Library:             "CUDA",
		Index:               0,
		IndexMatchesBackend: true,
		DeviceID:            "0000:01:00.0",
		ComputeMajor:        6,
		ComputeMinor:        1,
		CUDADriverMajor:     13,
		NVIDIADriverMajor:   570,
	}})
	if len(devices) != 0 {
		t.Fatalf("got %d devices, want unsupported CUDA device filtered", len(devices))
	}

	output = `system_info: n_threads = 4 | CUDA : ARCHS = 610,750,800 |
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 6GB (6063 MiB, 5900 MiB free)
`
	devices = parseLlamaServerDevicesWithNative(output, "", []string{"/lib/ollama", "/lib/ollama/cuda_v12"}, []nativeProbeDevice{{
		Library:             "CUDA",
		Index:               0,
		IndexMatchesBackend: true,
		DeviceID:            "0000:01:00.0",
		ComputeMajor:        6,
		ComputeMinor:        1,
		CUDADriverMajor:     12,
		NVIDIADriverMajor:   570,
	}})
	if len(devices) != 1 {
		t.Fatalf("got %d devices, want 1", len(devices))
	}
	got := devices[0]
	if got.Compute() != "6.1" {
		t.Fatalf("compute = %q, want 6.1", got.Compute())
	}
	if got.PCIID != "0000:01:00.0" {
		t.Fatalf("PCIID = %q, want 0000:01:00.0", got.PCIID)
	}
	if got.Driver() != "12.0" {
		t.Fatalf("driver = %q, want 12.0", got.Driver())
	}
	if got.NVIDIADriverMajor != 570 {
		t.Fatalf("NVIDIADriverMajor = %d, want 570", got.NVIDIADriverMajor)
	}
}

func TestParseLlamaServerDevicesUsesNativeROCmMetadata(t *testing.T) {
	output := `ggml_vulkan: 0 = AMD Radeon RX 7600 | uma: 1 | fp16: 1 |
Available devices:
  ROCm0: AMD Radeon RX 7600 (8176 MiB, 7900 MiB free)
`
	devices := parseLlamaServerDevicesWithNative(output, "", []string{"/lib/ollama", "/lib/ollama/rocm_v7_2"}, []nativeProbeDevice{{
		Library:             "ROCm",
		Index:               0,
		IndexMatchesBackend: true,
		DeviceID:            "0000:03:00.0",
		GFXTarget:           "gfx1102",
		Integrated:          false,
		IntegratedKnown:     true,
	}})
	if len(devices) != 1 {
		t.Fatalf("got %d devices, want 1", len(devices))
	}
	got := devices[0]
	if got.PCIID != "0000:03:00.0" {
		t.Fatalf("PCIID = %q, want 0000:03:00.0", got.PCIID)
	}
	if got.GFXTarget != "gfx1102" {
		t.Fatalf("GFXTarget = %q, want gfx1102", got.GFXTarget)
	}
	if got.Compute() != "gfx1102" {
		t.Fatalf("compute = %q, want gfx1102", got.Compute())
	}
	if got.Integrated {
		t.Fatalf("Integrated = true, want false")
	}
}

func TestNVIDIADriverMajorFromDevices(t *testing.T) {
	devices := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{Library: "CUDA"}, NVIDIADriverMajor: 565},
	}
	if got := nvidiaDriverMajorFromDevices(devices); got != 565 {
		t.Fatalf("driver = %d, want 565", got)
	}
}

func TestMergeNativeProbeDevicesAvoidsUnreliableIndexMatch(t *testing.T) {
	tests := []struct {
		name       string
		base       []nativeProbeDevice
		supplement []nativeProbeDevice
		wantLen    int
		wantPCI    string
		wantKnown  bool
		wantIGPU   bool
	}{
		{
			name: "filtered sysfs cannot overwrite a different backend device by index",
			base: []nativeProbeDevice{{
				Library:             "ROCm",
				Index:               0,
				IndexMatchesBackend: true,
				DeviceID:            "0000:03:00.0",
			}},
			supplement: []nativeProbeDevice{{
				Library:         "ROCm",
				Index:           0,
				DeviceID:        "0000:04:00.0",
				Integrated:      true,
				IntegratedKnown: true,
			}},
			wantLen: 1,
			wantPCI: "0000:03:00.0",
		},
		{
			name: "filtered sysfs can still merge by PCI ID",
			base: []nativeProbeDevice{{
				Library:             "ROCm",
				Index:               0,
				IndexMatchesBackend: true,
				DeviceID:            "0000:04:00.0",
			}},
			supplement: []nativeProbeDevice{{
				Library:         "ROCm",
				Index:           1,
				DeviceID:        "0000:04:00.0",
				Integrated:      true,
				IntegratedKnown: true,
			}},
			wantLen:   1,
			wantPCI:   "0000:04:00.0",
			wantKnown: true,
			wantIGPU:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mergeNativeProbeDevices(tt.base, tt.supplement)
			if len(got) != tt.wantLen {
				t.Fatalf("got %d devices, want %d: %#v", len(got), tt.wantLen, got)
			}
			if got[0].DeviceID != tt.wantPCI {
				t.Fatalf("DeviceID = %q, want %q", got[0].DeviceID, tt.wantPCI)
			}
			if got[0].IntegratedKnown != tt.wantKnown {
				t.Fatalf("IntegratedKnown = %v, want %v", got[0].IntegratedKnown, tt.wantKnown)
			}
			if got[0].Integrated != tt.wantIGPU {
				t.Fatalf("Integrated = %v, want %v", got[0].Integrated, tt.wantIGPU)
			}
		})
	}
}
