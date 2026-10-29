package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
)

func TestGPUFreeMemoryForDevices(t *testing.T) {
	a := ml.DeviceID{Library: "CUDA", ID: "0"}
	b := ml.DeviceID{Library: "CUDA", ID: "1"}
	c := ml.DeviceID{Library: "ROCm", ID: "0"}
	gpus := []ml.DeviceInfo{
		{DeviceID: a, FreeMemory: 100},
		{DeviceID: b, FreeMemory: 10000},
		{DeviceID: c, FreeMemory: 200},
	}
	for _, test := range []struct {
		name     string
		devices  []ml.DeviceID
		want     uint64
		complete bool
	}{
		{"only runner device", []ml.DeviceID{a}, 100, true},
		{"multiple backends", []ml.DeviceID{a, c}, 300, true},
		{"missing device", []ml.DeviceID{a, {Library: "CUDA", ID: "2"}}, 0, false},
		{"empty devices", nil, 0, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, complete := gpuFreeMemoryForDevices(gpus, test.devices)
			if got != test.want || complete != test.complete {
				t.Fatalf("free = %d, complete = %v; want %d, %v", got, complete, test.want, test.complete)
			}
		})
	}
}

func TestVRAMRecoveryRejectsDecreaseAndUnrelatedGPU(t *testing.T) {
	a := ml.DeviceID{Library: "CUDA", ID: "0"}
	b := ml.DeviceID{Library: "CUDA", ID: "1"}
	queries := make(chan int, 4)
	proceed := make(chan struct{})
	defer close(proceed)
	s := &Scheduler{waitForRecovery: 5 * time.Second}
	call := 0
	s.getGpuFn = func(ctx context.Context, _ []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		call++
		queries <- call
		if call == 1 {
			return []ml.DeviceInfo{{DeviceID: a, FreeMemory: 1000}, {DeviceID: b, FreeMemory: 0}}
		}
		if call == 2 {
			// Less free memory on the active GPU must not wrap to a huge recovery.
			return []ml.DeviceInfo{{DeviceID: a, FreeMemory: 900}, {DeviceID: b, FreeMemory: 10000}}
		}
		select {
		case <-proceed:
		case <-ctx.Done():
		}
		return []ml.DeviceInfo{{DeviceID: a, FreeMemory: 1800}, {DeviceID: b, FreeMemory: 10000}}
	}
	runner := &runnerRef{gpus: []ml.DeviceID{a}, discreteGPUs: true, vramSize: 1000}
	finished := s.waitForVRAMRecovery(runner, nil)
	for want := 1; want <= 3; want++ {
		select {
		case got := <-queries:
			if got != want {
				t.Fatalf("query = %d, want %d", got, want)
			}
		case <-finished:
			t.Fatal("declared recovery before the active GPU recovered")
		case <-time.After(2 * time.Second):
			t.Fatalf("did not reach query %d", want)
		}
	}
	proceed <- struct{}{}
	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("did not recognize actual VRAM recovery")
	}
}

func TestVRAMRecoveryBaselineUsesTimeout(t *testing.T) {
	s := &Scheduler{waitForRecovery: 20 * time.Millisecond}
	s.getGpuFn = func(ctx context.Context, _ []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		if _, ok := ctx.Deadline(); !ok {
			t.Fatal("baseline discovery has no deadline")
		}
		<-ctx.Done()
		return nil
	}
	runner := &runnerRef{gpus: []ml.DeviceID{{Library: "CUDA", ID: "0"}}, discreteGPUs: true, vramSize: 1000}
	finished := s.waitForVRAMRecovery(runner, nil)
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("baseline discovery exceeded the recovery timeout")
	}
}
