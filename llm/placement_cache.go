package llm

import (
	"crypto/sha256"
	"encoding/json"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/ml"
)

type placementObservation struct {
	bytes uint64
	used  uint64
}

var placements = struct {
	sync.Mutex
	clock   uint64
	entries map[[32]byte]placementObservation
}{entries: make(map[[32]byte]placementObservation)}

func placementKey(path string, model *gguf.Model, opts api.Options, devices []ml.DeviceInfo, parallel int, config LlamaServerConfig) [32]byte {
	if trained := int(model.KV().ContextLength()); trained > 0 && opts.NumCtx > trained {
		opts.NumCtx = trained
	}
	var gpu []any
	for _, device := range devices {
		gpu = append(gpu, []any{device.DeviceID, device.TotalMemory, device.Driver(), device.Compute()})
	}
	var environment []string
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if strings.HasPrefix(key, "LLAMA_ARG_") || strings.HasPrefix(key, "GGML_") || key == "OLLAMA_FLASH_ATTENTION" || key == "OLLAMA_KV_CACHE_TYPE" {
			environment = append(environment, entry)
		}
	}
	slices.Sort(environment)
	var modified int64
	if info, err := os.Stat(path); err == nil {
		modified = info.ModTime().UnixNano()
	}
	encoded, _ := json.Marshal([]any{path, modified, opts.Runner, gpu, parallel, config, environment})
	return sha256.Sum256(encoded)
}

func ObservedServerVRAM(path string, model *gguf.Model, opts api.Options, devices []ml.DeviceInfo, parallel int, config LlamaServerConfig) (uint64, bool) {
	key := placementKey(path, model, opts, devices, parallel, config)
	placements.Lock()
	defer placements.Unlock()
	entry, ok := placements.entries[key]
	if ok {
		placements.clock++
		entry.used = placements.clock
		placements.entries[key] = entry
	}
	return entry.bytes, ok
}

func (s *llamaServerRunner) rememberPlacement() {
	if s.launch.numParallel != 1 || s.launch.modelExperts != 0 || len(s.launch.projectors) != 0 || s.launch.draftType != "" || s.launch.forceNoMMProjOffload {
		return
	}
	s.memoryMu.RLock()
	bytes := s.memGPU
	full := s.totalLayers > 0 && s.gpuLayers >= s.totalLayers && s.gpuLayerOverflow == 0 && s.moeRAMWeightLimit == 0
	s.memoryMu.RUnlock()
	if !full || bytes == 0 || len(s.launch.gpus) == 0 {
		return
	}
	// Keep allocator headroom; uncertain and partial placements use the old estimate.
	bytes += max(uint64(64<<20), bytes/10)
	key := placementKey(s.modelPath, s.metadata, s.launch.opts, s.launch.gpus, s.launch.numParallel, s.launch.config)
	placements.Lock()
	defer placements.Unlock()
	placements.clock++
	placements.entries[key] = placementObservation{bytes: bytes, used: placements.clock}
	if len(placements.entries) > 64 {
		var oldest [32]byte
		age := ^uint64(0)
		for key, entry := range placements.entries {
			if entry.used < age {
				oldest, age = key, entry.used
			}
		}
		delete(placements.entries, oldest)
	}
}
