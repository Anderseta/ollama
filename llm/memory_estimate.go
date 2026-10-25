package llm

import (
	"math"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/gguf"
)

// ServerVRAMEstimate is a conservative full-offload plan, not a measurement of
// the CUDA allocator. Graph/allocator uncertainty is explicit; larger physical
// batches retain their separate LlamaServerBatchMemory allowance.
type ServerVRAMEstimate struct {
	Weights     uint64
	KV          uint64
	WorstCaseKV uint64
	InitialKV   uint64
	Recurrent   uint64
	Auxiliary   uint64
	Valid       bool
}

func (e ServerVRAMEstimate) Total() uint64 {
	if !e.Valid {
		return math.MaxUint64
	}
	return vramAdd(e.Weights, e.KV, e.Recurrent, e.Auxiliary)
}

// PredictServerVRAM retains the legacy entry point. Scheduling should pass the
// real runner options and parallel count to EstimateServerVRAM instead.
func PredictServerVRAM(modelPath string, f *gguf.Model, numCtx int) uint64 {
	opts := api.DefaultOptions()
	opts.NumCtx = numCtx
	return EstimateServerVRAM(modelPath, f, opts, 1).Total()
}

// EstimateServerVRAM respects per-layer GQA, explicit K/V head dimensions and
// the native launch precedence of common versus separate K/V cache formats.
// It never reduces the requested context. A zero context means the model's
// training context, as with native -c 0; invalid/unknown sizes cannot fit.
func EstimateServerVRAM(modelPath string, f *gguf.Model, opts api.Options, parallel int) ServerVRAMEstimate {
	e := ServerVRAMEstimate{Weights: modelFileSize(modelPath, f)}
	if f == nil || opts.NumCtx < 0 || parallel < 1 {
		return e
	}
	kv := f.KV()
	context := uint64(opts.NumCtx)
	if trained := kv.ContextLength(); context == 0 || (trained > 0 && context > trained) {
		context = trained
	}
	layers := kv.BlockCount()
	// The native model has bounded layer tables. Do not loop over malicious or
	// unsupported metadata, and never turn an arithmetic overflow into a fit.
	if context == 0 || layers == 0 || layers > 4096 {
		return e
	}
	initial, err := api.ResolveKVCacheInitialTokens(opts)
	if err != nil {
		return e
	}
	initialContext := context
	if initial > 0 {
		initialContext = min(initialContext, uint64(initial))
	}
	initialContext = vramMul(vramRoundUp(initialContext, 256), uint64(parallel))
	context = vramMul(vramRoundUp(context, 256), uint64(parallel))
	if context == math.MaxUint64 {
		return e
	}
	kindK, kindV := resolvedKVCacheTypes(opts)
	var recurrent uint64
	for layer := uint64(0); layer < layers; layer++ {
		recr, ok := estimatedRecurrentLayer(kv, layer, layers)
		if !ok {
			return e
		}
		if recr {
			recurrent++
			continue
		}
		// Noncausal pooling reads the complete input rather than retaining KV.
		if kv.Has("pooling_type") && !kv.Bool("attention.causal", true) {
			continue
		}
		heads, ok := estimateLayerUint(kv, "attention.head_count", layer, layers, 0)
		if !ok {
			return e
		}
		if heads == 0 && !kv.Has("attention.head_count") {
			return e
		}
		kvHeads, ok := estimateLayerUint(kv, "attention.head_count_kv", layer, layers, heads)
		if !ok {
			return e
		}
		if kvHeads == 0 {
			continue
		}
		if heads == 0 {
			return e
		}
		dim := kv.EmbeddingLength() / heads
		keyDim, okK := estimateLayerUint(kv, "attention.key_length", layer, layers, dim)
		valueDim, okV := estimateLayerUint(kv, "attention.value_length", layer, layers, dim)
		if !okK || !okV || keyDim == 0 || valueDim == 0 ||
			(!kv.Has("attention.key_length") && kv.EmbeddingLength()%heads != 0) ||
			(!kv.Has("attention.value_length") && kv.EmbeddingLength()%heads != 0) {
			return e
		}
		row := vramAdd(kvRowBytes(vramMul(kvHeads, keyDim), kindK), kvRowBytes(vramMul(kvHeads, valueDim), kindV))
		e.WorstCaseKV = vramAdd(e.WorstCaseKV, vramMul(row, context))
		e.InitialKV = vramAdd(e.InitialKV, vramMul(row, initialContext))
	}
	if recurrent > 0 {
		var ok bool
		e.Recurrent, ok = estimateRecurrentBytes(kv, recurrent, uint64(parallel), opts)
		if !ok {
			return e
		}
	}
	// InitialKV is only a candidate commitment for eligible native CUDA VMM
	// buffers. Discovery does not prove all eligibility conditions (VMM, SWA,
	// transposed V, offload and stream count), so Go admission stays nominal.
	e.KV = e.WorstCaseKV
	// This covers unmodeled output, graph and allocator buffers. It is not KV
	// precision or a reduction in the requested context. Native fit remains the
	// authority; exact observed placements can replace this conservative plan.
	e.Auxiliary = max(uint64(256<<20), e.Weights/20)
	e.Valid = vramAdd(e.Weights, e.KV, e.Recurrent, e.Auxiliary) != math.MaxUint64
	return e
}

func resolvedKVCacheTypes(opts api.Options) (string, string) {
	// A nonempty OLLAMA_KV_CACHE_TYPE becomes explicit CLI flags and therefore
	// overrides LLAMA_ARG_CACHE_TYPE_K/V. A present empty runner override clears
	// inheritance, matching NewLlamaServerRunner's environment construction.
	if common := gpuNativeEnvironment(opts, "OLLAMA_KV_CACHE_TYPE"); common != "" {
		common = strings.ToLower(common)
		return common, common
	}
	k, v := gpuNativeEnvironment(opts, "LLAMA_ARG_CACHE_TYPE_K"), gpuNativeEnvironment(opts, "LLAMA_ARG_CACHE_TYPE_V")
	if k == "" {
		k = "f16"
	}
	if v == "" {
		v = "f16"
	}
	return strings.ToLower(k), strings.ToLower(v)
}

func kvRowBytes(values uint64, kind string) uint64 {
	if values == math.MaxUint64 {
		return math.MaxUint64
	}
	var block, size uint64 = 1, 2
	switch kind {
	case "f16", "bf16", "":
	case "f32":
		size = 4
	case "q8_0":
		block, size = 32, 34
	case "q4_0", "iq4_nl":
		block, size = 32, 18
	case "q4_1":
		block, size = 32, 20
	case "q5_0":
		block, size = 32, 22
	case "q5_1":
		block, size = 32, 24
	default:
		return math.MaxUint64
	}
	values = vramRoundUp(values, block)
	if values == math.MaxUint64 {
		return math.MaxUint64
	}
	return vramMul(values/block, size)
}

func estimateLayerUint(kv *gguf.Metadata, key string, layer, layers, fallback uint64) (uint64, bool) {
	v := kv.KeyValue(key)
	if !v.Valid() {
		return fallback, true
	}
	if n, ok := kv.UintOK(key); ok {
		return n, true
	}
	if values := v.Uints(); uint64(len(values)) == layers {
		return values[layer], true
	}
	if values := v.Ints(); uint64(len(values)) == layers && values[layer] >= 0 {
		return uint64(values[layer]), true
	}
	return 0, false
}

func estimatedRecurrentLayer(kv *gguf.Metadata, layer, layers uint64) (bool, bool) {
	if v := kv.KeyValue("attention.recurrent_layers"); v.Valid() {
		if scalar, ok := v.Any().(bool); ok {
			return scalar, true
		}
		if values := v.Bools(); uint64(len(values)) == layers {
			return values[layer], true
		}
		if value, ok := estimateLayerUint(kv, "attention.recurrent_layers", layer, layers, 0); ok && value <= 1 {
			return value != 0, true
		}
		return false, false
	}
	switch kv.Architecture() {
	case "qwen35", "qwen35moe", "qwen3next":
		interval := kv.Uint("full_attention_interval", 4)
		if interval == 0 {
			return false, false
		}
		return (layer+1)%interval != 0, true
	case "mamba", "mamba2":
		return true, true
	default:
		return false, true
	}
}

func estimateRecurrentBytes(kv *gguf.Metadata, layers, parallel uint64, opts api.Options) (uint64, bool) {
	inner, state := kv.Uint("ssm.inner_size"), kv.Uint("ssm.state_size")
	if inner == 0 || state == 0 {
		return 0, false
	}
	conv := kv.Uint("ssm.conv_kernel")
	if conv > 0 {
		conv--
	}
	r := vramMul(conv, vramAdd(inner, vramMul(2, kv.Uint("ssm.group_count"), state)))
	s := vramMul(state, inner)
	rows := parallel
	// Enabled target-model speculation can keep recurrent rollback rows. The
	// native implementation decides the exact span; reserve the requested bound.
	if opts.DraftNumPredict > 0 && opts.SpeculativeType != "none" &&
		((opts.SpeculativeType != "" && opts.SpeculativeType != "auto") || hasMTPDraftMetadata(kv)) {
		rows = vramMul(rows, vramAdd(1, uint64(opts.DraftNumPredict)))
	}
	bytes := vramMul(vramAdd(r, s), 4, rows, layers)
	return bytes, bytes != math.MaxUint64
}

func hasMTPDraftMetadata(kv *gguf.Metadata) bool { return kv.Uint("nextn_predict_layers") > 0 }

func vramAdd(values ...uint64) uint64 {
	var result uint64
	for _, value := range values {
		if value > math.MaxUint64-result {
			return math.MaxUint64
		}
		result += value
	}
	return result
}

func vramMul(values ...uint64) uint64 {
	result := uint64(1)
	for _, value := range values {
		if value == 0 {
			return 0
		}
		if result > math.MaxUint64/value {
			return math.MaxUint64
		}
		result *= value
	}
	return result
}

func vramRoundUp(value, alignment uint64) uint64 {
	if rem := value % alignment; rem != 0 {
		return vramAdd(value, alignment-rem)
	}
	return value
}
