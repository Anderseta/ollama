#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda-tuning.h"
#include "ggml-cuda/resident-tuning-policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace tuning = wildllama_cuda_tuning;
static volatile std::sig_atomic_t interrupted = 0;
static void interrupt(int) { interrupted = 1; }
static bool abort_calibration(void *) { return interrupted != 0; }
static void require(bool value, const char * text) { if (!value) { throw std::runtime_error(text); } }

struct options {
    std::string backend, fingerprint, output, type = "q8_0";
    int64_t k = 2048, m = 2048, n = 1;
    uint32_t pairs = 5, time_ms = 1500;
    size_t max_mib = 64, scratch_mib = 8, device = 0;
    bool warmup = true;
};

static const char * status_name(ggml_cuda_tuning_status status) {
    switch (status) {
        case GGML_CUDA_TUNING_APPROVED: return "approved";
        case GGML_CUDA_TUNING_UNSUPPORTED: return "unsupported-or-no-warmup";
        case GGML_CUDA_TUNING_INVALID: return "invalid-input";
        case GGML_CUDA_TUNING_CAPTURE: return "capture-or-active-graph";
        case GGML_CUDA_TUNING_BUDGET: return "budget-exhausted";
        case GGML_CUDA_TUNING_CANCELLED: return "cancelled";
        case GGML_CUDA_TUNING_NONFINITE: return "nonfinite-output";
        case GGML_CUDA_TUNING_DIFFERENT: return "outputs-not-bitwise-equal";
        case GGML_CUDA_TUNING_NO_GAIN: return "no-repeatable-gain";
        default: return "cuda-error";
    }
}

static void report(ggml_cuda_tuning_status status, const ggml_cuda_tuning_result & result) {
    std::printf("{\"status\":\"%s\",\"reason\":\"%s\",\"original\":%d,\"selected\":%d,"
        "\"pairs\":%u,\"mmvq_ms\":%.9g,\"mmq_ms\":%.9g,\"minimum_speedup\":%.9g,\"scratch_bytes\":%zu,\"workspace_bytes\":%zu}\n",
        status == GGML_CUDA_TUNING_APPROVED ? "approved" : "fallback", status_name(status),
        int(result.original), int(result.selected), result.pairs, result.mmvq_ms, result.mmq_ms,
        std::isfinite(result.minimum_speedup) ? result.minimum_speedup : 0.0, result.scratch_bytes, result.workspace_bytes);
}

static uint64_t number(const std::string & text) {
    require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos, "numeric option must be a positive integer");
    size_t used;
    const auto value = std::stoull(text, &used);
    require(used == text.size() && value > 0 && value <= uint64_t(INT32_MAX), "numeric option outside supported range");
    return value;
}

static options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::puts("Explicit synthetic CUDA calibration (no model loaded).\n"
                "Required: --backend <absolute ggml-cuda.dll/.so> --fingerprint <64 lowercase hex> --output <absolute NEW cache file>\n"
                "Optional: --type q8_0 --k 2048 --m 2048 --n 1 --pairs 5 --time-ms 1500 --max-mib 64 --scratch-mib 8 --device 0 --no-warmup\n"
                "Three finite deterministic fixtures; bitwise equality and >=5% improvement in every timed pair.\n"
                "--no-warmup reports fallback before loading CUDA. Ctrl+C cancels between launches.\n"
                "--max-mib bounds owned fixture/host/scratch bytes, excluding driver/library/context overhead.\n"
                "Time limits stop new work; cleanup cannot preempt an already submitted CUDA kernel.\n"
                "No policy is published on fallback. Existing output files are never overwritten.\n"
                "Fingerprint must come from the trusted runtime for THIS physical GPU, full driver and exact payload.");
            std::exit(0);
        }
        if (arg == "--no-warmup") { o.warmup = false; continue; }
        require(i + 1 < argc, "option needs a value");
        const std::string value = argv[++i];
        if (arg == "--backend") { o.backend = value; }
        else if (arg == "--fingerprint") { o.fingerprint = value; }
        else if (arg == "--output") { o.output = value; }
        else if (arg == "--type") { o.type = value; }
        else if (arg == "--k") { o.k = int64_t(number(value)); }
        else if (arg == "--m") { o.m = int64_t(number(value)); }
        else if (arg == "--n") { o.n = int64_t(number(value)); }
        else if (arg == "--pairs") { o.pairs = uint32_t(number(value)); }
        else if (arg == "--time-ms") { o.time_ms = uint32_t(number(value)); }
        else if (arg == "--max-mib") { o.max_mib = size_t(number(value)); }
        else if (arg == "--scratch-mib") { o.scratch_mib = size_t(number(value)); }
        else if (arg == "--device") { o.device = value == "0" ? 0 : size_t(number(value)); }
        else { throw std::runtime_error("unknown option: " + arg); }
    }
    require(!o.backend.empty() && fs::u8path(o.backend).is_absolute(), "backend must be an absolute path");
    require(!o.output.empty() && fs::u8path(o.output).is_absolute(), "output must be an absolute path");
    require(tuning::fingerprint_valid(o.fingerprint), "fingerprint must be 64 lowercase hexadecimal characters");
    require(o.k >= 256 && o.k <= 16384 && o.k % 256 == 0 && o.m >= 128 && o.m <= 16384 && o.m % 128 == 0 && o.n <= 8,
        "unsupported shape: k multiple256<=16384, m multiple128<=16384, n1..8");
    require(o.pairs >= 3 && o.pairs <= 16 && o.time_ms <= 10000 && o.max_mib <= 256 && o.scratch_mib <= o.max_mib,
        "limits: pairs3..16, time<=10000ms, scratch<=max<=256MiB");
    return o;
}

static ggml_type quant_type(const std::string & name) {
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                           GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
        if (name == ggml_type_name(type)) { return type; }
    }
    throw std::runtime_error("unsupported quantization type");
}

// Write a private temporary file exclusively, flush it, then publish by hard
// link (atomic and no replacement). NTFS/ext4 support this; unsupported file
// systems return an explicit error and never replace the user's old policy.
static void publish(const fs::path & destination, const std::string & text) {
    require(!fs::exists(destination), "output exists; choose a new versioned cache path");
    require(fs::is_directory(destination.parent_path()), "output parent directory must already exist");
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path temporary = destination;
    temporary += ".tmp-" + std::to_string(stamp);
#ifdef _WIN32
    const int fd = _wopen(temporary.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    const int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
#endif
    require(fd >= 0, "cannot create private temporary policy file");
    bool closed = false;
    try {
        size_t offset = 0;
        while (offset < text.size()) {
#ifdef _WIN32
            const int wrote = _write(fd, text.data() + offset, unsigned(text.size() - offset));
#else
            const auto wrote = write(fd, text.data() + offset, text.size() - offset);
#endif
            require(wrote > 0, "cannot write policy file");
            offset += size_t(wrote);
        }
#ifdef _WIN32
        require(_commit(fd) == 0, "cannot flush policy file");
        const int close_status = _close(fd);
#else
        require(fsync(fd) == 0, "cannot flush policy file");
        const int close_status = close(fd);
#endif
        closed = true;
        require(close_status == 0, "cannot close policy file");
        fs::create_hard_link(temporary, destination);
        fs::remove(temporary);
    } catch (...) {
        if (!closed) {
#ifdef _WIN32
            _close(fd);
#else
            close(fd);
#endif
        }
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
}

int main(int argc, char ** argv) {
    try {
        const auto o = parse(argc, argv);
        if (!o.warmup) { report(GGML_CUDA_TUNING_UNSUPPORTED, {}); return 0; }
        const auto type = quant_type(o.type);
        require(!fs::exists(fs::u8path(o.output)), "output already exists; choose a new cache filename");
        require(fs::is_directory(fs::u8path(o.output).parent_path()), "cache parent directory must exist");
        const size_t max_bytes = o.max_mib * 1024 * 1024, scratch_bytes = o.scratch_mib * 1024 * 1024;
        const size_t weight_floats = size_t(o.k) * size_t(o.m), input_floats = size_t(o.k) * size_t(o.n);
        const size_t quant_bytes = ggml_row_size(type, o.k) * size_t(o.m);
        const size_t fixture_gpu = quant_bytes + (input_floats + size_t(o.m) * size_t(o.n)) * sizeof(float) + 1024;
        const size_t owned_bytes = weight_floats * sizeof(float) + quant_bytes + input_floats * sizeof(float) + fixture_gpu + scratch_bytes;
        require(owned_bytes <= max_bytes, "fixture exceeds --max-mib (host + device fixture + scratch)");
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
        // No registry autodetection or fallback to another backend/vendor.
        const auto reg = ggml_backend_load(o.backend.c_str());
        require(reg && std::string(ggml_backend_reg_name(reg)) == "CUDA", "explicit CUDA backend could not be loaded");
        require(o.device < ggml_backend_reg_dev_count(reg), "device index is unavailable");
        const auto api = static_cast<const ggml_backend_cuda_tuning_api *>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_tuning_api"));
        require(api && api->abi_version == 1, "backend does not support tuning ABI1");
        using set_limit_fn = bool (*)(ggml_backend_dev_t, size_t, bool);
        auto set_limit = reinterpret_cast<set_limit_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_memory_limit"));
        require(set_limit, "backend lacks quota API");
        auto device = ggml_backend_reg_dev_get(reg, o.device);
        require(set_limit(device, max_bytes, true), "cannot register bounded calibration GPU quota");
        struct release_quota {
            set_limit_fn fn; ggml_backend_dev_t device;
            ~release_quota() { fn(device, 0, false); }
        } quota{set_limit, device};
        ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        require(backend != nullptr, "backend initialization failed");
        ggml_context_ptr context(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}));
        require(context != nullptr, "tensor context allocation failed");
        auto weights = ggml_new_tensor_2d(context.get(), type, o.k, o.m);
        auto input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, o.k, o.n);
        auto output = ggml_mul_mat(context.get(), weights, input);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()));
        require(buffer != nullptr, "fixture allocation denied");
        ggml_backend_buffer_clear(buffer.get(), 0);
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> floats(weight_floats), activations(input_floats);
        std::vector<uint8_t> quantized(quant_bytes);
        tuning::entry selected;
        selected.shape = {int(type), o.k, o.m, o.n};
        selected.minimum_speedup = std::numeric_limits<double>::infinity();
        ggml_cuda_tuning_result result{};
        const auto start = std::chrono::steady_clock::now();
        for (unsigned fixture = 0; fixture < 3; ++fixture) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            if (elapsed >= o.time_ms || interrupted) {
                report(interrupted ? GGML_CUDA_TUNING_CANCELLED : GGML_CUDA_TUNING_BUDGET, result);
                return 0;
            }
            for (size_t i = 0; i < floats.size(); ++i) { floats[i] = std::sin(float(i % 104729) * 0.017f + float(fixture) * 0.41f) * (fixture + 1); }
            for (size_t i = 0; i < activations.size(); ++i) { activations[i] = std::cos(float(i) * 0.013f + float(fixture) * 0.37f) / (fixture + 1); }
            require(ggml_quantize_chunk(type, floats.data(), quantized.data(), 0, o.m, o.k, nullptr) == quant_bytes,
                "weight quantization failed");
            if (interrupted || std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(o.time_ms)) {
                report(interrupted ? GGML_CUDA_TUNING_CANCELLED : GGML_CUDA_TUNING_BUDGET, result); return 0;
            }
            ggml_backend_tensor_set(weights, quantized.data(), 0, quant_bytes);
            ggml_backend_tensor_set(input, activations.data(), 0, activations.size() * sizeof(float));
            const auto used = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            if (used >= o.time_ms) { report(GGML_CUDA_TUNING_BUDGET, result); return 0; }
            ggml_cuda_tuning_limits limits{1, o.pairs, uint32_t(o.time_ms - used), scratch_bytes, true, abort_calibration, nullptr};
            const auto status = api->calibrate(backend.get(), output, &limits, &result);
            if (status != GGML_CUDA_TUNING_APPROVED) { report(status, result); return 0; }
            if (fixture && (selected.original != result.original || selected.selected != result.selected)) {
                report(GGML_CUDA_TUNING_NO_GAIN, result); return 0;
            }
            selected.original = result.original;
            selected.selected = result.selected;
            selected.pairs = result.pairs;
            selected.minimum_speedup = std::min(selected.minimum_speedup, result.minimum_speedup);
            selected.workspace_bytes = std::max(selected.workspace_bytes, result.workspace_bytes);
            std::fprintf(stderr, "validated synthetic fixture %u/3 (finite, bitwise, repeatable)\n", fixture + 1);
        }
        tuning::policy policy;
        policy.entries[0] = selected; policy.count = 1;
        std::ostringstream encoded;
        require(tuning::serialize(encoded, o.fingerprint, policy), "cannot serialize approved result");
        tuning::policy checked;
        std::istringstream check(encoded.str());
        require(tuning::parse(check, o.fingerprint, checked), "produced policy failed strict round-trip validation");
        publish(fs::u8path(o.output), encoded.str());
        require(api->configure(backend.get(), o.output.c_str(), o.fingerprint.c_str()), "native consumer rejected published cache");
        ggml_cuda_tuning_stats stats{};
        require(api->get_stats(backend.get(), &stats) && stats.enabled && stats.entries == 1, "native policy was not installed");
        result.minimum_speedup = selected.minimum_speedup;
        report(GGML_CUDA_TUNING_APPROVED, result);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Calibration failed: %s\n", error.what());
        return 1;
    }
}
