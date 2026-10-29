#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

int main(int argc, char ** argv) {
    try {
        const bool disabled = argc == 2 && std::strcmp(argv[1], "--no-graphs") == 0;
        require(argc == 1 || disabled, "unknown fixture argument");
#ifdef _WIN32
        _putenv_s("GGML_CUDA_DISABLE_GRAPHS", disabled ? "1" : "");
#else
        if (disabled) { setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1); }
        else { unsetenv("GGML_CUDA_DISABLE_GRAPHS"); }
#endif
        ggml_backend_load_all();
        auto * registry = ggml_backend_reg_by_name("CUDA");
        if (!registry || !ggml_backend_reg_dev_count(registry)) {
            std::fprintf(stderr, "SKIP: no available CUDA device\n");
            return 77;
        }
        auto * device = ggml_backend_reg_dev_get(registry, 0);
        const auto set_limit = reinterpret_cast<decltype(&ggml_backend_cuda_set_memory_limit)>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_cuda_set_memory_limit"));
        const auto get_stats = reinterpret_cast<decltype(&ggml_backend_cuda_get_memory_stats)>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_cuda_get_memory_stats"));
        require(set_limit && get_stats, "CUDA allocation quota unavailable");
        constexpr uint64_t limit = 64*1024*1024;
        require(set_limit(device, limit, true), "cannot set bounded fixture quota");
        struct owner {
            decltype(set_limit) release;
            ggml_backend_dev_t device;
            ~owner() { release(device, 0, false); }
        } quota{set_limit, device};

        ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        require(backend != nullptr, "cannot initialize CUDA backend");
        ggml_context_ptr context(ggml_init({16*ggml_tensor_overhead()+ggml_graph_overhead(), nullptr, true}));
        require(context != nullptr, "cannot allocate graph metadata");
        constexpr int width = 32, rows = 4;
        auto * a = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, width);
        auto * b = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, rows);
        auto * bias = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, rows);
        auto * product = ggml_mul_mat(context.get(), a, b);
        auto * scaled = ggml_scale(context.get(), product, 0.5f);
        auto * output = ggml_add(context.get(), scaled, bias);
        auto * graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()));
        require(buffer != nullptr, "cannot allocate tiny CUDA tensors");
        std::vector<float> av(width*width), bv(width*rows), biasv(width*rows), actual(width*rows), expected(width*rows);
        for (int j=0; j<rows; ++j) {
            for (int k=0; k<width; ++k) {
                bv[j*width+k] = float((j+k)%5-2)*0.25f;
                biasv[j*width+k] = float(j-k)*0.125f;
            }
        }
        ggml_backend_tensor_set(b, bv.data(), 0, ggml_nbytes(b));
        ggml_backend_tensor_set(bias, biasv.data(), 0, ggml_nbytes(bias));
        auto run = [&](float scale, int phase) {
            for (int iteration=0; iteration<8; ++iteration) {
                for (int j=0; j<width; ++j) {
                    for (int k=0; k<width; ++k) { av[j*width+k] = float((j+2*k+iteration+phase)%7-3)*0.125f; }
                }
                for (int j=0; j<rows; ++j) {
                    for (int i=0; i<width; ++i) {
                        float sum = 0;
                        for (int k=0; k<width; ++k) { sum += av[i*width+k]*bv[j*width+k]; }
                        expected[j*width+i] = sum*scale+biasv[j*width+i];
                    }
                }
                ggml_backend_tensor_set(a, av.data(), 0, ggml_nbytes(a));
                require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "CUDA graph execution failed");
                ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
                require(std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(float)) == 0,
                        "capture/replay changed exact fixture outputs");
            }
        };
        graph->uid = UINT64_C(0x57544c4c00000001);
        run(0.5f, 0); // warmup, capture, replay with changed input contents
        ggml_set_op_params_f32(scaled, 0, 0.25f);
        graph->uid = UINT64_C(0x57544c4c00000002); // metadata change invalidates the identity
        run(0.25f, 1); // same node addresses, changed metadata and UID
        graph->uid = 0;
        run(0.25f, 2); // uncached identity follows the full metadata path

        ggml_backend_cuda_memory_stats stats{};
        require(get_stats(device, &stats) && stats.managed_peak_bytes <= limit,
                "fixture exceeded its managed CUDA quota");
        std::printf("PASS: 24 exact CUDA graph executions, changing inputs/new UID/UID0, "
                    "disabled=%d, managed_peak=%llu, quota=%llu\n", int(disabled),
                    (unsigned long long)stats.managed_peak_bytes, (unsigned long long)limit);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
