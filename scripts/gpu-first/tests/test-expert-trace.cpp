// Build as a standalone C++17 executable with native/src on the include path.
#include "llama-expert-trace.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) { return 2; }
    const std::string path = argv[1];
    if (std::ifstream(path).good()) { return 3; }
#ifdef _WIN32
    _putenv_s("WILDLLAMA_EXPERT_TRACE", path.c_str());
    _putenv_s("WILDLLAMA_EXPERT_TRACE_MAX_BYTES", "4096");
#else
    setenv("WILDLLAMA_EXPERT_TRACE", path.c_str(), 1);
    setenv("WILDLLAMA_EXPERT_TRACE_MAX_BYTES", "4096", 1);
#endif
    const int32_t ids[] = { 1, 3, 1, 7 };
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&] {
            for (uint64_t evaluation = 0; evaluation < 100; ++evaluation) {
                llama_expert_trace::record_route(evaluation, 2, ids, 4, 2, 2, 12345, 9);
            }
        });
    }
    for (auto & thread : threads) { thread.join(); }
    llama_expert_trace::finish();
    std::ifstream stream(path, std::ios::binary);
    const std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    auto check = [](bool condition, const char * message) {
        if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
    };
    check(!content.empty() && content.size() <= 4096, "trace exceeded its byte cap or was not created");
    check(content.find("\"kind\":\"header\"") != std::string::npos, "missing trace header");
    check(content.find("\"kind\":\"end\"") != std::string::npos, "missing trace end marker");
    check(content.find("\"truncated\":true") != std::string::npos, "byte cap was not marked truncated");
    check(content.find("\"model\":9") != std::string::npos, "missing model identity");
    check(content.find("\"experts\":[1,3,1,7]") != std::string::npos, "expert identities were changed");
    check(content.find(path) == std::string::npos, "trace exposed its output path");
    check(content.back() == '\n', "trace ended with a partial record");
    llama_expert_trace::recorder duplicate;
    duplicate.route(999, 2, ids, 4, 2, 2, 12345, 9);
    duplicate.finish();
    std::ifstream preserved(path, std::ios::binary);
    const std::string after((std::istreambuf_iterator<char>(preserved)), std::istreambuf_iterator<char>());
    check(after == content, "exclusive trace creation overwrote an earlier observation");

    const std::string complete_path = path + ".complete";
#ifdef _WIN32
    _putenv_s("WILDLLAMA_EXPERT_TRACE", complete_path.c_str());
#else
    setenv("WILDLLAMA_EXPERT_TRACE", complete_path.c_str(), 1);
#endif
    llama_expert_trace::recorder complete;
    complete.route(1, 2, ids, 4, 2, 2, 12345, 9);
    complete.finish();
    std::ifstream complete_stream(complete_path, std::ios::binary);
    const std::string complete_content((std::istreambuf_iterator<char>(complete_stream)), std::istreambuf_iterator<char>());
    check(complete_content.find("\"truncated\":false") != std::string::npos, "complete trace was reported truncated");
    check(complete_content.find("\"routes\":1") != std::string::npos, "complete trace lost its route");
    return 0;
}
