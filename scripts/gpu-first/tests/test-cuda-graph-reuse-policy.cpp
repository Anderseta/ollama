#include "ggml-cuda/graph-reuse.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

namespace reuse = ggml_cuda_graph_reuse;
enum class op { dense, custom, needs_sync };
struct fixture {
    uint64_t accepted_uid = 0;
    size_t accepted_nodes = 0;
    size_t custom_scans = 0, compatibility_scans = 0, inspected_nodes = 0, resets = 0;

    reuse::eligibility execute(uint64_t uid, const std::vector<op> & nodes, bool enabled = true) {
        const auto result = reuse::classify(uid, accepted_uid, nodes.size(), accepted_nodes, enabled,
            [&] {
                ++custom_scans;
                for (auto node : nodes) {
                    ++inspected_nodes;
                    if (node == op::custom) { ++resets; return true; }
                }
                return false;
            },
            [&] {
                ++compatibility_scans;
                for (auto node : nodes) {
                    ++inspected_nodes;
                    if (node != op::dense) { return false; }
                }
                return true;
            });
        // This mirrors publication by update_required only after successful
        // eligibility, not a second cache maintained by the policy.
        if (result.compatible) { accepted_uid = uid; accepted_nodes = nodes.size(); }
        return result;
    }
};

int main() {
    try {
        const std::vector<op> dense(4096, op::dense);
        fixture first;
        require(first.execute(7, dense).compatible, "cold graph was rejected");
        const auto first_inspections = first.inspected_nodes;
        require(first.custom_scans == 1 && first.compatibility_scans == 1 &&
                first_inspections == 2*dense.size(), "cold graph did not scan metadata completely");
        for (int i=0; i<256; ++i) {
            const auto result = first.execute(7, dense);
            require(result.compatible && !result.has_custom, "certified graph was not reused");
        }
        require(first.inspected_nodes == first_inspections &&
                first.custom_scans == 1 && first.compatibility_scans == 1,
                "warm UID repeated graph metadata scans");

        fixture other_backend;
        other_backend.execute(7, dense);
        require(other_backend.custom_scans == 1 && other_backend.compatibility_scans == 1,
                "another backend borrowed a graph certificate");

        auto custom = dense; custom[1] = op::custom;
        for (int i=0; i<3; ++i) {
            const auto result = first.execute(8, custom);
            require(result.has_custom && !result.compatible, "custom graph became capturable");
        }
        require(first.accepted_uid == 7 && first.resets == 3,
                "custom graph was certified or stopped resetting streams");
        auto fallback = dense; fallback[2] = op::needs_sync;
        const auto checks = first.compatibility_scans;
        for (int i=0; i<3; ++i) {
            require(!first.execute(9, fallback).compatible, "sync-requiring operation was certified");
        }
        require(first.accepted_uid == 7 && first.compatibility_scans == checks+3,
                "negative compatibility was cached by UID");

        require(first.execute(10, dense).compatible, "new compatible UID was rejected");
        const auto after_new = first.custom_scans;
        require(first.execute(10, dense).compatible && first.custom_scans == after_new,
                "new certificate was not reused");

        fixture zero_uid;
        for (int i=0; i<3; ++i) { require(zero_uid.execute(0, dense).compatible, "UID0 fallback failed"); }
        require(zero_uid.custom_scans == 3 && zero_uid.compatibility_scans == 3,
                "UID0 was treated as certified");

        const auto before_disabled = first.compatibility_scans;
        const auto before_custom = first.custom_scans;
        require(!first.execute(10, dense, false).compatible, "disabled graphs reused a certificate");
        require(first.custom_scans == before_custom+1 && first.compatibility_scans == before_disabled,
                "disabled path changed scanner behavior");

        // Count mismatch is merely a conservative policy fallback. It does not
        // make metadata mutation under a stable UID a supported execution API.
        unsigned scans = 0;
        const auto mismatched = reuse::classify(10, 10, 5, 4, true,
            [&] { ++scans; return false; }, [&] { ++scans; return true; });
        require(mismatched.compatible && scans == 2, "count mismatch trusted an old certificate");
        scans = 0;
        reuse::classify(10, 10, 0, 0, true,
            [&] { ++scans; return false; }, [&] { ++scans; return true; });
        require(scans == 2, "empty graph was treated as certified");

        std::printf("PASS: stable UID metadata inspections 257 x 8192 -> 8192; "
                    "new/zero/disabled/custom/incompatible/backend/count fallbacks retained\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
