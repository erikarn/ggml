// GDN prefill corruption test for Intel Vulkan backends.
//
// This is a standalone test that exercises GGML_OP_GATED_DELTA_NET with
// the exact dimensions used by Qwen3.5 models (head_count=32, head_size=128)
// across a range of n_seq_tokens values. It compares GPU backend results
// against the CPU reference implementation and reports NMSE errors.
//
// The goal is to reproduce issue #21888 at the libggml level:
// https://github.com/ggml-org/llama.cpp/issues/21888
//
// Build: add to tests/CMakeLists.txt as a new target, or compile manually:
//   c++ -std=c++17 -I../include -I.. test-gdn-prefill.cpp \
//       ../src/libggml.so.0.16.0 ../src/libggml-base.so.0.16.0 \
//       -pthread -o test-gdn-prefill
//
// Run from build/bin so dynamic backends are found:
//   cd build/bin && ./test-gdn-prefill [-b <backend>] [-p <n_tokens,...>]
//
// Examples:
//   ./test-gdn-prefill                          # run all default tests
//   ./test-gdn-prefill -b "Vulkan"              # test only Vulkan backend
//   ./test-gdn-prefill -p 5,9,10,16,32          # specific token counts
//   ./test-gdn-prefill -p 5,6,7,8,9,10,11,12    # boundary sweep

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ── Helpers ────────────────────────────────────────────────────────────────

static void init_tensor_uniform(ggml_tensor *t, float min = -1.0f, float max = 1.0f) {
    size_t nels = ggml_nelements(t);
    std::vector<float> data(nels);
    thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(min, max);
    for (size_t i = 0; i < nels; i++) {
        data[i] = dist(gen);
    }
    if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(float));
    } else {
        GGML_ABORT("unsupported tensor type");
    }
}

static double nmse(const float *a, const float *b, size_t n) {
    double mse_ab = 0.0, mse_a0 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse_ab += d * d;
        mse_a0 += (double)a[i] * (double)a[i];
    }
    return mse_a0 > 0 ? mse_ab / mse_a0 : 0.0;
}

// ── Test case struct ───────────────────────────────────────────────────────

struct gdn_prefill_test {
    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;
    const int     v_repeat;
    const bool    kda;
    const int64_t K;

    std::string label() const {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "GDN(head_count=%ld,head_size=%ld,n_tokens=%ld,seqs=%ld,v_rep=%d,kda=%s,K=%ld)",
                 (long)head_count, (long)head_size, (long)n_seq_tokens,
                 (long)n_seqs, v_repeat, kda ? "true" : "false", (long)K);
        return buf;
    }

    gdn_prefill_test(int64_t head_count = 32, int64_t head_size = 128,
                     int64_t n_seq_tokens = 1, int64_t n_seqs = 1,
                     int v_repeat = 1, bool kda = true, int64_t K = 1)
        : head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens),
          n_seqs(n_seqs), v_repeat(v_repeat), kda(kda), K(K) {}

    // Build compute graph and compare GPU vs CPU via ggml_backend_compare_graph_backend.
    // Returns true if test passed.
    bool run(ggml_backend_t gpu, ggml_backend_t cpu_ref, double &out_nmse,
             bool verbose = false) const {
        // Context with no_alloc=true — only metadata, data allocated by backend
        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead(),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        auto ctx = std::unique_ptr<ggml_context, decltype(&free)>(
            (ggml_context *)malloc(params.mem_size), free);
        if (!ctx) {
            fprintf(stderr, "%s: context allocation failed\n", label().c_str());
            return false;
        }

        ggml_context *gc = ggml_init(params);
        if (!gc) {
            fprintf(stderr, "%s: ggml_init failed\n", label().c_str());
            return false;
        }

        // Create tensors — same shapes as test_gated_delta_net in test-backend-ops.cpp
        const int64_t g_ne0 = kda ? head_size : 1;
        const int64_t h_v = head_count * v_repeat;

        ggml_tensor *q  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, head_size, head_count,      n_seq_tokens, n_seqs);
        ggml_tensor *k  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, head_size, head_count,      n_seq_tokens, n_seqs);
        ggml_tensor *v  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, head_size, h_v,             n_seq_tokens, n_seqs);
        ggml_tensor *g  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, g_ne0,     h_v,             n_seq_tokens, n_seqs);
        ggml_tensor *b  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, 1,         h_v,             n_seq_tokens, n_seqs);
        ggml_tensor *st = ggml_new_tensor_4d(gc, GGML_TYPE_F32, head_size, head_size,       h_v,          n_seqs);

        // L2-normalize Q and K (as done in qwen35/kimi-linear)
        q = ggml_l2_norm(gc, q, 1e-6f);
        k = ggml_l2_norm(gc, k, 1e-6f);

        // GDN op
        ggml_tensor *out = ggml_gated_delta_net(gc, q, k, v, g, b, st, K);

        // Build forward graph
        ggml_cgraph *gf = ggml_new_graph(gc);
        ggml_build_forward_expand(gf, out);

        // Allocate tensors on CPU backend first
        ggml_backend_buffer_ptr buf_cpu(ggml_backend_alloc_ctx_tensors(gc, cpu_ref));
        if (!buf_cpu) {
            fprintf(stderr, "%s: failed to allocate CPU tensors\n", label().c_str());
            return false;
        }

        // Initialize leaf input tensors
        for (ggml_tensor *t = ggml_get_first_tensor(gc); t != nullptr; t = ggml_get_next_tensor(gc, t)) {
            if (ggml_is_view(t)) {
                continue;
            }
            init_tensor_uniform(t);
        }

        // Compare GPU vs CPU using ggml_backend_compare_graph_backend
        struct callback_userdata {
            bool   ok;
            double nmse_sum;
            size_t nmse_count;
            bool   has_nan;
            bool   has_inf;
        };

        callback_userdata ud {
            .ok = true,
            .nmse_sum = 0.0,
            .nmse_count = 0,
            .has_nan = false,
            .has_inf = false,
        };

        auto callback = [](int /*index*/, ggml_tensor *t1, ggml_tensor *t2, void *user_data) -> bool {
            callback_userdata *ud = (callback_userdata *)user_data;

            // Skip no-op tensors
            if (t1->op == GGML_OP_NONE) {
                return true;
            }

            size_t nelems = ggml_nelements(t1);
            std::vector<float> f1(nelems), f2(nelems);
            ggml_backend_tensor_get(t1, f1.data(), 0, ggml_nbytes(t1));
            ggml_backend_tensor_get(t2, f2.data(), 0, ggml_nbytes(t2));

            for (size_t i = 0; i < nelems; i++) {
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    ud->has_nan = true;
                    ud->ok = false;
                    return true;
                }
                if (std::isinf(f1[i]) || std::isinf(f2[i])) {
                    ud->has_inf = true;
                    ud->ok = false;
                    return true;
                }
            }

            double err = nmse(f1.data(), f2.data(), nelems);
            ud->nmse_sum += err;
            ud->nmse_count++;

            if (err > 1e-7) {
                ud->ok = false;
            }

            return true;
        };

        bool compare_ok = ggml_backend_compare_graph_backend(
            cpu_ref, gpu, gf, callback, &ud, nullptr, 0);

        out_nmse = ud.nmse_count > 0 ? ud.nmse_sum / ud.nmse_count : 0.0;

        if (verbose) {
            printf("%s\n", label().c_str());
            printf("  NMSE: %.9e  (tolerance: 1e-7)\n", out_nmse);
            if (ud.has_nan)  printf("  WARNING: NaN detected\n");
            if (ud.has_inf)  printf("  WARNING: Inf detected\n");
            printf("  %s\n", compare_ok && ud.ok ? "[PASS]" : "[FAIL]");
        }

        return compare_ok && ud.ok;
    }
};

// ── Main ───────────────────────────────────────────────────────────────────

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  -b <backend>   Backend name (default: all available)\n"
        "  -p <list>      Comma-separated n_seq_tokens values (default: all)\n"
        "  -h             Show this help\n"
        "\n"
        "Examples:\n"
        "  %s                              # run all tests on all backends\n"
        "  %s -b \"Vulkan\"                  # test only Vulkan\n"
        "  %s -p 5,9,10,16,32              # specific token counts\n"
        "  %s -p 5,6,7,8,9,10,11,12        # boundary sweep\n"
        "\n", prog, prog, prog, prog, prog);
}

int main(int argc, char **argv) {
    const char *backend_filter = nullptr;
    std::vector<int64_t> token_counts; // empty = use defaults

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            backend_filter = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            const char *tokstr = argv[++i];
            const char *start = tokstr;
            while (*start) {
                char *end = nullptr;
                long val = strtol(start, &end, 10);
                if (end > start) {
                    token_counts.push_back(val);
                    start = end;
                }
                if (*start == ',') start++;
                else break;
            }
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    // Load backends
    ggml_backend_load_all();

    // Collect backends to test
    std::vector<std::pair<ggml_backend_dev_t, ggml_backend_t>> backends;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char *name = ggml_backend_dev_name(dev);
        if (backend_filter && strcmp(name, backend_filter) != 0) {
            continue;
        }
        // Only test accelerators (not CPU for comparison)
        enum ggml_backend_dev_type btype = ggml_backend_dev_type(dev);
        if (btype == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend) {
            backends.emplace_back(dev, backend);
        }
    }

    if (backends.empty()) {
        fprintf(stderr, "No suitable GPU backends found.\n");
        return 1;
    }

    // CPU reference backend
    ggml_backend_dev_t cpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_dev = dev;
            break;
        }
    }
    if (!cpu_dev) {
        fprintf(stderr, "CPU backend not found.\n");
        return 1;
    }
    ggml_backend_t cpu_ref = ggml_backend_dev_init(cpu_dev, nullptr);
    using set_use_ref_fn = void (*)(ggml_backend_t, bool);
    auto *reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto set_use_ref = (set_use_ref_fn)ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_cpu_set_use_ref");
    if (set_use_ref) set_use_ref(cpu_ref, true);

    // Define test parameters
    // Default: full sweep matching GDN_PREFILL_CORRUPTION_TEST.md
    std::vector<gdn_prefill_test> tests;

    if (token_counts.empty()) {
        // Full sweep: working length, failing lengths, boundaries
        std::vector<int64_t> tokens = {
            1, 2, 3, 4, 5,   // short sequences (working)
            6, 7, 8, 9, 10,  // transition zone
            11, 12, 13, 14, 15, 16,  // power-of-2 boundary
            32, 64, 127, 128, 200, 256  // larger contexts
        };
        for (int64_t t : tokens) {
            // Primary: Qwen3.5 dimensions, KDA mode
            tests.emplace_back(32, 128, t, 1, 1, true, 1);
            // Scalar gate variant
            tests.emplace_back(32, 128, t, 1, 1, false, 1);
        }
        // K > 1 variants at key token counts
        for (int64_t t : {5, 9, 16}) {
            tests.emplace_back(32, 128, t, 1, 1, true, 2);
            tests.emplace_back(32, 128, t, 1, 1, true, 4);
            tests.emplace_back(32, 128, t, 1, 1, true, 8);
        }
        // Overflow cases (n_tokens > K)
        tests.emplace_back(32, 128, 16, 1, 1, true, 4);
        tests.emplace_back(32, 128, 32, 1, 1, true, 4);
    } else {
        for (int64_t t : token_counts) {
            tests.emplace_back(32, 128, t, 1, 1, true, 1);
            tests.emplace_back(32, 128, t, 1, 1, false, 1);
        }
    }

    // Print header
    printf("=============================================================\n");
    printf("GDN Prefill Corruption Test (libggml only)\n");
    printf("Issue: #21888 — Qwen3.5 GDN chunked prefill on Intel iGPU\n");
    printf("=============================================================\n\n");

    printf("Backends:\n");
    for (auto &[dev, bk] : backends) {
        printf("  - %s (%s)\n", ggml_backend_dev_name(dev),
               ggml_backend_dev_description(dev));
    }
    printf("  - CPU (reference)\n\n");

    printf("Tests: %zu configurations × %zu backends = %zu total\n\n",
           tests.size(), backends.size(), tests.size() * backends.size());

    // Run tests sequentially per backend (backends are not thread-safe for
    // concurrent graph compute; see ggml-backend-compare API docs)
    int total_pass = 0, total_fail = 0;

    for (auto &[gpu_dev, gpu_backend] : backends) {
        printf("Testing backend: %s\n", ggml_backend_dev_name(gpu_dev));
        printf("-----------------------------------------------\n");

        int n_pass = 0, n_fail = 0;
        for (const auto &test : tests) {
            double nmse_val = 0.0;
            bool pass = test.run(gpu_backend, cpu_ref, nmse_val, false);
            printf("%-70s ", test.label().c_str());
            if (pass) {
                printf("[PASS]  NMSE=%.2e\n", nmse_val);
                n_pass++;
            } else {
                printf("[FAIL]  NMSE=%.2e\n", nmse_val);
                n_fail++;
            }
        }
        printf("\n  Results: %d passed, %d failed\n\n", n_pass, n_fail);
        total_pass += n_pass;
        total_fail += n_fail;
    }

    // Summary
    printf("=============================================================\n");
    printf("SUMMARY: %d passed, %d failed, %zu total tests\n",
           total_pass, total_fail, (size_t)(total_pass + total_fail));
    printf("=============================================================\n");

    // Cleanup
    for (auto &[dev, bk] : backends) {
        ggml_backend_free(bk);
    }
    ggml_backend_free(cpu_ref);

    return total_fail > 0 ? 1 : 0;
}
