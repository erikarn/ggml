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

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
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
        // Fallback for non-float types (should not be hit in this test)
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

    // Build compute graph and run on two backends, compare results.
    // Returns true if test passed.
    bool run(ggml_backend_t gpu, ggml_backend_t cpu_ref, double &out_nmse,
             bool verbose = false) const {
        // Allocate contexts
        size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
        auto ctx_cpu = std::unique_ptr<ggml_context, decltype(&free)>(
            (ggml_context *)malloc(ctx_size), free);
        auto ctx_gpu = std::unique_ptr<ggml_context, decltype(&free)>(
            (ggml_context *)malloc(ctx_size), free);
        if (!ctx_cpu || !ctx_gpu) {
            fprintf(stderr, "%s: context allocation failed\n", label().c_str());
            return false;
        }

        ggml_init_params cp = { .mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = false };
        // We use malloc'd memory, so init manually
        // Actually let's use ggml_init with the buffer
        ggml_context *gc = ctx_cpu.get();
        ggml_context *gg = ctx_gpu.get();

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

        // Initialize inputs
        init_tensor_uniform(q, -1.0f, 1.0f);
        init_tensor_uniform(k, -1.0f, 1.0f);
        init_tensor_uniform(v, -0.3f, 5.0f);
        init_tensor_uniform(g, -20.0f, -1e-4f);
        init_tensor_uniform(b, 0.0f, 1.0f);
        init_tensor_uniform(st, -1.0f, 1.0f);

        // Copy graph for GPU backend (same operations, different device placement)
        ggml_tensor *q_g  = ggml_dup_tensor(gg, q);
        ggml_tensor *k_g  = ggml_dup_tensor(gg, k);
        ggml_tensor *v_g  = ggml_dup_tensor(gg, v);
        ggml_tensor *g_g  = ggml_dup_tensor(gg, g);
        ggml_tensor *b_g  = ggml_dup_tensor(gg, b);
        ggml_tensor *st_g = ggml_dup_tensor(gg, st);

        // Need to rebuild the graph on GPU context with same structure
        // Simpler approach: copy tensors then rebuild
        ggml_backend_tensor_copy(q, q_g);
        ggml_backend_tensor_copy(k, k_g);
        ggml_backend_tensor_copy(v, v_g);
        ggml_backend_tensor_copy(g, g_g);
        ggml_backend_tensor_copy(b, b_g);
        ggml_backend_tensor_copy(st, st_g);

        // Rebuild graph on GPU context
        q_g = ggml_l2_norm(gg, q_g, 1e-6f);
        k_g = ggml_l2_norm(gg, k_g, 1e-6f);
        ggml_tensor *out_g = ggml_gated_delta_net(gg, q_g, k_g, v_g, g_g, b_g, st_g, K);

        // Build forward graphs
        ggml_cgraph *gf = ggml_new_graph(gc);
        ggml_build_forward_expand(gf, out);

        ggml_cgraph *gf_gpu = ggml_new_graph(gg);
        ggml_build_forward_expand(gf_gpu, out_g);

        // Execute on CPU reference
        ggml_backend_graph_compute(cpu_ref, gf);

        // Execute on GPU backend
        ggml_backend_graph_compute(gpu, gf_gpu);

        // Read results
        size_t out_nels = ggml_nelements(out);
        std::vector<float> result_cpu(out_nels);
        std::vector<float> result_gpu(out_nels);
        ggml_backend_tensor_get(out, result_cpu.data(), 0, out_nels * sizeof(float));
        ggml_backend_tensor_get(out_g, result_gpu.data(), 0, out_nels * sizeof(float));

        // Compute NMSE
        double err = nmse(result_gpu.data(), result_cpu.data(), out_nels);
        out_nmse = err;

        // Check for NaN/Inf
        bool has_nan = false, has_inf = false;
        for (size_t i = 0; i < out_nels; i++) {
            if (std::isnan(result_gpu[i])) { has_nan = true; break; }
            if (std::isinf(result_gpu[i])) { has_inf = true; break; }
        }

        if (verbose) {
            printf("%s\n", label().c_str());
            printf("  output elements: %zu\n", out_nels);
            printf("  NMSE: %.9e  (tolerance: 1e-7)\n", err);
            if (has_nan)  printf("  WARNING: NaN detected in GPU output\n");
            if (has_inf)  printf("  WARNING: Inf detected in GPU output\n");
            printf("  %s\n", err <= 1e-7 ? "[PASS]" : "[FAIL]");
        }

        return err <= 1e-7 && !has_nan;
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

    // Run tests
    int total_pass = 0, total_fail = 0, total_skip = 0;
    std::mutex mtx;
    std::vector<std::tuple<std::string, double, bool>> results;

    for (auto &[gpu_dev, gpu_backend] : backends) {
        printf("Testing backend: %s\n", ggml_backend_dev_name(gpu_dev));
        printf("-----------------------------------------------\n");

        std::atomic<size_t> next_test(0);
        std::vector<std::thread> workers;
        const int n_workers = std::min((int)std::thread::hardware_concurrency(), 4);

        auto run_worker = [&](ggml_backend_t gpu, ggml_backend_dev_t gpu_dev) {
            while (true) {
                size_t idx = next_test.fetch_add(1);
                if (idx >= tests.size()) break;

                auto &test = tests[idx];
                double nmse_val = 0.0;
                bool pass = test.run(gpu, cpu_ref, nmse_val, false);

                std::lock_guard<std::mutex> lock(mtx);
                results.emplace_back(test.label(), nmse_val, pass);
            }
        };

        for (int w = 0; w < n_workers; w++) {
            workers.emplace_back(run_worker, gpu_backend, gpu_dev);
        }
        for (auto &th : workers) th.join();

        // Sort results for this backend by label
        std::sort(results.begin(), results.end(),
                  [](const auto &a, const auto &b) {
                      return std::get<0>(a) < std::get<0>(b);
                  });

        // Print results
        int n_pass = 0, n_fail = 0;
        for (auto &[label, nmse, pass] : results) {
            printf("%-70s ", label.c_str());
            if (pass) {
                printf("[PASS]  NMSE=%.2e\n", nmse);
                n_pass++;
            } else {
                printf("[FAIL]  NMSE=%.2e\n", nmse);
                n_fail++;
            }
        }
        printf("\n  Results: %d passed, %d failed\n\n", n_pass, n_fail);
        total_pass += n_pass;
        total_fail += n_fail;

        // Clear results for next backend
        results.clear();
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
