// Narrow GDN NaN diagnostic: compare Vulkan vs CPU on the smallest
// sequence length where they disagree.
//
// From test-gdn-prefill sweep: n_tokens=512, kda=true produces NaN on
// Vulkan but not on CPU. This test isolates that single configuration
// and dumps per-tensor diagnostics to help pinpoint where NaN originates.
//
// Build:
//   c++ -std=c++17 -I../include -I.. test-narrow-gdn.cpp \
//       ../src/libggml.so.0.16.0 ../src/libggml-base.so.0.16.0 \
//       -pthread -o test-narrow-gdn
//
// Run from build/bin:
//   cd build/bin && ./test-narrow-gdn [-b <backend>] [--verbose] [--seed N]
//
// Examples:
//   ./test-narrow-gdn                          # default: all GPU backends vs CPU
//   ./test-narrow-gdn -b "Vulkan"              # only Vulkan
//   ./test-narrow-gdn --verbose                # dump tensor stats on failure
//   ./test-narrow-gdn --seed 42                # fixed seed for reproducibility

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

// ── Helpers ────────────────────────────────────────────────────────────────

static thread_local std::mt19937 g_gen(1);

static void init_tensor_uniform(ggml_tensor *t, float min = -1.0f, float max = 1.0f) {
    size_t nels = ggml_nelements(t);
    std::vector<float> data(nels);
    std::uniform_real_distribution<float> dist(min, max);
    for (size_t i = 0; i < nels; i++) {
        data[i] = dist(g_gen);
    }
    if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(float));
    } else {
        GGML_ABORT("unsupported tensor type");
    }
}

struct tensor_stats {
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    float mean = 0.0f;
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t total = 0;

    void update(const float *data, size_t n) {
        for (size_t i = 0; i < n; i++) {
            float v = data[i];
            if (std::isnan(v)) { nan_count++; continue; }
            if (std::isinf(v)) { inf_count++; continue; }
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            mean += v;
            total++;
        }
        if (total > 0) mean /= total;
    }
};

static double nmse(const float *a, const float *b, size_t n) {
    double mse_ab = 0.0, mse_a0 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse_ab += d * d;
        mse_a0 += (double)a[i] * (double)a[i];
    }
    return mse_a0 > 0 ? mse_ab / mse_a0 : 0.0;
}

// ── Test config ────────────────────────────────────────────────────────────
// First point of divergence between Vulkan and CPU:
//   n_tokens=512, kda=true → Vulkan NaN, CPU clean
static const int64_t TEST_HEAD_COUNT   = 32;
static const int64_t TEST_HEAD_SIZE    = 128;
static const int64_t TEST_N_SEQ_TOKENS = 512;
static const int64_t TEST_N_SEQS       = 1;
static const int     TEST_V_REPEAT     = 1;
static const bool    TEST_KDA          = true;
static const int64_t TEST_K            = 1;

// ── Single run helper ─────────────────────────────────────────────────────

struct run_result {
    bool ok;
    double nmse;
    bool has_nan, has_inf;
    std::string label;
    std::vector<float> output;
    std::vector<tensor_stats> tensor_stats; // per-input-tensor stats
};

static run_result run_once(ggml_backend_t backend, const char *backend_name,
                           uint32_t seed) {
    run_result res;
    res.ok = true;
    res.nmse = 0.0;
    res.has_nan = false;
    res.has_inf = false;

    g_gen.seed(seed);

    ggml_init_params params = {
        /* .mem_size = */ 64 * 1024 * 1024,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    auto ctx = std::unique_ptr<ggml_context, decltype(&free)>(
        (ggml_context *)malloc(params.mem_size), free);
    if (!ctx) {
        res.label = backend_name;
        fprintf(stderr, "%s: context alloc failed\n", backend_name);
        res.ok = false;
        return res;
    }

    ggml_context *gc = ggml_init(params);
    if (!gc) {
        res.label = backend_name;
        fprintf(stderr, "%s: ggml_init failed\n", backend_name);
        res.ok = false;
        return res;
    }

    const int64_t g_ne0 = TEST_KDA ? TEST_HEAD_SIZE : 1;
    const int64_t h_v = TEST_HEAD_COUNT * TEST_V_REPEAT;

    ggml_tensor *q  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, TEST_HEAD_SIZE, TEST_HEAD_COUNT,      TEST_N_SEQ_TOKENS, TEST_N_SEQS);
    ggml_tensor *k  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, TEST_HEAD_SIZE, TEST_HEAD_COUNT,      TEST_N_SEQ_TOKENS, TEST_N_SEQS);
    ggml_tensor *v  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, TEST_HEAD_SIZE, h_v,                 TEST_N_SEQ_TOKENS, TEST_N_SEQS);
    ggml_tensor *g  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, g_ne0,        h_v,                 TEST_N_SEQ_TOKENS, TEST_N_SEQS);
    ggml_tensor *b  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, 1,            h_v,                 TEST_N_SEQ_TOKENS, TEST_N_SEQS);
    ggml_tensor *st = ggml_new_tensor_4d(gc, GGML_TYPE_F32, TEST_HEAD_SIZE, TEST_HEAD_SIZE,      h_v,               TEST_N_SEQS);

    q = ggml_l2_norm(gc, q, 1e-6f);
    k = ggml_l2_norm(gc, k, 1e-6f);
    ggml_tensor *out = ggml_gated_delta_net(gc, q, k, v, g, b, st, TEST_K);

    ggml_cgraph *gf = ggml_new_graph(gc);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_ptr buf(
        ggml_backend_alloc_ctx_tensors(gc, backend));
    if (!buf) {
        res.label = backend_name;
        fprintf(stderr, "%s: allocation failed\n", backend_name);
        res.ok = false;
        return res;
    }

    // Collect input tensor info before init (for naming/stats)
    struct input_info { const char *name; ggml_tensor *ptr; };
    std::vector<input_info> inputs;
    for (ggml_tensor *t = ggml_get_first_tensor(gc); t != nullptr;
         t = ggml_get_next_tensor(gc, t)) {
        if (ggml_is_view(t)) continue;
        if (t->op == GGML_OP_L2_NORM) continue;
        inputs.push_back({t->name, t});
    }

    // Initialize
    for (auto &inp : inputs) {
        init_tensor_uniform(inp.ptr);
    }

    // Execute
    ggml_backend_graph_compute(backend, gf);

    // Read output
    size_t nelems = ggml_nelements(out);
    res.output.resize(nelems);
    ggml_backend_tensor_get(out, res.output.data(), 0, ggml_nbytes(out));

    // Compute per-tensor stats
    for (auto &inp : inputs) {
        size_t nels = ggml_nelements(inp.ptr);
        std::vector<float> tmp(nels);
        ggml_backend_tensor_get(inp.ptr, tmp.data(), 0, ggml_nbytes(inp.ptr));
        tensor_stats s;
        s.update(tmp.data(), nels);
        res.tensor_stats.push_back(s);
    }

    // Check output
    tensor_stats out_stats;
    out_stats.update(res.output.data(), nelems);
    res.has_nan = (out_stats.nan_count > 0);
    res.has_inf = (out_stats.inf_count > 0);
    res.label = backend_name;

    if (res.has_nan || res.has_inf) {
        res.ok = false;
    }

    return res;
}

// ── Main ───────────────────────────────────────────────────────────────────

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  -b <backend>   Backend name (default: all available accelerators)\n"
        "  --verbose      Dump per-tensor statistics on failure\n"
        "  --seed <N>     Fixed RNG seed (default: random)\n"
        "  -h             Show this help\n"
        "\n"
        "Tests: GDN(head_count=32,head_size=128,n_tokens=512,kda=true,K=1)\n"
        "This is the smallest window size where Vulkan produces NaN but CPU does not.\n"
        "\n", prog);
}

int main(int argc, char **argv) {
    const char *backend_filter = nullptr;
    bool verbose = false;
    uint32_t seed = 0; // 0 = random

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            backend_filter = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint32_t)strtol(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (seed == 0) {
        g_gen.seed(std::random_device{}());
    } else {
        g_gen.seed(seed);
    }

    ggml_backend_load_all();

    // Collect GPU backends
    std::vector<std::pair<ggml_backend_dev_t, ggml_backend_t>> gpu_backends;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char *name = ggml_backend_dev_name(dev);
        if (backend_filter && strcmp(name, backend_filter) != 0) continue;
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
        ggml_backend_t bk = ggml_backend_dev_init(dev, nullptr);
        if (bk) gpu_backends.emplace_back(dev, bk);
    }

    // CPU reference
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

    printf("=============================================================\n");
    printf("Narrow GDN NaN Diagnostic\n");
    printf("Config: head_count=%ld, head_size=%ld, n_tokens=%ld, kda=%s, K=%ld\n",
           (long)TEST_HEAD_COUNT, (long)TEST_HEAD_SIZE, (long)TEST_N_SEQ_TOKENS,
           TEST_KDA ? "true" : "false", (long)TEST_K);
    printf("Seed: %u\n", seed);
    printf("=============================================================\n\n");

    // Run CPU first as baseline
    printf("Baseline: CPU\n");
    run_result cpu_res = run_once(cpu_ref, "CPU", seed);
    printf("  %s  NaN=%d  Inf=%d  min=%.6e  max=%.6e  mean=%.6e\n",
           cpu_res.ok ? "[PASS]" : "[FAIL]",
           cpu_res.has_nan ? 1 : 0, cpu_res.has_inf ? 1 : 0,
           cpu_res.tensor_stats.empty() ? 0.0 : cpu_res.tensor_stats.back().min_val,
           cpu_res.tensor_stats.empty() ? 0.0 : cpu_res.tensor_stats.back().max_val,
           cpu_res.tensor_stats.empty() ? 0.0 : cpu_res.tensor_stats.back().mean);

    if (!cpu_res.ok) {
        printf("\n  WARNING: CPU baseline already has NaN/Inf!\n");
        printf("  The instability exists in both backends at this config.\n\n");
    }

    // Run each GPU backend
    int total_pass = 0, total_fail = 0;
    for (auto &[dev, bk] : gpu_backends) {
        printf("\nTesting: %s (%s)\n", ggml_backend_dev_name(dev),
               ggml_backend_dev_description(dev));

        run_result gpu_res = run_once(bk, ggml_backend_dev_name(dev), seed);

        printf("  %s  NaN=%d  Inf=%d  min=%.6e  max=%.6e  mean=%.6e\n",
               gpu_res.ok ? "[PASS]" : "[FAIL]",
               gpu_res.has_nan ? 1 : 0, gpu_res.has_inf ? 1 : 0,
               gpu_res.tensor_stats.empty() ? 0.0 : gpu_res.tensor_stats.back().min_val,
               gpu_res.tensor_stats.empty() ? 0.0 : gpu_res.tensor_stats.back().max_val,
               gpu_res.tensor_stats.empty() ? 0.0 : gpu_res.tensor_stats.back().mean);

        // Compare GPU vs CPU
        if (!gpu_res.has_nan && !gpu_res.has_inf && !cpu_res.has_nan && !cpu_res.has_inf) {
            double err = nmse(gpu_res.output.data(), cpu_res.output.data(),
                              gpu_res.output.size());
            printf("  NMSE vs CPU: %.2e\n", err);
        }

        if (gpu_res.ok && cpu_res.ok) {
            total_pass++;
        } else {
            total_fail++;
        }

        // Divergence indicator
        bool diverges = (!gpu_res.ok && cpu_res.ok) || (gpu_res.ok && !cpu_res.ok);
        if (diverges) {
            printf("  >>> BACKEND DIVERGENCE detected! <<<\n");
        }

    if (verbose && !gpu_res.tensor_stats.empty() && gpu_res.tensor_stats.size() <= 6) {
            printf("\n  Per-tensor stats:\n");
            size_t idx = 0;
            const char *input_names[] = {"q", "k", "v", "g", "beta", "state"};
            for (auto &s : gpu_res.tensor_stats) {
                printf("    %-8s  elems=%zu  nan=%zu  inf=%zu\n",
                       input_names[idx], s.total, s.nan_count, s.inf_count);
                idx++;
            }
        }
    }

    printf("\n=============================================================\n");
    printf("SUMMARY: %d passed, %d failed\n", total_pass, total_fail);
    printf("=============================================================\n");

    for (auto &[dev, bk] : gpu_backends) {
        ggml_backend_free(bk);
    }
    ggml_backend_free(cpu_ref);

    return total_fail > 0 ? 1 : 0;
}
