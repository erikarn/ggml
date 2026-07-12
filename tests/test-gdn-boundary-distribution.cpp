// Boundary sweep for GDN NaN instability, with per-token distribution output.
//
// Walks a range of n_seq_tokens values and reports per-backend NaN/Inf counts
// at each point. Additionally, outputs the full value distribution (histogram
// buckets, min/max/mean/stddev) so we can see how the output shape degrades
// as instability approaches.
//
// Build:
//   c++ -std=c++17 -I../include -I.. test-gdn-boundary-distribution.cpp \
//       ../src/libggml.so.0.16.0 ../src/libggml-base.so.0.16.0 \
//       -pthread -o test-gdn-boundary-distribution
//
// Run from build/bin:
//   cd build/bin && ./test-gdn-boundary-distribution [-b <backend>] [--seed N] [--range START END STEP]
//
// Examples:
//   ./test-gdn-boundary-distribution                          # default sweep 256..512 step 1
//   ./test-gdn-boundary-distribution --range 480 520 1        # narrow window around 512
//   ./test-gdn-boundary-distribution --range 1 2048 32        # coarse sweep
//   ./test-gdn-boundary-distribution -b "Vulkan"              # only Vulkan backend

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

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

struct run_result {
    bool has_nan, has_inf;
    int nan_count, inf_count;      // number of tensors with NaN/Inf
    int total_tensors;             // output tensor element count
    float out_min, out_max, out_mean;
    float out_stddev;              // stddev of valid elements
    std::vector<int> histogram;    // 10-bucket histogram of valid values
    int graph_tensor_count;        // total tensors in graph
    int graph_op_count;            // non-noop ops in graph
};

static run_result run_once(ggml_backend_t backend, const char *backend_name,
                           int64_t n_tokens, uint32_t seed,
                           int head_count = 32, int head_size = 128,
                           int n_seqs = 1, int v_repeat = 1,
                           bool kda = true, int K = 1) {
    run_result res{};
    g_gen.seed(seed);

    ggml_init_params params = {
        /* .mem_size = */ 64 * 1024 * 1024,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    auto ctx = std::unique_ptr<ggml_context, decltype(&free)>(
        (ggml_context *)malloc(params.mem_size), free);
    if (!ctx) return res;

    ggml_context *gc = ggml_init(params);
    if (!gc) return res;

    const int64_t g_ne0 = kda ? head_size : 1;
    const int64_t h_v = head_count * v_repeat;

    ggml_tensor *q  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, head_size, head_count,      n_tokens, n_seqs);
    ggml_tensor *k  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, head_size, head_count,      n_tokens, n_seqs);
    ggml_tensor *v  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, head_size, h_v,             n_tokens, n_seqs);
    ggml_tensor *g  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, g_ne0,     h_v,             n_tokens, n_seqs);
    ggml_tensor *b  = ggml_new_tensor_4d(gc, GGML_TYPE_F32, 1,         h_v,             n_tokens, n_seqs);
    ggml_tensor *st = ggml_new_tensor_4d(gc, GGML_TYPE_F32, head_size, head_size,       h_v,          n_seqs);

    q = ggml_l2_norm(gc, q, 1e-6f);
    k = ggml_l2_norm(gc, k, 1e-6f);
    ggml_tensor *out = ggml_gated_delta_net(gc, q, k, v, g, b, st, K);

    ggml_cgraph *gf = ggml_new_graph(gc);
    ggml_build_forward_expand(gf, out);

    // Count graph structure
    res.graph_tensor_count = 0;
    res.graph_op_count = 0;
    for (ggml_tensor *t = ggml_get_first_tensor(gc); t != nullptr;
         t = ggml_get_next_tensor(gc, t)) {
        res.graph_tensor_count++;
        if (t->op != GGML_OP_NONE && !ggml_is_view(t)) {
            res.graph_op_count++;
        }
    }

    ggml_backend_buffer_ptr buf(
        ggml_backend_alloc_ctx_tensors(gc, backend));
    if (!buf) return res;

    // Initialize leaf inputs
    for (ggml_tensor *t = ggml_get_first_tensor(gc); t != nullptr;
         t = ggml_get_next_tensor(gc, t)) {
        if (ggml_is_view(t)) continue;
        if (t->op == GGML_OP_L2_NORM) continue;
        init_tensor_uniform(t);
    }

    // Execute
    ggml_backend_graph_compute(backend, gf);

    // Read output and check for NaN/Inf
    size_t nelems = ggml_nelements(out);
    res.total_tensors = (int)nelems;
    std::vector<float> tmp(nelems);
    ggml_backend_tensor_get(out, tmp.data(), 0, ggml_nbytes(out));

    res.nan_count = 0;
    res.inf_count = 0;
    res.out_min = FLT_MAX;
    res.out_max = -FLT_MAX;
    double sum = 0.0;
    double sum_sq = 0.0;
    int valid = 0;
    // 10-bucket histogram over [out_min, out_max] of valid values
    std::vector<int> hist(10, 0);
    for (size_t i = 0; i < nelems; i++) {
        float v = tmp[i];
        if (std::isnan(v)) { res.nan_count++; continue; }
        if (std::isinf(v)) { res.inf_count++; continue; }
        if (v < res.out_min) res.out_min = v;
        if (v > res.out_max) res.out_max = v;
        sum += v;
        sum_sq += v * v;
        valid++;
    }
    res.out_mean = valid > 0 ? (float)(sum / valid) : 0.0f;
    if (valid > 1) {
        double mean_d = sum / valid;
        double var = (sum_sq / valid) - (mean_d * mean_d);
        if (var < 0.0) var = 0.0;
        res.out_stddev = (float)sqrt(var);
    } else {
        res.out_stddev = 0.0f;
    }
    // Build histogram
    if (res.out_max > res.out_min && valid > 0) {
        float bucket_width = (res.out_max - res.out_min) / 10.0f;
        for (size_t i = 0; i < nelems; i++) {
            float v = tmp[i];
            if (std::isnan(v) || std::isinf(v)) continue;
            int b = (int)((v - res.out_min) / bucket_width);
            if (b >= 10) b = 9;
            if (b < 0) b = 0;
            hist[b]++;
        }
    }
    res.histogram = hist;
    res.has_nan = (res.nan_count > 0);
    res.has_inf = (res.inf_count > 0);

    return res;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  -b <backend>       Backend name (default: all accelerators)\n"
        "  --range S E [STEP] Range of n_tokens to sweep (default: 256 512 1)\n"
        "  --seed <N>         Fixed RNG seed\n"
        "  -h                 Show this help\n"
        "\n"
        "Tests GDN(head_count=32,head_size=128,kda=true,K=1) across token counts.\n"
        "Reports NaN/Inf counts and value distribution (min/max/mean/stddev,\n"
        "histogram buckets) per backend per token count to characterize how\n"
        "the output distribution degrades as instability approaches.\n", prog);
}

int main(int argc, char **argv) {
    const char *backend_filter = nullptr;
    int64_t range_start = 256, range_end = 512, range_step = 1;
    uint32_t seed = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            backend_filter = argv[++i];
        } else if (strcmp(argv[i], "--range") == 0 && i + 3 < argc) {
            range_start = strtol(argv[++i], nullptr, 10);
            range_end   = strtol(argv[++i], nullptr, 10);
            range_step  = strtol(argv[++i], nullptr, 10);
            if (range_step <= 0) range_step = 1;
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
    printf("GDN NaN Boundary Sweep — Distribution Analysis\n");
    printf("Config: head_count=32, head_size=128, kda=true, K=1\n");
    printf("Range:  [%ld .. %ld] step=%ld  Seed: %u\n",
           (long)range_start, (long)range_end, (long)range_step, seed);
    printf("=============================================================\n\n");

    // Header
    printf("%-8s  %-16s  %-16s  %-10s  %s\n",
           "n_tok", "GPU", "CPU", "Graph", "Divergence");
    printf("%-8s  %-16s  %-16s  %-10s  %s\n",
           "--------", "----------------", "----------------", "----------", "----------");

    int first_fail_cpu = -1, first_fail_gpu = -1;
    int last_pass_cpu = -1, last_pass_gpu = -1;

    for (int64_t n = range_start; n <= range_end; n += range_step) {
        // CPU
        run_result cpu_res = run_once(cpu_ref, "CPU", n, seed);

        // GPU (use first available)
        run_result gpu_res{};
        if (!gpu_backends.empty()) {
            gpu_res = run_once(gpu_backends[0].second,
                               ggml_backend_dev_name(gpu_backends[0].first),
                               n, seed);
        }

        // Format result strings with distribution
        char cpu_str[128], gpu_str[128], div_str[32] = "";
        snprintf(cpu_str, sizeof(cpu_str),
                 "%-6s nan=%d inf=%d μ=%.4g σ=%.4g",
                 (cpu_res.has_nan || cpu_res.has_inf) ? "FAIL" : "PASS",
                 cpu_res.nan_count, cpu_res.inf_count,
                 cpu_res.out_mean, cpu_res.out_stddev);
        snprintf(gpu_str, sizeof(gpu_str),
                 "%-6s nan=%d inf=%d μ=%.4g σ=%.4g",
                 (gpu_res.has_nan || gpu_res.has_inf) ? "FAIL" : "PASS",
                 gpu_res.nan_count, gpu_res.inf_count,
                 gpu_res.out_mean, gpu_res.out_stddev);
        char graph_str[32];
        snprintf(graph_str, sizeof(graph_str),
                 "%d/%d",
                 cpu_res.graph_tensor_count, cpu_res.graph_op_count);

        bool diverges = ((cpu_res.has_nan || cpu_res.has_inf) !=
                         (gpu_res.has_nan || gpu_res.has_inf));
        if (diverges) {
            snprintf(div_str, sizeof(div_str), ">>> DIVERGE <<<");
        }

        printf("%-8ld  %-16s  %-16s  %-10s  %s\n",
               (long)n, gpu_str, cpu_str, graph_str, div_str);

        // Distribution detail when instability detected
        auto print_dist = [](const char *label, const run_result &r) {
            printf("  [%s] range=[%.6g .. %.6g] hist=", label, r.out_min, r.out_max);
            for (int i = 0; i < 10; i++) {
                printf("%d", r.histogram[i]);
            }
            printf("\n");
        };
        if (cpu_res.has_nan || cpu_res.has_inf || gpu_res.has_nan || gpu_res.has_inf) {
            print_dist("CPU", cpu_res);
            if (!gpu_backends.empty()) {
                print_dist("GPU", gpu_res);
            }
        }

        // Track transition points
        bool cpu_ok = !(cpu_res.has_nan || cpu_res.has_inf);
        bool gpu_ok = !(gpu_res.has_nan || gpu_res.has_inf);

        if (!cpu_ok && first_fail_cpu < 0) first_fail_cpu = (int)n;
        if (cpu_ok) last_pass_cpu = (int)n;
        if (!gpu_ok && first_fail_gpu < 0) first_fail_gpu = (int)n;
        if (gpu_ok) last_pass_gpu = (int)n;
    }

    printf("\n=============================================================\n");
    printf("Transition summary:\n");
    if (first_fail_cpu >= 0)
        printf("  CPU first failure:  n_tokens=%d (last pass: %d)\n",
               first_fail_cpu, last_pass_cpu);
    if (first_fail_gpu >= 0)
        printf("  GPU first failure:  n_tokens=%d (last pass: %d)\n",
               first_fail_gpu, last_pass_gpu);
    printf("=============================================================\n");

    for (auto &[dev, bk] : gpu_backends) {
        ggml_backend_free(bk);
    }
    ggml_backend_free(cpu_ref);

    return 0;
}
