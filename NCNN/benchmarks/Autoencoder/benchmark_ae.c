#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>
#include <c_api.h>

#define N_RUNS   1000
#define N_WARMUP 5

static double time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void compute_stats(double *lat, int n, double *mean, double *stdv,
                           double *mn, double *mx, double *p95, double *p99) {
    double sum = 0.0;
    *mn = lat[0];
    *mx = lat[0];
    for (int i = 0; i < n; i++) {
        sum += lat[i];
        if (lat[i] < *mn) *mn = lat[i];
        if (lat[i] > *mx) *mx = lat[i];
    }
    *mean = sum / n;
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = lat[i] - *mean;
        var += d * d;
    }
    *stdv = sqrt(var / n);
    double sorted[N_RUNS];
    memcpy(sorted, lat, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double);
    *p95 = sorted[(int)(0.95 * n)];
    *p99 = sorted[(int)(0.99 * n)];
}

typedef struct {
    float **X;
    float  *y;
    int     n_samples;
    int     n_features;
} TestData;

static void load_test_data(const char *path, TestData *td) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int n_out;
    fscanf(f, "%d %d %d", &td->n_samples, &td->n_features, &n_out);
    float *flat = malloc(td->n_samples * td->n_features * sizeof(float));
    td->X = malloc(td->n_samples * sizeof(float *));
    td->y = malloc(td->n_samples * sizeof(float));
    for (int i = 0; i < td->n_samples; i++) {
        td->X[i] = flat + i * td->n_features;
        for (int j = 0; j < td->n_features; j++)
            fscanf(f, "%f", &td->X[i][j]);
        fscanf(f, "%f", &td->y[i]);
    }
    fclose(f);
}

static void save_csv(const char *path, double *lat, uint64_t (*pmc)[5], int n, const char *label) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "run,latency_us,cycles,instructions,l1_loads,l1_misses,branch_misses,label\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%d,%.3f,%lu,%lu,%lu,%lu,%lu,%s\n", i + 1, lat[i], (unsigned long)pmc[i][0], (unsigned long)pmc[i][1], (unsigned long)pmc[i][2], (unsigned long)pmc[i][3], (unsigned long)pmc[i][4], label);
    fclose(f);
}

static long perf_event_open_syscall(struct perf_event_attr *hw, pid_t pid,
                                     int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw, pid, cpu, group_fd, flags);
}

typedef struct { int fd[6]; } PerfFds;

static int open_perf_fd(uint32_t type, uint64_t config) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type       = type;
    attr.size       = sizeof(attr);
    attr.config     = config;
    attr.disabled   = 1;
    return (int)perf_event_open_syscall(&attr, 0, -1, -1, 0);
}

static PerfFds perf_open_all(void) {
    PerfFds p;
    /* This SoC (SiFive U74) has no hardware event for total L1D accesses -
     * confirmed against both the mainline Linux sifive/u74 pmu-events
     * tables and the hpc-ulisboa RISC-V-Perf-Events-Unmatched project,
     * which only document cache MISS events for this core. Retired load
     * instruction count (integer + FP, summed in perf_disable_read) is
     * used as an approximate proxy for L1D read accesses instead of a
     * real cache-level counter - confirmed on real hardware that a single
     * raw event bit reads correctly (fd[2]=100075961 for a pure-integer
     * busy loop) but OR-combining two bits into one raw config silently
     * reads 0, so integer and FP loads need two separate counters.
     * l1_misses uses the documented SiFive raw event for real data-cache
     * misses (also fires on MMIO accesses, negligible for these
     * CPU-bound workloads). */
    p.fd[0] = open_perf_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    p.fd[1] = open_perf_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    p.fd[2] = open_perf_fd(PERF_TYPE_RAW, 0x0000200); /* INTEGER_LOAD_RETIRED */
    p.fd[3] = open_perf_fd(PERF_TYPE_RAW, 0x0000202); /* DCACHE_MISS_MMIO_ACCESSES */
    p.fd[4] = open_perf_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
    p.fd[5] = open_perf_fd(PERF_TYPE_RAW, 0x0080000); /* FP_LOAD_RETIRED, summed into l1_loads below */
    return p;
}

static void perf_reset_enable(PerfFds *p) {
    for (int _i = 0; _i < 6; _i++) {
        if (p->fd[_i] >= 0) {
            ioctl(p->fd[_i], PERF_EVENT_IOC_RESET,  0);
            ioctl(p->fd[_i], PERF_EVENT_IOC_ENABLE, 0);
        }
    }
}

static void perf_disable_read(PerfFds *p, uint64_t *out) {
    uint64_t raw[6] = {0};
    for (int _i = 0; _i < 6; _i++) {
        if (p->fd[_i] >= 0) {
            ioctl(p->fd[_i], PERF_EVENT_IOC_DISABLE, 0);
            read(p->fd[_i], &raw[_i], sizeof(uint64_t));
        }
    }
    out[0] = raw[0];
    out[1] = raw[1];
    out[2] = raw[2] + raw[5]; /* l1_loads = integer + FP load retired */
    out[3] = raw[3];
    out[4] = raw[4];
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <size> [f32|int8] [n_warmup]\n", argv[0]);
        return 1;
    }
    const char *size = argv[1];
    const char *prec = "f32";
    int n_warmup = N_WARMUP;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "int8") == 0) prec = "int8";
        else if (strcmp(argv[i], "f32") == 0) prec = "f32";
        else n_warmup = atoi(argv[i]);
    }

    char param_path[256], bin_path[256], thresh_path[256], label[64], csv_path[256];
    snprintf(param_path,  sizeof(param_path),  "model/ae_%s_%s.ncnn.param", size, prec);
    snprintf(bin_path,    sizeof(bin_path),    "model/ae_%s_%s.ncnn.bin",   size, prec);
    snprintf(thresh_path, sizeof(thresh_path), "model/ae_%s_threshold.txt",  size);
    snprintf(label, sizeof(label), "ae_%s_%s", size, prec);
    mkdir("latencies", 0755);
    snprintf(csv_path, sizeof(csv_path), "latencies/latencies_%s.csv", label);

    float threshold;
    FILE *tf = fopen(thresh_path, "r");
    if (!tf) { fprintf(stderr, "Cannot open %s\n", thresh_path); return 1; }
    fscanf(tf, "%f", &threshold); fclose(tf);

    TestData td;
    load_test_data("model/test_data.txt", &td);

    ncnn_net_t net = ncnn_net_create();
    if (ncnn_net_load_param(net, param_path) != 0) {
        fprintf(stderr, "Failed to load %s\n", param_path);
        ncnn_net_destroy(net); return 1;
    }
    if (ncnn_net_load_model(net, bin_path) != 0) {
        fprintf(stderr, "Failed to load %s\n", bin_path);
        ncnn_net_destroy(net); return 1;
    }

    ncnn_mat_t in_mat = ncnn_mat_create_1d(td.n_features, NULL);

    for (int i = 0; i < n_warmup; i++) {
        float *ptr = (float *)ncnn_mat_get_data(in_mat);
        memcpy(ptr, td.X[i % td.n_samples], td.n_features * sizeof(float));
        ncnn_extractor_t ex = ncnn_extractor_create(net);
        ncnn_extractor_input(ex, "in0", in_mat);
        ncnn_mat_t out_mat;
        ncnn_extractor_extract(ex, "out0", &out_mat);
        ncnn_mat_destroy(out_mat); ncnn_extractor_destroy(ex);
    }

    static uint64_t pmc_data[N_RUNS][5];
    PerfFds perf = perf_open_all();
    double latencies[N_RUNS];
    int correct = 0;
    for (int i = 0; i < N_RUNS; i++) {
        int idx = i % td.n_samples;
        float *ptr = (float *)ncnn_mat_get_data(in_mat);
        memcpy(ptr, td.X[idx], td.n_features * sizeof(float));
        ncnn_extractor_t ex = ncnn_extractor_create(net);
        ncnn_extractor_input(ex, "in0", in_mat);
        ncnn_mat_t out_mat;
        perf_reset_enable(&perf);
        double t0 = time_us();
        ncnn_extractor_extract(ex, "out0", &out_mat);
        latencies[i] = time_us() - t0;
        perf_disable_read(&perf, pmc_data[i]);
        float *out_data = (float *)ncnn_mat_get_data(out_mat);
        float mse = 0.0f;
        for (int j = 0; j < td.n_features; j++) {
            float d = out_data[j] - td.X[idx][j];
            mse += d * d;
        }
        mse /= td.n_features;
        int pred = (mse >= threshold) ? 1 : 0;
        if (pred == (int)td.y[idx]) correct++;
        ncnn_mat_destroy(out_mat); ncnn_extractor_destroy(ex);
    }
    ncnn_mat_destroy(in_mat);
    ncnn_net_destroy(net);

    double mean, stdv, mn, mx, p95, p99;
    compute_stats(latencies, N_RUNS, &mean, &stdv, &mn, &mx, &p95, &p99);
    save_csv(csv_path, latencies, pmc_data, N_RUNS, label);

    printf("BENCH %s\n", label);
    printf("LATENCY_MEAN_US %.2f\n", mean);
    printf("LATENCY_STD_US  %.2f\n", stdv);
    printf("LATENCY_MIN_US  %.2f\n", mn);
    printf("LATENCY_MAX_US  %.2f\n", mx);
    printf("LATENCY_P95_US  %.2f\n", p95);
    printf("LATENCY_P99_US  %.2f\n", p99);
    printf("ACCURACY_PCT    %.2f\n", 100.0 * correct / N_RUNS);
    printf("TEST_SAMPLES    %d\n", td.n_samples);

    free(td.X[0]); free(td.X); free(td.y);
    for (int _i = 0; _i < 5; _i++)
        if (perf.fd[_i] >= 0) close(perf.fd[_i]);
    return 0;
}
