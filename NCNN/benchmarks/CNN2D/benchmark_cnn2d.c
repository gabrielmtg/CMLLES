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
    *mn = lat[0]; *mx = lat[0];
    for (int i = 0; i < n; i++) {
        sum += lat[i];
        if (lat[i] < *mn) *mn = lat[i];
        if (lat[i] > *mx) *mx = lat[i];
    }
    *mean = sum / n;
    double var = 0.0;
    for (int i = 0; i < n; i++) { double d = lat[i] - *mean; var += d * d; }
    *stdv = sqrt(var / n);
    double sorted[N_RUNS];
    memcpy(sorted, lat, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double);
    *p95 = sorted[(int)(0.95 * n)];
    *p99 = sorted[(int)(0.99 * n)];
}

typedef struct {
    float *images;
    int   *labels;
    int    n_samples, channels, height, width;
} ImageData;

static void load_test_images(const char *path, ImageData *td) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int header[4];
    fread(header, sizeof(int), 4, f);
    td->n_samples = header[0]; td->channels = header[1];
    td->height = header[2]; td->width = header[3];
    int sz = td->n_samples * td->channels * td->height * td->width;
    td->images = malloc(sz * sizeof(float));
    fread(td->images, sizeof(float), sz, f);
    fclose(f);
}

static void load_test_labels(const char *path, ImageData *td) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int n; fread(&n, sizeof(int), 1, f);
    td->labels = malloc(n * sizeof(int));
    fread(td->labels, sizeof(int), n, f);
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

typedef struct { int fd[5]; } PerfFds;

static int open_perf_fd(uint32_t type, uint64_t config) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type       = type;
    attr.size       = sizeof(attr);
    attr.config     = config;
    attr.disabled   = 1;
    attr.exclude_hv = 1;
    return (int)perf_event_open_syscall(&attr, 0, -1, -1, 0);
}

static PerfFds perf_open_all(void) {
    PerfFds p;
    uint64_t l1_loads  = PERF_COUNT_HW_CACHE_L1D
                       | ((uint64_t)PERF_COUNT_HW_CACHE_OP_READ << 8)
                       | ((uint64_t)PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    uint64_t l1_misses = PERF_COUNT_HW_CACHE_L1D
                       | ((uint64_t)PERF_COUNT_HW_CACHE_OP_READ << 8)
                       | ((uint64_t)PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    p.fd[0] = open_perf_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    p.fd[1] = open_perf_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    p.fd[2] = open_perf_fd(PERF_TYPE_HW_CACHE, l1_loads);
    p.fd[3] = open_perf_fd(PERF_TYPE_HW_CACHE, l1_misses);
    p.fd[4] = open_perf_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
    return p;
}

static void perf_reset_enable(PerfFds *p) {
    for (int _i = 0; _i < 5; _i++) {
        if (p->fd[_i] >= 0) {
            ioctl(p->fd[_i], PERF_EVENT_IOC_RESET,  0);
            ioctl(p->fd[_i], PERF_EVENT_IOC_ENABLE, 0);
        }
    }
}

static void perf_disable_read(PerfFds *p, uint64_t *out) {
    for (int _i = 0; _i < 5; _i++) {
        if (p->fd[_i] >= 0) {
            ioctl(p->fd[_i], PERF_EVENT_IOC_DISABLE, 0);
            read(p->fd[_i], &out[_i], sizeof(uint64_t));
        } else {
            out[_i] = 0;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config> [n_warmup]\n", argv[0]);
        return 1;
    }
    const char *cfg = argv[1];
    int n_warmup = (argc > 2) ? atoi(argv[2]) : N_WARMUP;

    const char *img_file = (strcmp(cfg, "mobilenet") == 0)
        ? "model/test_images_96.bin" : "model/test_images_32.bin";

    char param_path[256], bin_path[256], label[64], csv_path[256];
    snprintf(param_path, sizeof(param_path), "model/cnn_%s_f32.ncnn.param", cfg);
    snprintf(bin_path,   sizeof(bin_path),   "model/cnn_%s_f32.ncnn.bin",   cfg);
    snprintf(label, sizeof(label), "cnn_%s_f32", cfg);
    mkdir("latencies", 0755);
    snprintf(csv_path, sizeof(csv_path), "latencies/latencies_%s.csv", label);

    ImageData td;
    load_test_images(img_file, &td);
    load_test_labels("model/test_labels.bin", &td);
    int img_sz = td.channels * td.height * td.width;

    ncnn_net_t net = ncnn_net_create();
    if (ncnn_net_load_param(net, param_path) != 0) {
        fprintf(stderr, "Failed to load %s\n", param_path);
        ncnn_net_destroy(net); return 1;
    }
    if (ncnn_net_load_model(net, bin_path) != 0) {
        fprintf(stderr, "Failed to load %s\n", bin_path);
        ncnn_net_destroy(net); return 1;
    }

    ncnn_mat_t in_mat = ncnn_mat_create_3d(td.width, td.height, td.channels, NULL);

    for (int i = 0; i < n_warmup; i++) {
        float *ptr = (float *)ncnn_mat_get_data(in_mat);
        memcpy(ptr, td.images + (i % td.n_samples) * img_sz, img_sz * sizeof(float));
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
        memcpy(ptr, td.images + idx * img_sz, img_sz * sizeof(float));
        ncnn_extractor_t ex = ncnn_extractor_create(net);
        ncnn_extractor_input(ex, "in0", in_mat);
        ncnn_mat_t out_mat;
        perf_reset_enable(&perf);
        double t0 = time_us();
        ncnn_extractor_extract(ex, "out0", &out_mat);
        latencies[i] = time_us() - t0;
        perf_disable_read(&perf, pmc_data[i]);
        float *out_data = (float *)ncnn_mat_get_data(out_mat);
        int pred = (out_data[1] > out_data[0]) ? 1 : 0;
        if (pred == td.labels[idx]) correct++;
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

    free(td.images); free(td.labels);
    for (int _i = 0; _i < 5; _i++)
        if (perf.fd[_i] >= 0) close(perf.fd[_i]);
    return 0;
}
