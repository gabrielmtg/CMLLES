#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
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

static void save_csv(const char *path, double *lat, int n, const char *label) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "run,latency_us,label\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%d,%.3f,%s\n", i + 1, lat[i], label);
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <size> <activation> [n_warmup]\n", argv[0]);
        return 1;
    }
    const char *size = argv[1];
    const char *act  = argv[2];
    int n_warmup = (argc > 3) ? atoi(argv[3]) : N_WARMUP;

    char param_path[256], bin_path[256], label[64], csv_path[256];
    snprintf(param_path, sizeof(param_path), "model/iaes_%s_%s_f32.ncnn.param", size, act);
    snprintf(bin_path,   sizeof(bin_path),   "model/iaes_%s_%s_f32.ncnn.bin",   size, act);
    snprintf(label, sizeof(label), "iaes_%s_%s_f32", size, act);
    mkdir("latencies", 0755);
    snprintf(csv_path, sizeof(csv_path), "latencies/latencies_%s.csv", label);

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

    double latencies[N_RUNS];
    int correct = 0;
    for (int i = 0; i < N_RUNS; i++) {
        int idx = i % td.n_samples;
        float *ptr = (float *)ncnn_mat_get_data(in_mat);
        memcpy(ptr, td.X[idx], td.n_features * sizeof(float));
        ncnn_extractor_t ex = ncnn_extractor_create(net);
        ncnn_extractor_input(ex, "in0", in_mat);
        ncnn_mat_t out_mat;
        double t0 = time_us();
        ncnn_extractor_extract(ex, "out0", &out_mat);
        latencies[i] = time_us() - t0;
        float *out_data = (float *)ncnn_mat_get_data(out_mat);
        int pred = out_data[0] >= 0.5f ? 1 : 0;
        if (pred == (int)td.y[idx]) correct++;
        ncnn_mat_destroy(out_mat); ncnn_extractor_destroy(ex);
    }
    ncnn_mat_destroy(in_mat);
    ncnn_net_destroy(net);

    double mean, stdv, mn, mx, p95, p99;
    compute_stats(latencies, N_RUNS, &mean, &stdv, &mn, &mx, &p95, &p99);
    save_csv(csv_path, latencies, N_RUNS, label);

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
    return 0;
}
