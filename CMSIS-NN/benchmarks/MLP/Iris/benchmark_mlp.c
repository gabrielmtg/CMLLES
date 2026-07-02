#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>

#include "arm_nnfunctions.h"

#include MODEL_WEIGHTS_H
#include MODEL_PARAMS_H

#define MAX_BUF_SIZE 512
#define N_RUNS       1000
#define N_WARMUP     5
#define N_CLASSES    3

static int8_t bufs[2][MAX_BUF_SIZE];

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

static void quantize_input(const float *src, int8_t *dst, int n) {
    for (int i = 0; i < n; i++) {
        float v = src[i] / INPUT_SCALE_F32;
        v = v < -128.0f ? -128.0f : (v > 127.0f ? 127.0f : v);
        dst[i] = (int8_t)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f));
    }
}

static int run_inference_acc(const int8_t *input) {
    const int8_t *in_ptr  = input;
    int8_t       *out_ptr = bufs[0];
    int cur = 0;

    for (int l = 0; l < NUM_FC_LAYERS; l++) {
        int n_in  = LAYER_SIZES[l];
        int n_out = LAYER_SIZES[l + 1];

        cmsis_nn_context ctx = {NULL, 0};
        cmsis_nn_fc_params fc_params;
        fc_params.input_offset  = 0;
        fc_params.output_offset = 0;
        fc_params.filter_offset = 0;
        fc_params.activation.min = (l < NUM_FC_LAYERS - 1) ? 0 : -128;
        fc_params.activation.max = 127;

        cmsis_nn_per_tensor_quant_params quant;
        quant.multiplier = FC_MULTIPLIERS[l];
        quant.shift      = FC_SHIFTS[l];

        cmsis_nn_dims in_dims   = {1, 1, 1, n_in};
        cmsis_nn_dims flt_dims  = {1, 1, n_in, n_out};
        cmsis_nn_dims bias_dims = {1, 1, 1, n_out};
        cmsis_nn_dims out_dims  = {1, 1, 1, n_out};

        arm_fully_connected_s8(&ctx, &fc_params, &quant,
                               &in_dims,  in_ptr,
                               &flt_dims, FC_WEIGHTS[l],
                               &bias_dims, FC_BIAS[l],
                               &out_dims, out_ptr);

        in_ptr = out_ptr;
        cur ^= 1;
        out_ptr = bufs[cur];
    }

    int pred = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (in_ptr[c] > in_ptr[pred]) pred = c;
    return pred;
}

int main(int argc, char *argv[]) {
    int n_warmup = (argc > 1) ? atoi(argv[1]) : N_WARMUP;

    TestData td;
    load_test_data("model/test_data.txt", &td);

    int input_size = LAYER_SIZES[0];
    int8_t *input_int8 = malloc(input_size * sizeof(int8_t));

    mkdir("latencies", 0755);
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "latencies/latencies_" MODEL_LABEL ".csv");

    for (int i = 0; i < n_warmup; i++) {
        quantize_input(td.X[i % td.n_samples], input_int8, input_size);
        run_inference_acc(input_int8);
    }

    double latencies[N_RUNS];
    int correct = 0;
    for (int i = 0; i < N_RUNS; i++) {
        int idx = i % td.n_samples;
        quantize_input(td.X[idx], input_int8, input_size);
        double t0 = time_us();
        int pred = run_inference_acc(input_int8);
        latencies[i] = time_us() - t0;
        if (pred == (int)td.y[idx]) correct++;
    }

    double mean, stdv, mn, mx, p95, p99;
    compute_stats(latencies, N_RUNS, &mean, &stdv, &mn, &mx, &p95, &p99);
    save_csv(csv_path, latencies, N_RUNS, MODEL_LABEL);

    printf("BENCH " MODEL_LABEL "\n");
    printf("LATENCY_MEAN_US %.2f\n", mean);
    printf("LATENCY_STD_US  %.2f\n", stdv);
    printf("LATENCY_MIN_US  %.2f\n", mn);
    printf("LATENCY_MAX_US  %.2f\n", mx);
    printf("LATENCY_P95_US  %.2f\n", p95);
    printf("LATENCY_P99_US  %.2f\n", p99);
    printf("ACCURACY_PCT    %.2f\n", 100.0 * correct / N_RUNS);
    printf("TEST_SAMPLES    %d\n", td.n_samples);

    free(input_int8);
    free(td.X[0]); free(td.X); free(td.y);
    return 0;
}
