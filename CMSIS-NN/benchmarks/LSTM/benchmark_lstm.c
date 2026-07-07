#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "arm_nnfunctions.h"

#include MODEL_WEIGHTS_H
#include MODEL_PARAMS_H

#define N_RUNS    1000
#define N_WARMUP  5
#define MAX_RUL_F 125.0f
#define FC_OUT_SCALE (1.0f / 127.0f)

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
    float *X;
    float *y;
    int    n_samples;
    int    seq_feats;
} TestData;

static void load_test_data(const char *path, TestData *td) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int n, feats, nout;
    if (fscanf(f, "%d %d %d", &n, &feats, &nout) != 3) {
        fputs("Bad header\n", stderr); exit(1);
    }
    td->n_samples = n;
    td->seq_feats = feats;
    td->X = malloc((size_t)n * feats * sizeof(float));
    td->y = malloc((size_t)n * sizeof(float));
    if (!td->X || !td->y) { fputs("malloc failed\n", stderr); exit(1); }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < feats; j++)
            fscanf(f, "%f", &td->X[i * feats + j]);
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

static void quantize_input(const float *src, int8_t *dst, int n) {
    for (int i = 0; i < n; i++) {
        float v = src[i] / LSTM_INPUT_SCALE_F32;
        v = v < -128.0f ? -128.0f : (v > 127.0f ? 127.0f : v);
        dst[i] = (int8_t)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f));
    }
}

static float run_inference_rul(const int8_t *input_q, int8_t *output_q,
                                cmsis_nn_lstm_params *params,
                                int16_t *temp1, int16_t *temp2, int16_t *cell_state) {
    memset(cell_state, 0, LSTM_BATCH_SIZE * LSTM_HIDDEN_SIZE * sizeof(int16_t));
    cmsis_nn_lstm_context buffers;
    buffers.temp1       = temp1;
    buffers.temp2       = temp2;
    buffers.cell_state  = cell_state;
    arm_lstm_unidirectional_s8(input_q, output_q, params, &buffers);

    const int8_t *last_hidden = output_q + (LSTM_SEQ_LEN - 1) * LSTM_HIDDEN_SIZE;
    int8_t fc_out_buf[1];
    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_fc_params fp;
    fp.input_offset = 0; fp.filter_offset = 0; fp.output_offset = 0;
    fp.activation.min = -128; fp.activation.max = 127;
    cmsis_nn_per_tensor_quant_params fqp = {LSTM_FC_MULT, LSTM_FC_SHIFT};
    cmsis_nn_dims fi = {1, 1, 1, LSTM_HIDDEN_SIZE};
    cmsis_nn_dims ff = {1, 1, LSTM_HIDDEN_SIZE, 1};
    cmsis_nn_dims fb = {1, 1, 1, 1};
    cmsis_nn_dims fo = {1, 1, 1, 1};
    arm_fully_connected_s8(&ctx, &fp, &fqp, &fi, last_hidden, &ff,
                           fc_weights, &fb, fc_bias, &fo, fc_out_buf);
    float rul_norm = (float)fc_out_buf[0] * FC_OUT_SCALE;
    return rul_norm * MAX_RUL_F;
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
    int n_warmup = (argc > 1) ? atoi(argv[1]) : N_WARMUP;

    TestData td;
    load_test_data("model/test_data.txt", &td);

    int input_elems  = LSTM_SEQ_LEN * LSTM_BATCH_SIZE * LSTM_INPUT_SIZE;
    int output_elems = LSTM_SEQ_LEN * LSTM_BATCH_SIZE * LSTM_HIDDEN_SIZE;
    int state_elems  = LSTM_BATCH_SIZE * LSTM_HIDDEN_SIZE;

    int8_t  *input_q  = malloc(input_elems  * sizeof(int8_t));
    int8_t  *output_q = malloc(output_elems * sizeof(int8_t));
    int16_t *temp1    = malloc(state_elems  * sizeof(int16_t));
    int16_t *temp2    = malloc(state_elems  * sizeof(int16_t));
    int16_t *cell_st  = malloc(state_elems  * sizeof(int16_t));
    if (!input_q || !output_q || !temp1 || !temp2 || !cell_st) {
        fputs("malloc failed\n", stderr); return 1;
    }

    cmsis_nn_lstm_gate forget_gate = {
        LSTM_I2F_MULT, LSTM_I2F_SHIFT,
        forget_input_weights, forget_eff_bias_input,
        LSTM_R2F_MULT, LSTM_R2F_SHIFT,
        forget_hidden_weights, forget_eff_bias_hidden,
        NULL, ARM_SIGMOID
    };
    cmsis_nn_lstm_gate input_gate = {
        LSTM_I2I_MULT, LSTM_I2I_SHIFT,
        input_input_weights, input_eff_bias_input,
        LSTM_R2I_MULT, LSTM_R2I_SHIFT,
        input_hidden_weights, input_eff_bias_hidden,
        NULL, ARM_SIGMOID
    };
    cmsis_nn_lstm_gate cell_gate = {
        LSTM_I2C_MULT, LSTM_I2C_SHIFT,
        cell_input_weights, cell_eff_bias_input,
        LSTM_R2C_MULT, LSTM_R2C_SHIFT,
        cell_hidden_weights, cell_eff_bias_hidden,
        NULL, ARM_TANH
    };
    cmsis_nn_lstm_gate output_gate = {
        LSTM_I2O_MULT, LSTM_I2O_SHIFT,
        output_input_weights, output_eff_bias_input,
        LSTM_R2O_MULT, LSTM_R2O_SHIFT,
        output_hidden_weights, output_eff_bias_hidden,
        NULL, ARM_SIGMOID
    };

    cmsis_nn_lstm_params params;
    params.time_major               = 0;
    params.batch_size               = LSTM_BATCH_SIZE;
    params.time_steps               = LSTM_SEQ_LEN;
    params.input_size               = LSTM_INPUT_SIZE;
    params.hidden_size              = LSTM_HIDDEN_SIZE;
    params.input_offset             = LSTM_INPUT_OFFSET;
    params.forget_to_cell_multiplier = LSTM_FORGET_TO_CELL_MULT;
    params.forget_to_cell_shift     = LSTM_FORGET_TO_CELL_SHIFT;
    params.input_to_cell_multiplier  = LSTM_INPUT_TO_CELL_MULT;
    params.input_to_cell_shift      = LSTM_INPUT_TO_CELL_SHIFT;
    params.cell_clip                = LSTM_CELL_CLIP;
    params.cell_scale_power         = LSTM_CELL_SCALE_POWER;
    params.output_multiplier        = LSTM_OUTPUT_MULT;
    params.output_shift             = LSTM_OUTPUT_SHIFT;
    params.output_offset            = LSTM_OUTPUT_OFFSET;
    params.forget_gate              = forget_gate;
    params.input_gate               = input_gate;
    params.cell_gate                = cell_gate;
    params.output_gate              = output_gate;

    for (int i = 0; i < n_warmup; i++) {
        int idx = i % td.n_samples;
        quantize_input(td.X + idx * td.seq_feats, input_q, input_elems);
        run_inference_rul(input_q, output_q, &params, temp1, temp2, cell_st);
    }

    mkdir("latencies", 0755);
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "latencies/latencies_" MODEL_LABEL ".csv");

    static uint64_t pmc_data[N_RUNS][5];
    PerfFds perf = perf_open_all();
    double latencies[N_RUNS];
    double mae_sum = 0.0;
    for (int i = 0; i < N_RUNS; i++) {
        int idx = i % td.n_samples;
        quantize_input(td.X + idx * td.seq_feats, input_q, input_elems);
        perf_reset_enable(&perf);
        double t0 = time_us();
        float rul_pred = run_inference_rul(input_q, output_q, &params, temp1, temp2, cell_st);
        latencies[i] = time_us() - t0;
        perf_disable_read(&perf, pmc_data[i]);
        float true_rul = td.y[idx] * MAX_RUL_F;
        float diff = rul_pred - true_rul;
        mae_sum += diff < 0.0f ? -diff : diff;
    }

    double mean, stdv, mn, mx, p95, p99;
    compute_stats(latencies, N_RUNS, &mean, &stdv, &mn, &mx, &p95, &p99);
    save_csv(csv_path, latencies, pmc_data, N_RUNS, MODEL_LABEL);

    printf("BENCH " MODEL_LABEL "\n");
    printf("LATENCY_MEAN_US %.2f\n", mean);
    printf("LATENCY_STD_US  %.2f\n", stdv);
    printf("LATENCY_MIN_US  %.2f\n", mn);
    printf("LATENCY_MAX_US  %.2f\n", mx);
    printf("LATENCY_P95_US  %.2f\n", p95);
    printf("LATENCY_P99_US  %.2f\n", p99);
    printf("MAE_CYCLES      %.2f\n", mae_sum / N_RUNS);
    printf("TEST_SAMPLES    %d\n", td.n_samples);

    free(input_q); free(output_q);
    free(temp1); free(temp2); free(cell_st);
    free(td.X); free(td.y);
    for (int _i = 0; _i < 5; _i++)
        if (perf.fd[_i] >= 0) close(perf.fd[_i]);
    return 0;
}
