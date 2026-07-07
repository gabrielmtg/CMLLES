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

static int8_t *xmalloc8(int n) {
    int8_t *p = malloc(n);
    if (!p) { fputs("malloc failed\n", stderr); exit(1); }
    return p;
}

static void nchw_float_to_nhwc_int8(const float *src, int8_t *dst,
                                     int c, int h, int w) {
    for (int ih = 0; ih < h; ih++) {
        for (int iw = 0; iw < w; iw++) {
            for (int ic = 0; ic < c; ic++) {
                float v = src[ic * h * w + ih * w + iw] / INPUT_SCALE_F32;
                v = v < -128.0f ? -128.0f : (v > 127.0f ? 127.0f : v);
                dst[ih * w * c + iw * c + ic] = (int8_t)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f));
            }
        }
    }
}

static int run_cnn(const int8_t *input_nhwc) {
    int h = IN_HEIGHT, w = IN_WIDTH, oh, ow;
    int8_t *buf = NULL;
    const int8_t *in = input_nhwc;

    for (int l = 0; l < NUM_LAYER_INFO; l++) {
        const LayerInfo *li = &LAYER_INFO[l];
        const int8_t  *W = NULL;
        const int32_t *B = NULL;

        switch (l) {
#if NUM_CONV_LAYERS >= 1
            case 0: W = conv0_weights; B = conv0_bias; break;
#endif
#if NUM_CONV_LAYERS >= 2
            case 1: W = conv1_weights; B = conv1_bias; break;
#endif
#if NUM_CONV_LAYERS >= 3
            case 2: W = conv2_weights; B = conv2_bias; break;
#endif
#if NUM_CONV_LAYERS >= 4
            case 3: W = conv3_weights; B = conv3_bias; break;
#endif
            default: break;
        }

        oh = li->out_h; ow = li->out_w;
        int8_t *next = xmalloc8(oh * ow * li->out_c);

        cmsis_nn_context ctx = {NULL, 0};
        cmsis_nn_conv_params cp;
        cp.input_offset = 0; cp.output_offset = 0;
        cp.stride.h = li->s;  cp.stride.w = li->s;
        cp.dilation.h = 1;    cp.dilation.w = 1;
        cp.padding.h = li->p; cp.padding.w = li->p;
        cp.activation.min = 0; cp.activation.max = 127;

        int32_t ch_mults[128], ch_shifts[128];
        for (int c = 0; c < li->out_c; c++) {
            ch_mults[c] = CONV_MULTIPLIERS[l];
            ch_shifts[c] = CONV_SHIFTS[l];
        }
        cmsis_nn_per_channel_quant_params qp = {ch_mults, ch_shifts};
        cmsis_nn_dims id       = {1, li->in_h,  li->in_w,  li->in_c};
        cmsis_nn_dims fd       = {li->out_c, li->k, li->k, li->in_c};
        cmsis_nn_dims bd       = {1, 1, 1, li->out_c};
        cmsis_nn_dims upscale  = {0, 0, 0, 0};
        cmsis_nn_dims od       = {1, oh, ow, li->out_c};
        arm_convolve_s8(&ctx, &cp, &qp, &id, in, &fd, W, &bd, B, &upscale, &od, next);

        if (buf) free(buf);
        buf = next;
        in = buf;
        h = oh; w = ow;
    }

    int8_t *fc_out = xmalloc8(FC_OUT_SIZE);
    cmsis_nn_context ctx2 = {NULL, 0};
    cmsis_nn_fc_params fp  = {0, 0, 0, {-128, 127}};
    cmsis_nn_per_tensor_quant_params fqp = {FC_MULTIPLIER, FC_SHIFT};
    cmsis_nn_dims fi = {1, 1, 1, FC_IN_SIZE};
    cmsis_nn_dims ff = {1, 1, FC_IN_SIZE, FC_OUT_SIZE};
    cmsis_nn_dims fb = {1, 1, 1, FC_OUT_SIZE};
    cmsis_nn_dims fo = {1, 1, 1, FC_OUT_SIZE};
    arm_fully_connected_s8(&ctx2, &fp, &fqp, &fi, buf, &ff, fc_weights, &fb, fc_bias, &fo, fc_out);

    int pred = (fc_out[1] > fc_out[0]) ? 1 : 0;
    free(fc_out);
    free(buf);
    return pred;
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

    const char *img_file = (IN_WIDTH == 96) ? "model/test_images_96.bin" : "model/test_images_32.bin";
    ImageData td;
    load_test_images(img_file, &td);
    load_test_labels("model/test_labels.bin", &td);

    int img_sz_f  = td.channels * td.height * td.width;
    int8_t *input_nhwc = xmalloc8(td.height * td.width * td.channels);

    mkdir("latencies", 0755);
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "latencies/latencies_" MODEL_LABEL ".csv");

    for (int i = 0; i < n_warmup; i++) {
        int idx = i % td.n_samples;
        nchw_float_to_nhwc_int8(td.images + idx * img_sz_f, input_nhwc,
                                 td.channels, td.height, td.width);
        run_cnn(input_nhwc);
    }

    static uint64_t pmc_data[N_RUNS][5];
    PerfFds perf = perf_open_all();
    double latencies[N_RUNS];
    int correct = 0;
    for (int i = 0; i < N_RUNS; i++) {
        int idx = i % td.n_samples;
        nchw_float_to_nhwc_int8(td.images + idx * img_sz_f, input_nhwc,
                                 td.channels, td.height, td.width);
        perf_reset_enable(&perf);
        double t0 = time_us();
        int pred = run_cnn(input_nhwc);
        latencies[i] = time_us() - t0;
        perf_disable_read(&perf, pmc_data[i]);
        if (pred == td.labels[idx]) correct++;
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
    printf("ACCURACY_PCT    %.2f\n", 100.0 * correct / N_RUNS);
    printf("TEST_SAMPLES    %d\n", td.n_samples);

    free(input_nhwc);
    free(td.images); free(td.labels);
    for (int _i = 0; _i < 5; _i++)
        if (perf.fd[_i] >= 0) close(perf.fd[_i]);
    return 0;
}
