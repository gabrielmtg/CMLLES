#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <onnxruntime_c_api.h>

static const OrtApi *g_ort = NULL;

#define N_RUNS   1000
#define N_WARMUP 5

#define CHECK_ORT(expr) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        fprintf(stderr, "ORT error: %s\n", g_ort->GetErrorMessage(_s)); \
        g_ort->ReleaseStatus(_s); \
        exit(1); \
    } \
} while (0)

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
    float *images;
    int   *labels;
    int    n_samples;
    int    channels;
    int    height;
    int    width;
} ImageData;

static void load_test_images(const char *path, ImageData *td) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
    }
    int header[4];
    fread(header, sizeof(int), 4, f);
    td->n_samples = header[0];
    td->channels  = header[1];
    td->height    = header[2];
    td->width     = header[3];
    int sz = td->n_samples * td->channels * td->height * td->width;
    td->images = malloc(sz * sizeof(float));
    fread(td->images, sizeof(float), sz, f);
    fclose(f);
}

static void load_test_labels(const char *path, ImageData *td) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
    }
    int n;
    fread(&n, sizeof(int), 1, f);
    td->labels = malloc(n * sizeof(int));
    fread(td->labels, sizeof(int), n, f);
    fclose(f);
}

static void save_csv(const char *path, double *lat, int n, const char *label) {
    FILE *f = fopen(path, "w");
    if (!f)
        return;
    fprintf(f, "run,latency_us,label\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%d,%.3f,%s\n", i + 1, lat[i], label);
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config> [n_warmup]\n", argv[0]);
        fprintf(stderr, "  config: simple | intermediate | mobilenet\n");
        return 1;
    }
    const char *cfg = argv[1];
    int n_warmup = (argc > 2) ? atoi(argv[2]) : N_WARMUP;

    const char *img_file;
    if (strcmp(cfg, "mobilenet") == 0) {
        img_file = "model/test_images_96.bin";
    } else if (strcmp(cfg, "simple") == 0 || strcmp(cfg, "intermediate") == 0) {
        img_file = "model/test_images_32.bin";
    } else {
        fprintf(stderr, "Unknown config: %s\n", cfg);
        return 1;
    }

    char model_path[256], label[64], csv_path[256];
    snprintf(model_path, sizeof(model_path), "model/cnn_%s_f32.onnx", cfg);
    snprintf(label, sizeof(label), "cnn_%s_f32", cfg);
    mkdir("latencies", 0755);
    snprintf(csv_path, sizeof(csv_path), "latencies/latencies_%s.csv", label);

    ImageData td;
    load_test_images(img_file, &td);
    load_test_labels("model/test_labels.bin", &td);

    int img_sz = td.channels * td.height * td.width;

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv *env;
    CHECK_ORT(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bench", &env));

    OrtSessionOptions *opts;
    CHECK_ORT(g_ort->CreateSessionOptions(&opts));
    CHECK_ORT(g_ort->SetIntraOpNumThreads(opts, 1));
    CHECK_ORT(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));

    OrtSession *session;
    CHECK_ORT(g_ort->CreateSession(env, model_path, opts, &session));

    OrtMemoryInfo *mem_info;
    CHECK_ORT(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    int64_t shape[] = {1, td.channels, td.height, td.width};
    const char *input_names[]  = {"input"};
    const char *output_names[] = {"output"};

    float *input_buf = malloc(img_sz * sizeof(float));
    OrtValue *in_tensor = NULL;
    CHECK_ORT(g_ort->CreateTensorWithDataAsOrtValue(mem_info, input_buf,
        img_sz * sizeof(float), shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_tensor));

    for (int i = 0; i < n_warmup; i++) {
        int idx = i % td.n_samples;
        memcpy(input_buf, td.images + idx * img_sz, img_sz * sizeof(float));
        OrtValue *out_t = NULL;
        CHECK_ORT(g_ort->Run(session, NULL, input_names,
            (const OrtValue *const *)&in_tensor, 1, output_names, 1, &out_t));
        g_ort->ReleaseValue(out_t);
    }

    double latencies[N_RUNS];
    int correct = 0;
    for (int i = 0; i < N_RUNS; i++) {
        int idx = i % td.n_samples;
        memcpy(input_buf, td.images + idx * img_sz, img_sz * sizeof(float));
        OrtValue *out_t = NULL;
        double t0 = time_us();
        CHECK_ORT(g_ort->Run(session, NULL, input_names,
            (const OrtValue *const *)&in_tensor, 1, output_names, 1, &out_t));
        latencies[i] = time_us() - t0;

        float *out_data;
        CHECK_ORT(g_ort->GetTensorMutableData(out_t, (void **)&out_data));
        int pred = (out_data[1] > out_data[0]) ? 1 : 0;
        if (pred == td.labels[idx])
            correct++;
        g_ort->ReleaseValue(out_t);
    }

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

    free(input_buf);
    free(td.images);
    free(td.labels);
    g_ort->ReleaseValue(in_tensor);
    g_ort->ReleaseMemoryInfo(mem_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(opts);
    g_ort->ReleaseEnv(env);
    return 0;
}
