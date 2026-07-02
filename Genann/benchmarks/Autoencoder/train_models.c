#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "genann.h"

#define N_FEATS 122
#define EPOCHS   50
#define LR       0.01

typedef struct {
    const char *size;
    int         hidden_layers;
    int         hidden_neurons;
} AEConfig;

static AEConfig CONFIGS[] = {
    {"small",  1, 64},
    {"medium", 2, 64},
    {"large",  3, 64},
};
#define N_CONFIGS 3

static double *train_X  = NULL;
static int     n_train  = 0;
static double *test_X   = NULL;
static double *test_y   = NULL;
static int     n_test   = 0;
static int     n_feats  = 0;

static void load_train_data(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int n_inp, n_out;
    fscanf(f, "%d %d %d", &n_train, &n_inp, &n_out);
    train_X = malloc((size_t)n_train * n_inp * sizeof(double));
    double dummy_buf[N_FEATS];
    for (int i = 0; i < n_train; i++) {
        for (int j = 0; j < n_inp; j++) {
            fscanf(f, "%lf", &train_X[i * n_inp + j]);
        }
        for (int j = 0; j < n_out; j++) fscanf(f, "%lf", &dummy_buf[j]);
    }
    fclose(f);
}

static void load_test_data(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int n_out;
    fscanf(f, "%d %d %d", &n_test, &n_feats, &n_out);
    test_X = malloc((size_t)n_test * n_feats * sizeof(double));
    test_y = malloc((size_t)n_test * sizeof(double));
    for (int i = 0; i < n_test; i++) {
        for (int j = 0; j < n_feats; j++)
            fscanf(f, "%lf", &test_X[i * n_feats + j]);
        fscanf(f, "%lf", &test_y[i]);
    }
    fclose(f);
}

static double clock_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(void) {
    load_train_data("model/nsl_ae_train.data");
    load_test_data("model/test_data.txt");

    FILE *jf = fopen("model/training_metrics.json", "w");
    if (!jf) { perror("training_metrics.json"); return 1; }
    fprintf(jf, "{\n");

    for (int ci = 0; ci < N_CONFIGS; ci++) {
        AEConfig *cfg = &CONFIGS[ci];
        char key[64], net_path[128], thresh_path[128];
        snprintf(key,        sizeof(key),        "ae_%s", cfg->size);
        snprintf(net_path,   sizeof(net_path),   "model/ae_%s_f64.genann",    cfg->size);
        snprintf(thresh_path, sizeof(thresh_path), "model/ae_%s_threshold.txt", cfg->size);

        genann *ann = genann_init(N_FEATS, cfg->hidden_layers, cfg->hidden_neurons, N_FEATS);
        if (!ann) { fprintf(stderr, "genann_init failed\n"); return 1; }

        double t0 = clock_us();
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (int s = 0; s < n_train; s++) {
                double *xi = &train_X[s * N_FEATS];
                genann_train(ann, xi, xi, LR);
            }
        }
        double elapsed_s = (clock_us() - t0) / 1e6;

        double mse_sum = 0.0, mse_sq_sum = 0.0;
        for (int s = 0; s < n_train; s++) {
            double *xi = &train_X[s * N_FEATS];
            const double *out = genann_run(ann, xi);
            double mse = 0.0;
            for (int j = 0; j < N_FEATS; j++) {
                double d = out[j] - xi[j];
                mse += d * d;
            }
            mse /= N_FEATS;
            mse_sum    += mse;
            mse_sq_sum += mse * mse;
        }
        double mse_mean  = mse_sum / n_train;
        double mse_std   = sqrt(mse_sq_sum / n_train - mse_mean * mse_mean);
        double threshold = mse_mean + 2.0 * mse_std;

        FILE *tf = fopen(thresh_path, "w");
        fprintf(tf, "%.10f\n", threshold);
        fclose(tf);

        int correct = 0;
        for (int i = 0; i < n_test; i++) {
            double *xi = &test_X[i * n_feats];
            const double *out = genann_run(ann, xi);
            double mse = 0.0;
            for (int j = 0; j < n_feats; j++) {
                double d = out[j] - xi[j];
                mse += d * d;
            }
            mse /= n_feats;
            int pred = (mse >= threshold) ? 1 : 0;
            if (pred == (int)test_y[i]) correct++;
        }
        double acc = 100.0 * correct / n_test;

        FILE *fp = fopen(net_path, "w");
        genann_write(ann, fp);
        fclose(fp);

        printf("%s: threshold=%.5f acc=%.1f%% time=%.2fs\n", key, threshold, acc, elapsed_s);

        fprintf(jf, "  \"%s\": {\n", key);
        fprintf(jf, "    \"training_time_s\": %.3f,\n", elapsed_s);
        fprintf(jf, "    \"epochs\": %d,\n", EPOCHS);
        fprintf(jf, "    \"threshold\": %.10f,\n", threshold);
        fprintf(jf, "    \"final_test_accuracy_pct\": %.2f,\n", acc);
        fprintf(jf, "    \"test_samples\": %d\n", n_test);
        fprintf(jf, "  }%s\n", (ci < N_CONFIGS - 1) ? "," : "");

        genann_free(ann);
    }

    fprintf(jf, "}\n");
    fclose(jf);
    free(train_X);
    free(test_X);
    free(test_y);
    printf("Saved model/training_metrics.json\n");
    return 0;
}
