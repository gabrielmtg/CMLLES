#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "genann.h"

#define N_INPUTS  4
#define N_OUTPUTS 3
#define EPOCHS    500
#define LR        0.05

typedef struct {
    const char *size;
    int         hidden_layers;
    int         hidden_neurons;
} ModelConfig;

static ModelConfig CONFIGS[] = {
    {"small",  1, 64},
    {"medium", 2, 64},
    {"large",  3, 64},
};
#define N_CONFIGS 3

static double *train_X   = NULL;
static double *train_Y   = NULL;
static int     n_train   = 0;
static double  test_X[30][N_INPUTS];
static int     test_y[30];
static int     n_test    = 0;

static void load_train_data(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int n_inp, n_out;
    fscanf(f, "%d %d %d", &n_train, &n_inp, &n_out);
    train_X = malloc((size_t)n_train * n_inp  * sizeof(double));
    train_Y = malloc((size_t)n_train * n_out  * sizeof(double));
    for (int i = 0; i < n_train; i++) {
        for (int j = 0; j < n_inp;  j++) fscanf(f, "%lf", &train_X[i * n_inp  + j]);
        for (int j = 0; j < n_out;  j++) fscanf(f, "%lf", &train_Y[i * n_out  + j]);
    }
    fclose(f);
}

static void load_test_data(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int n_feat, n_out;
    fscanf(f, "%d %d %d", &n_test, &n_feat, &n_out);
    for (int i = 0; i < n_test; i++) {
        for (int j = 0; j < N_INPUTS; j++) fscanf(f, "%lf", &test_X[i][j]);
        fscanf(f, "%d", &test_y[i]);
    }
    fclose(f);
}

static double clock_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(void) {
    load_train_data("model/iris_genann_train.data");
    load_test_data("model/test_data.txt");

    FILE *jf = fopen("model/training_metrics.json", "w");
    if (!jf) { perror("training_metrics.json"); return 1; }
    fprintf(jf, "{\n");

    for (int ci = 0; ci < N_CONFIGS; ci++) {
        ModelConfig *cfg = &CONFIGS[ci];
        char key[64], net_path[128];
        snprintf(key,      sizeof(key),      "mlp_%s_sigmoid", cfg->size);
        snprintf(net_path, sizeof(net_path), "model/mlp_%s_sigmoid_f64.genann", cfg->size);

        genann *ann = genann_init(N_INPUTS, cfg->hidden_layers, cfg->hidden_neurons, N_OUTPUTS);
        if (!ann) { fprintf(stderr, "genann_init failed\n"); return 1; }

        double t0 = clock_us();
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (int s = 0; s < n_train; s++) {
                genann_train(ann,
                    &train_X[s * N_INPUTS],
                    &train_Y[s * N_OUTPUTS],
                    LR);
            }
        }
        double elapsed_s = (clock_us() - t0) / 1e6;

        double mse = 0.0;
        int correct = 0;
        for (int s = 0; s < n_train; s++) {
            const double *out = genann_run(ann, &train_X[s * N_INPUTS]);
            for (int j = 0; j < N_OUTPUTS; j++) {
                double d = out[j] - train_Y[s * N_OUTPUTS + j];
                mse += d * d;
            }
        }
        mse /= (double)(n_train * N_OUTPUTS);

        for (int i = 0; i < n_test; i++) {
            const double *out = genann_run(ann, test_X[i]);
            int pred = 0;
            for (int c = 1; c < N_OUTPUTS; c++)
                if (out[c] > out[pred]) pred = c;
            if (pred == test_y[i]) correct++;
        }
        double acc = 100.0 * correct / n_test;

        FILE *fp = fopen(net_path, "w");
        genann_write(ann, fp);
        fclose(fp);

        printf("%s: mse=%.5f acc=%.1f%% time=%.2fs\n", key, mse, acc, elapsed_s);

        fprintf(jf, "  \"%s\": {\n", key);
        fprintf(jf, "    \"training_time_s\": %.3f,\n", elapsed_s);
        fprintf(jf, "    \"epochs\": %d,\n", EPOCHS);
        fprintf(jf, "    \"final_train_mse\": %.6f,\n", mse);
        fprintf(jf, "    \"final_test_accuracy_pct\": %.2f,\n", acc);
        fprintf(jf, "    \"test_samples\": %d\n", n_test);
        fprintf(jf, "  }%s\n", (ci < N_CONFIGS - 1) ? "," : "");

        genann_free(ann);
    }

    fprintf(jf, "}\n");
    fclose(jf);
    free(train_X);
    free(train_Y);
    printf("Saved model/training_metrics.json\n");
    return 0;
}
