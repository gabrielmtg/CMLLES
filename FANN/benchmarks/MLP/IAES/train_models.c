#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <fann.h>

#define N_FEATS 12
#define EPOCHS  200

typedef struct {
    const char   *size;
    const char   *act_name;
    unsigned int  n_layers;
    unsigned int  layers[7];
    enum fann_activationfunc_enum act_fn;
} ModelConfig;

static ModelConfig CONFIGS[] = {
    {"small",  "relu",    4, {12, 32, 64, 1, 0, 0, 0}, FANN_LINEAR_PIECE_RECT},
    {"small",  "sigmoid", 4, {12, 32, 64, 1, 0, 0, 0}, FANN_SIGMOID},
    {"medium", "relu",    5, {12, 64, 128, 64, 1, 0, 0}, FANN_LINEAR_PIECE_RECT},
    {"medium", "sigmoid", 5, {12, 64, 128, 64, 1, 0, 0}, FANN_SIGMOID},
    {"large",  "relu",    6, {12, 128, 256, 128, 64, 1, 0}, FANN_LINEAR_PIECE_RECT},
    {"large",  "sigmoid", 6, {12, 128, 256, 128, 64, 1, 0}, FANN_SIGMOID},
};
#define N_CONFIGS 6

static float  *test_X    = NULL;
static float  *test_y    = NULL;
static int     n_test    = 0;

static void load_test_data(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int n_feat, n_out;
    fscanf(f, "%d %d %d", &n_test, &n_feat, &n_out);
    test_X = malloc((size_t)n_test * N_FEATS * sizeof(float));
    test_y = malloc((size_t)n_test * sizeof(float));
    for (int i = 0; i < n_test; i++) {
        for (int j = 0; j < N_FEATS; j++)
            fscanf(f, "%f", &test_X[i * N_FEATS + j]);
        fscanf(f, "%f", &test_y[i]);
    }
    fclose(f);
}

static double clock_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(void) {
    load_test_data("model/test_data.txt");

    FILE *jf = fopen("model/training_metrics.json", "w");
    if (!jf) { perror("training_metrics.json"); return 1; }
    fprintf(jf, "{\n");

    for (int ci = 0; ci < N_CONFIGS; ci++) {
        ModelConfig *cfg = &CONFIGS[ci];
        char key[64], net_path[128];
        snprintf(key,      sizeof(key),      "iaes_%s_%s", cfg->size, cfg->act_name);
        snprintf(net_path, sizeof(net_path), "model/iaes_%s_%s_f32.net",
                 cfg->size, cfg->act_name);

        struct fann *ann = fann_create_standard_array(cfg->n_layers, cfg->layers);
        fann_set_activation_function_hidden(ann, cfg->act_fn);
        fann_set_activation_function_output(ann, FANN_SIGMOID);
        fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
        fann_set_learning_rate(ann, 0.01f);

        struct fann_train_data *data = fann_read_train_from_file("model/iaes_fann_train.data");
        if (!data) { fprintf(stderr, "Cannot load iaes_fann_train.data\n"); return 1; }

        double t0 = clock_us();
        fann_train_on_data(ann, data, EPOCHS, 0, 0.0001f);
        double elapsed_s = (clock_us() - t0) / 1e6;

        float final_mse = fann_get_MSE(ann);

        fann_type input_buf[N_FEATS];
        int correct = 0;
        for (int i = 0; i < n_test; i++) {
            for (int j = 0; j < N_FEATS; j++)
                input_buf[j] = (fann_type)test_X[i * N_FEATS + j];
            fann_type *out = fann_run(ann, input_buf);
            int pred = out[0] >= 0.5f ? 1 : 0;
            if (pred == (int)test_y[i]) correct++;
        }
        float acc = 100.0f * correct / n_test;

        fann_save(ann, net_path);
        printf("%s: mse=%.5f acc=%.1f%% time=%.2fs\n", key, final_mse, acc, elapsed_s);

        fprintf(jf, "  \"%s\": {\n", key);
        fprintf(jf, "    \"training_time_s\": %.3f,\n", elapsed_s);
        fprintf(jf, "    \"epochs\": %d,\n", EPOCHS);
        fprintf(jf, "    \"final_train_mse\": %.6f,\n", final_mse);
        fprintf(jf, "    \"final_test_accuracy_pct\": %.2f,\n", acc);
        fprintf(jf, "    \"test_samples\": %d\n", n_test);
        fprintf(jf, "  }%s\n", (ci < N_CONFIGS - 1) ? "," : "");

        fann_destroy_train(data);
        fann_destroy(ann);
    }

    fprintf(jf, "}\n");
    fclose(jf);
    free(test_X);
    free(test_y);
    printf("Saved model/training_metrics.json\n");
    return 0;
}
