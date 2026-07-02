#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <fann.h>

#define N_FEATS 122
#define EPOCHS  50

typedef struct {
    const char   *size;
    unsigned int  n_layers;
    unsigned int  layers[8];
} AEConfig;

static AEConfig CONFIGS[] = {
    {"small",  7, {122, 64, 32, 16, 32, 64, 122, 0}},
    {"medium", 7, {122, 96, 48, 24, 48, 96, 122, 0}},
    {"large",  7, {122, 128, 64, 32, 64, 128, 122, 0}},
};
#define N_CONFIGS 3

static float *test_X  = NULL;
static float *test_y  = NULL;
static int    n_test  = 0;
static int    n_feats = 0;

static void load_test_data(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int n_out;
    fscanf(f, "%d %d %d", &n_test, &n_feats, &n_out);
    test_X = malloc((size_t)n_test * n_feats * sizeof(float));
    test_y = malloc((size_t)n_test * sizeof(float));
    for (int i = 0; i < n_test; i++) {
        for (int j = 0; j < n_feats; j++)
            fscanf(f, "%f", &test_X[i * n_feats + j]);
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

    struct fann_train_data *data = fann_read_train_from_file("model/nsl_ae_train.data");
    if (!data) { fprintf(stderr, "Cannot load train data\n"); return 1; }

    fann_type *input_buf = malloc(n_feats * sizeof(fann_type));

    FILE *jf = fopen("model/training_metrics.json", "w");
    if (!jf) { perror("training_metrics.json"); return 1; }
    fprintf(jf, "{\n");

    for (int ci = 0; ci < N_CONFIGS; ci++) {
        AEConfig *cfg = &CONFIGS[ci];
        char key[64], net_path[128], thresh_path[128];
        snprintf(key,        sizeof(key),        "ae_%s", cfg->size);
        snprintf(net_path,   sizeof(net_path),   "model/ae_%s_f32.net", cfg->size);
        snprintf(thresh_path, sizeof(thresh_path), "model/ae_%s_threshold.txt", cfg->size);

        struct fann *ann = fann_create_standard_array(cfg->n_layers, cfg->layers);
        fann_set_activation_function_hidden(ann, FANN_SIGMOID);
        fann_set_activation_function_output(ann, FANN_SIGMOID);
        fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

        double t0 = clock_us();
        fann_train_on_data(ann, data, EPOCHS, 0, 0.0001f);
        double elapsed_s = (clock_us() - t0) / 1e6;

        float final_mse = fann_get_MSE(ann);

        unsigned int n_normal = fann_length_train_data(data);
        double mse_sum = 0.0, mse_sq_sum = 0.0;
        for (unsigned int i = 0; i < n_normal; i++) {
            fann_type *inp = fann_get_train_input(data, i);
            fann_type *out = fann_run(ann, inp);
            float mse = 0.0f;
            for (int j = 0; j < N_FEATS; j++) {
                float d = (float)out[j] - (float)inp[j];
                mse += d * d;
            }
            mse /= N_FEATS;
            mse_sum    += mse;
            mse_sq_sum += (double)mse * mse;
        }
        double mse_mean = mse_sum / n_normal;
        double mse_std  = sqrt(mse_sq_sum / n_normal - mse_mean * mse_mean);
        float threshold = (float)(mse_mean + 2.0 * mse_std);

        FILE *tf = fopen(thresh_path, "w");
        fprintf(tf, "%.8f\n", threshold);
        fclose(tf);

        int correct = 0;
        for (int i = 0; i < n_test; i++) {
            for (int j = 0; j < n_feats; j++)
                input_buf[j] = (fann_type)test_X[i * n_feats + j];
            fann_type *out = fann_run(ann, input_buf);
            float mse = 0.0f;
            for (int j = 0; j < n_feats; j++) {
                float d = (float)out[j] - test_X[i * n_feats + j];
                mse += d * d;
            }
            mse /= n_feats;
            int pred = (mse >= threshold) ? 1 : 0;
            if (pred == (int)test_y[i]) correct++;
        }
        float acc = 100.0f * correct / n_test;

        fann_save(ann, net_path);
        printf("%s: mse=%.5f threshold=%.5f acc=%.1f%% time=%.2fs\n",
               key, final_mse, threshold, acc, elapsed_s);

        fprintf(jf, "  \"%s\": {\n", key);
        fprintf(jf, "    \"training_time_s\": %.3f,\n", elapsed_s);
        fprintf(jf, "    \"epochs\": %d,\n", EPOCHS);
        fprintf(jf, "    \"final_train_mse\": %.6f,\n", final_mse);
        fprintf(jf, "    \"threshold\": %.8f,\n", threshold);
        fprintf(jf, "    \"final_test_accuracy_pct\": %.2f,\n", acc);
        fprintf(jf, "    \"test_samples\": %d\n", n_test);
        fprintf(jf, "  }%s\n", (ci < N_CONFIGS - 1) ? "," : "");

        fann_destroy(ann);
    }

    fprintf(jf, "}\n");
    fclose(jf);
    free(input_buf);
    free(test_X);
    free(test_y);
    fann_destroy_train(data);
    printf("Saved model/training_metrics.json\n");
    return 0;
}
