#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <c_api.h>

#define NUM_FEATURES  4
#define NUM_CLASSES   3
#define MAX_SAMPLES   200
#define MODEL_PARAM   "model/iris_mlp.ncnn.param"
#define MODEL_BIN     "model/iris_mlp.ncnn.bin"
#define DATASET_PATH  "../../../datasets/iris/iris.data"

static const char *class_names[NUM_CLASSES] = {
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
};

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int main() {
    printf("========================================\n");
    printf("  Iris MLP — NCNN (C Puro)\n");
    printf("========================================\n\n");

    ncnn_net_t net = ncnn_net_create();

    if (ncnn_net_load_param(net, MODEL_PARAM) != 0) {
        printf("Erro ao carregar o arquivo .param: %s\n", MODEL_PARAM);
        ncnn_net_destroy(net);
        return 1;
    }
    if (ncnn_net_load_model(net, MODEL_BIN) != 0) {
        printf("Erro ao carregar o arquivo .bin: %s\n", MODEL_BIN);
        ncnn_net_destroy(net);
        return 1;
    }

    FILE *fin = fopen(DATASET_PATH, "r");
    if (!fin) {
        printf("Erro ao abrir %s\n", DATASET_PATH);
        ncnn_net_destroy(net);
        return 1;
    }

    float** x_data = malloc(MAX_SAMPLES * sizeof(float*));
    int* y_data = malloc(MAX_SAMPLES * sizeof(int));
    int num_samples = 0;
    char line[1024];

    while (fgets(line, sizeof(line), fin) && num_samples < MAX_SAMPLES) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;

        x_data[num_samples] = malloc(NUM_FEATURES * sizeof(float));
        char *tok = strtok(line, ",");
        for (int j = 0; j < NUM_FEATURES && tok; j++) {
            x_data[num_samples][j] = (float)atof(tok);
            tok = strtok(NULL, ",");
        }
        
        if (!tok) {
            free(x_data[num_samples]);
            continue;
        }

        tok[strcspn(tok, "\r\n")] = '\0';
        if (strcmp(tok, class_names[0]) == 0) y_data[num_samples] = 0;
        else if (strcmp(tok, class_names[1]) == 0) y_data[num_samples] = 1;
        else if (strcmp(tok, class_names[2]) == 0) y_data[num_samples] = 2;
        else {
            free(x_data[num_samples]);
            continue;
        }
        num_samples++;
    }
    fclose(fin);

    float fmin[NUM_FEATURES], fmax[NUM_FEATURES];
    for (int j = 0; j < NUM_FEATURES; j++) {
        fmin[j] = x_data[0][j];
        fmax[j] = x_data[0][j];
    }
    for (int i = 1; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (x_data[i][j] < fmin[j]) fmin[j] = x_data[i][j];
            if (x_data[i][j] > fmax[j]) fmax[j] = x_data[i][j];
        }
    }
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            float range = fmax[j] - fmin[j];
            x_data[i][j] = (range > 0.0f) ? (x_data[i][j] - fmin[j]) / range : 0.0f;
        }
    }

    printf("Modelo carregado: %s / %s\n", MODEL_PARAM, MODEL_BIN);
    printf("[NCNN] %d amostras carregadas de '%s'\n\n", num_samples, DATASET_PATH);

    int correct = 0;
    double total_infer_ns = 0.0;

    for (int i = 0; i < num_samples; i++) {
        ncnn_mat_t in_mat = ncnn_mat_create_1d(NUM_FEATURES, NULL);
        float* in_ptr = (float*)ncnn_mat_get_data(in_mat);
        for (int j = 0; j < NUM_FEATURES; j++) {
            in_ptr[j] = x_data[i][j];
        }

        double t0 = time_ns();
        
        ncnn_extractor_t ex = ncnn_extractor_create(net);
        ncnn_extractor_input(ex, "in0", in_mat);
        
        ncnn_mat_t out_mat;
        ncnn_extractor_extract(ex, "out0", &out_mat);
        
        total_infer_ns += (time_ns() - t0);

        float* out_ptr = (float*)ncnn_mat_get_data(out_mat);
        int pred = argmax(out_ptr, NUM_CLASSES);
        if (pred == y_data[i]) correct++;

        ncnn_mat_destroy(in_mat);
        ncnn_mat_destroy(out_mat);
        ncnn_extractor_destroy(ex);
    }

    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("─── Resultados ─────────────────────────\n");
    printf("Acurácia:              %d/%d (%.1f%%)\n", correct, num_samples, accuracy);
    printf("Latência inferência:   %.2f us (média por amostra)\n", avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n", total_infer_ms, num_samples);
    printf("─────────────────────────────────────────\n");

    ncnn_net_destroy(net);

    for(int i = 0; i < num_samples; i++) free(x_data[i]);
    free(x_data);
    free(y_data);

    return 0;
}
