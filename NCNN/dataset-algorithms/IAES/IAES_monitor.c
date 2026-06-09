#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <c_api.h>

#define TEST_DATA "model/iaes_test_data.txt"
#define NCNN_PARAM "model/iaes_model.ncnn.param"
#define NCNN_BIN "model/iaes_model.ncnn.bin"
#define INPUTS 12

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    printf("========================================\n");
    printf("  IAES Monitor MLP — NCNN C API (C)\n");
    printf("========================================\n\n");

    ncnn_net_t net = ncnn_net_create();

    if (ncnn_net_load_param(net, NCNN_PARAM) != 0) {
        printf("Erro ao carregar o arquivo .param: %s\n", NCNN_PARAM);
        ncnn_net_destroy(net);
        return 1;
    }
    if (ncnn_net_load_model(net, NCNN_BIN) != 0) {
        printf("Erro ao carregar o arquivo .bin: %s\n", NCNN_BIN);
        ncnn_net_destroy(net);
        return 1;
    }

    FILE *fin = fopen(TEST_DATA, "r");
    if (!fin) {
        printf("Erro ao abrir %s. Rode o script Python primeiro!\n", TEST_DATA);
        ncnn_net_destroy(net);
        return 1;
    }
    
    int num_samples, num_inputs, num_outputs;
    if (fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs) != 3) {
        printf("Erro ao ler cabeçalho do dataset.\n");
        fclose(fin);
        ncnn_net_destroy(net);
        return 1;
    }

    float** x_data = (float**)malloc(num_samples * sizeof(float*));
    float* y_data = (float*)malloc(num_samples * sizeof(float));

    for (int i = 0; i < num_samples; i++) {
        x_data[i] = (float*)malloc(INPUTS * sizeof(float));
        for (int j = 0; j < INPUTS; j++) {
            if (fscanf(fin, "%f", &x_data[i][j]) != 1) {
                printf("Aviso: Falha ao ler amostra %d, input %d\n", i, j);
            }
        }
        if (fscanf(fin, "%f", &y_data[i]) != 1) {
            printf("Aviso: Falha ao ler label da amostra %d\n", i);
        }
    }
    fclose(fin);

    printf("Modelo NCNN em C puro carregado. Avaliando %d amostras\n", num_samples);

    int correct = 0;
    double total_infer_ns = 0.0;

    for (int i = 0; i < num_samples; i++) {
        ncnn_mat_t in_mat = ncnn_mat_create_1d(INPUTS, NULL);
        
        float* in_ptr = (float*)ncnn_mat_get_data(in_mat);
        for (int j = 0; j < INPUTS; j++) {
            in_ptr[j] = x_data[i][j];
        }

        double t0 = time_ns();
        
        ncnn_extractor_t ex = ncnn_extractor_create(net);
        
        ncnn_extractor_input(ex, "in0", in_mat);
        
        ncnn_mat_t out_mat;
        ncnn_extractor_extract(ex, "out0", &out_mat);
        
        total_infer_ns += (time_ns() - t0);

        float* out_ptr = (float*)ncnn_mat_get_data(out_mat);
        float out_val = out_ptr[0];

        int pred = (out_val >= 0.5f) ? 1 : 0;
        int true_val = (y_data[i] >= 0.5f) ? 1 : 0;

        if (pred == true_val) correct++;

        ncnn_mat_destroy(in_mat);
        ncnn_mat_destroy(out_mat);
        ncnn_extractor_destroy(ex);
    }

    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("\n─── Resultados NCNN (C API) ───────────────────\n");
    printf("Acurácia:              %d/%d (%.1f%%)\n", correct, num_samples, accuracy);
    printf("Latência inferência:   %.2f us (média por amostra)\n", avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n", total_infer_ms, num_samples);
    printf("─────────────────────────────────────────\n");

    for(int i = 0; i < num_samples; i++) free(x_data[i]);
    free(x_data);
    free(y_data);

    ncnn_net_destroy(net);

    return 0;
}
