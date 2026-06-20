#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "genann.h"

#define INPUTS 38
#define HIDDEN 16
#define OUTPUTS 38
#define TEST_DATA "model/nsl_test_data.txt"
#define WEIGHTS_FILE "model/nsl_genann_weights.txt"

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    printf("========================================\n");
    printf(" NSL-KDD Autoencoder — Genann (Inferencia)\n");
    printf("========================================\n\n");

    // 1 Camada oculta de compressão (16 neurônios)
    genann *ann = genann_init(INPUTS, 1, HIDDEN, OUTPUTS);
    
    FILE *f_weights = fopen(WEIGHTS_FILE, "r");
    if (!f_weights) {
        printf("Erro ao abrir pesos. Rode o script Python primeiro!\n");
        return 1;
    }
    for (int i = 0; i < ann->total_weights; i++) {
        fscanf(f_weights, "%lf", &ann->weight[i]);
    }
    fclose(f_weights);
    printf("Pesos do PyTorch carregados (%d parametros).\n", ann->total_weights);

    FILE *fin = fopen(TEST_DATA, "r");
    if (!fin) {
        printf("Erro ao abrir dados de teste.\n");
        return 1;
    }
    
    int num_samples, num_inputs, num_outputs;
    fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs);

    double **x_data = malloc(num_samples * sizeof(double*));
    for (int i = 0; i < num_samples; i++) {
        x_data[i] = malloc(INPUTS * sizeof(double));
        for (int j = 0; j < INPUTS; j++) {
            fscanf(fin, "%lf", &x_data[i][j]);
        }
    }
    fclose(fin);

    double total_mse = 0.0;
    double total_infer_ns = 0.0;

    for (int i = 0; i < num_samples; i++) {
        double t0 = time_ns();
        const double *out = genann_run(ann, x_data[i]);
        total_infer_ns += (time_ns() - t0);

        // Calcula o Erro Quadrático Médio da reconstrução
        double sample_mse = 0.0;
        for(int j = 0; j < OUTPUTS; j++) {
            double diff = x_data[i][j] - out[j];
            sample_mse += diff * diff;
        }
        total_mse += (sample_mse / OUTPUTS);
    }

    double final_mse = total_mse / num_samples;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;

    printf("\n─── Resultados Genann NSL-KDD ───────────\n");
    printf("MSE (Erro Reconstrucao): %.6f\n", final_mse);
    printf("Latencia inferencia:     %.2f us (media por amostra)\n", avg_infer_us);
    printf("Inferencia total:        %.2f ms (%d amostras)\n", total_infer_ns / 1e6, num_samples);
    printf("─────────────────────────────────────────\n");

    genann_free(ann);
    for(int i = 0; i < num_samples; i++) free(x_data[i]);
    free(x_data);

    return 0;
}
