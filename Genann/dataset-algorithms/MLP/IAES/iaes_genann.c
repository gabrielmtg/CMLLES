#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "genann.h"

#define INPUTS 12
#define HIDDEN 16
// Ajuste os caminhos relativos dependendo de onde a pasta 'model' é gerada
#define TEST_DATA "model/iaes_test_data.txt"
#define WEIGHTS_FILE "model/iaes_genann_weights.txt"

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    printf("========================================\n");
    printf("  IAES Monitor MLP — Genann (Inferencia)\n");
    printf("========================================\n\n");

    // Inicializa topologia idêntica ao PyTorch: 12 entradas, 2 camadas ocultas de 16, 1 saída
    genann *ann = genann_init(INPUTS, 2, HIDDEN, 1);
    if (!ann) {
        fprintf(stderr, "Erro ao criar rede neural Genann.\n");
        return 1;
    }

    FILE *f_weights = fopen(WEIGHTS_FILE, "r");
    if (!f_weights) {
        printf("Erro ao abrir pesos em %s\nExecute o script de treino Python primeiro!\n", WEIGHTS_FILE);
        return 1;
    }
    
    for (int i = 0; i < ann->total_weights; i++) {
        fscanf(f_weights, "%lf", &ann->weight[i]);
    }
    fclose(f_weights);
    printf("Pesos do PyTorch carregados com sucesso (%d parametros).\n", ann->total_weights);

    FILE *fin = fopen(TEST_DATA, "r");
    if (!fin) {
        printf("Erro ao abrir os dados de teste em %s\n", TEST_DATA);
        return 1;
    }
    
    int num_samples, num_inputs, num_outputs;
    fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs);

    double **x_data = malloc(num_samples * sizeof(double*));
    double *y_data = malloc(num_samples * sizeof(double));

    for (int i = 0; i < num_samples; i++) {
        x_data[i] = malloc(INPUTS * sizeof(double));
        for (int j = 0; j < INPUTS; j++) {
            fscanf(fin, "%lf", &x_data[i][j]);
        }
        fscanf(fin, "%lf", &y_data[i]);
    }
    fclose(fin);

    int correct = 0;
    double total_infer_ns = 0.0;

    for (int i = 0; i < num_samples; i++) {
        double t0 = time_ns();
        const double *out = genann_run(ann, x_data[i]);
        total_infer_ns += (time_ns() - t0);

        int pred = (out[0] >= 0.5) ? 1 : 0;
        int true_val = (y_data[i] >= 0.5) ? 1 : 0;

        if (pred == true_val) correct++;
    }

    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("\n─── Resultados Genann IAES ──────────────\n");
    printf("Acuracia:              %d/%d (%.1f%%)\n", correct, num_samples, accuracy);
    printf("Latencia inferencia:   %.2f us (media por amostra)\n", avg_infer_us);
    printf("Inferencia total:      %.2f ms (%d amostras)\n", total_infer_ms, num_samples);
    printf("─────────────────────────────────────────\n");

    genann_free(ann);
    for(int i = 0; i < num_samples; i++) free(x_data[i]);
    free(x_data);
    free(y_data);

    return 0;
}
