#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "arm_math.h"

#define INPUTS 38
#define HIDDEN 16
#define OUTPUTS 38
#define TEST_DATA "model/nsl_test_data.txt"
#define WEIGHTS_FILE "model/nsl_cmsis_weights.txt"

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// Função auxiliar para aplicar Sigmoid vetorizada
void arm_sigmoid_f32(float32_t *pSrc, float32_t *pDst, uint32_t blockSize) {
    for (uint32_t i = 0; i < blockSize; i++) {
        pDst[i] = 1.0f / (1.0f + expf(-pSrc[i]));
    }
}

int main() {
    printf("========================================\n");
    printf(" NSL-KDD Autoencoder — CMSIS-DSP\n");
    printf("========================================\n\n");

    // Alocação dos pesos e bias
    float32_t w1[HIDDEN * INPUTS], b1[HIDDEN];
    float32_t w2[OUTPUTS * HIDDEN], b2[OUTPUTS];

    FILE *f_weights = fopen(WEIGHTS_FILE, "r");
    if (!f_weights) {
        printf("Erro ao abrir pesos em %s\n", WEIGHTS_FILE);
        return 1;
    }

    // Leitura Camada 1 (Encoder)
    for (int i = 0; i < HIDDEN * INPUTS; i++) fscanf(f_weights, "%f", &w1[i]);
    for (int i = 0; i < HIDDEN; i++)          fscanf(f_weights, "%f", &b1[i]);
    
    // Leitura Camada 2 (Decoder)
    for (int i = 0; i < OUTPUTS * HIDDEN; i++) fscanf(f_weights, "%f", &w2[i]);
    for (int i = 0; i < OUTPUTS; i++)          fscanf(f_weights, "%f", &b2[i]);
    fclose(f_weights);
    printf("Pesos carregados no formato CMSIS-DSP.\n");

    // Instâncias de Matriz CMSIS
    arm_matrix_instance_f32 mat_w1; arm_mat_init_f32(&mat_w1, HIDDEN, INPUTS, w1);
    arm_matrix_instance_f32 mat_w2; arm_mat_init_f32(&mat_w2, OUTPUTS, HIDDEN, w2);

    FILE *fin = fopen(TEST_DATA, "r");
    if (!fin) {
        printf("Erro ao abrir dados de teste.\n");
        return 1;
    }
    
    int num_samples, num_inputs, num_outputs;
    fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs);

    float32_t **x_data = malloc(num_samples * sizeof(float32_t*));
    for (int i = 0; i < num_samples; i++) {
        x_data[i] = malloc(INPUTS * sizeof(float32_t));
        for (int j = 0; j < INPUTS; j++) {
            fscanf(fin, "%f", &x_data[i][j]);
        }
    }
    fclose(fin);

    double total_mse = 0.0;
    double total_infer_ns = 0.0;

    // Buffers intermediários para a inferência
    float32_t hidden_out[HIDDEN], final_out[OUTPUTS];
    
    arm_matrix_instance_f32 mat_in, mat_hidden, mat_final;
    arm_mat_init_f32(&mat_hidden, HIDDEN, 1, hidden_out);
    arm_mat_init_f32(&mat_final, OUTPUTS, 1, final_out);

    for (int i = 0; i < num_samples; i++) {
        arm_mat_init_f32(&mat_in, INPUTS, 1, x_data[i]);

        double t0 = time_ns();

        // Camada 1 (Encoder): W1*x + B1 -> Sigmoid
        arm_mat_mult_f32(&mat_w1, &mat_in, &mat_hidden);
        arm_add_f32(hidden_out, b1, hidden_out, HIDDEN);
        arm_sigmoid_f32(hidden_out, hidden_out, HIDDEN);

        // Camada 2 (Decoder): W2*hidden + B2 -> Sigmoid
        arm_mat_mult_f32(&mat_w2, &mat_hidden, &mat_final);
        arm_add_f32(final_out, b2, final_out, OUTPUTS);
        arm_sigmoid_f32(final_out, final_out, OUTPUTS);

        total_infer_ns += (time_ns() - t0);

        // Computa o MSE para a amostra
        double sample_mse = 0.0;
        for(int j = 0; j < OUTPUTS; j++) {
            double diff = x_data[i][j] - final_out[j];
            sample_mse += diff * diff;
        }
        total_mse += (sample_mse / OUTPUTS);
    }

    double final_mse = total_mse / num_samples;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;

    printf("\n─── Resultados CMSIS-DSP NSL-KDD ────────\n");
    printf("MSE (Erro Reconstrucao): %.6f\n", final_mse);
    printf("Latencia inferencia:     %.2f us (media por amostra)\n", avg_infer_us);
    printf("Inferencia total:        %.2f ms (%d amostras)\n", total_infer_ns / 1e6, num_samples);
    printf("─────────────────────────────────────────\n");

    for(int i = 0; i < num_samples; i++) free(x_data[i]);
    free(x_data);

    return 0;
}