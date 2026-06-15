#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "arm_math.h" // Requer o CMSIS-DSP no include path

#define INPUTS 12
#define HIDDEN 16
// Ajuste os caminhos relativos dependendo de onde a pasta 'model' é gerada
#define TEST_DATA "model/iaes_test_data.txt"
#define WEIGHTS_FILE "model/iaes_cmsis_weights.txt"

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// Função auxiliar para aplicar Sigmoid em um array f32
void arm_sigmoid_f32(float32_t *pSrc, float32_t *pDst, uint32_t blockSize) {
    for (uint32_t i = 0; i < blockSize; i++) {
        pDst[i] = 1.0f / (1.0f + expf(-pSrc[i]));
    }
}

int main() {
    printf("========================================\n");
    printf("  IAES Monitor MLP — CMSIS-DSP (Inferencia)\n");
    printf("========================================\n\n");

    // Alocação dos pesos e bias
    float32_t w1[HIDDEN * INPUTS], b1[HIDDEN];
    float32_t w2[HIDDEN * HIDDEN], b2[HIDDEN];
    float32_t w3[1 * HIDDEN],      b3[1];

    FILE *f_weights = fopen(WEIGHTS_FILE, "r");
    if (!f_weights) {
        printf("Erro ao abrir pesos em %s\nExecute o script de treino Python primeiro!\n", WEIGHTS_FILE);
        return 1;
    }

    // Leitura Camada 1
    for (int i = 0; i < HIDDEN * INPUTS; i++) fscanf(f_weights, "%f", &w1[i]);
    for (int i = 0; i < HIDDEN; i++)          fscanf(f_weights, "%f", &b1[i]);
    // Leitura Camada 2
    for (int i = 0; i < HIDDEN * HIDDEN; i++) fscanf(f_weights, "%f", &w2[i]);
    for (int i = 0; i < HIDDEN; i++)          fscanf(f_weights, "%f", &b2[i]);
    // Leitura Camada 3
    for (int i = 0; i < 1 * HIDDEN; i++)      fscanf(f_weights, "%f", &w3[i]);
    for (int i = 0; i < 1; i++)               fscanf(f_weights, "%f", &b3[i]);
    fclose(f_weights);
    printf("Pesos carregados no formato CMSIS-DSP com sucesso.\n");

    // Inicialização das estruturas de Matriz do CMSIS
    arm_matrix_instance_f32 mat_w1; arm_mat_init_f32(&mat_w1, HIDDEN, INPUTS, w1);
    arm_matrix_instance_f32 mat_w2; arm_mat_init_f32(&mat_w2, HIDDEN, HIDDEN, w2);
    arm_matrix_instance_f32 mat_w3; arm_mat_init_f32(&mat_w3, 1, HIDDEN, w3);

    FILE *fin = fopen(TEST_DATA, "r");
    if (!fin) {
        printf("Erro ao abrir os dados de teste em %s\n", TEST_DATA);
        return 1;
    }
    
    int num_samples, num_inputs, num_outputs;
    fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs);

    float32_t **x_data = malloc(num_samples * sizeof(float32_t*));
    float32_t *y_data  = malloc(num_samples * sizeof(float32_t));

    for (int i = 0; i < num_samples; i++) {
        x_data[i] = malloc(INPUTS * sizeof(float32_t));
        for (int j = 0; j < INPUTS; j++) {
            fscanf(fin, "%f", &x_data[i][j]);
        }
        fscanf(fin, "%f", &y_data[i]);
    }
    fclose(fin);

    int correct = 0;
    double total_infer_ns = 0.0;

    // Buffers intermediários
    float32_t out1[HIDDEN], out2[HIDDEN], final_out[1];
    
    // Instâncias de vetor/matriz para os dados dinâmicos
    arm_matrix_instance_f32 mat_in, mat_out1, mat_out2, mat_final;
    arm_mat_init_f32(&mat_out1, HIDDEN, 1, out1);
    arm_mat_init_f32(&mat_out2, HIDDEN, 1, out2);
    arm_mat_init_f32(&mat_final, 1, 1, final_out);

    for (int i = 0; i < num_samples; i++) {
        arm_mat_init_f32(&mat_in, INPUTS, 1, x_data[i]);

        double t0 = time_ns();

        // Camada 1: W1*x + B1 -> Sigmoid
        arm_mat_mult_f32(&mat_w1, &mat_in, &mat_out1);
        arm_add_f32(out1, b1, out1, HIDDEN);
        arm_sigmoid_f32(out1, out1, HIDDEN);

        // Camada 2: W2*out1 + B2 -> Sigmoid
        arm_mat_mult_f32(&mat_w2, &mat_out1, &mat_out2);
        arm_add_f32(out2, b2, out2, HIDDEN);
        arm_sigmoid_f32(out2, out2, HIDDEN);

        // Camada 3: W3*out2 + B3 -> Sigmoid
        arm_mat_mult_f32(&mat_w3, &mat_out2, &mat_final);
        arm_add_f32(final_out, b3, final_out, 1);
        arm_sigmoid_f32(final_out, final_out, 1);

        total_infer_ns += (time_ns() - t0);

        int pred = (final_out[0] >= 0.5f) ? 1 : 0;
        int true_val = (y_data[i] >= 0.5f) ? 1 : 0;

        if (pred == true_val) correct++;
    }

    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("\n─── Resultados CMSIS-DSP ────────────────\n");
    printf("Acuracia:              %d/%d (%.1f%%)\n", correct, num_samples, accuracy);
    printf("Latencia inferencia:   %.2f us (media por amostra)\n", avg_infer_us);
    printf("Inferencia total:      %.2f ms (%d amostras)\n", total_infer_ms, num_samples);
    printf("─────────────────────────────────────────\n");

    for(int i = 0; i < num_samples; i++) free(x_data[i]);
    free(x_data);
    free(y_data);

    return 0;
}