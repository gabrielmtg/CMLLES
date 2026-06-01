/*
 * Iris MLP Classifier — CMSIS-NN
 *
 * Inferência em C puro usando funções CMSIS-NN fully connected (INT8).
 * Pesos pré-treinados e quantizados em Python (train_iris.py).
 *
 * Rede: 4 inputs → 16 hidden (ReLU) → 3 outputs
 * Todos os cálculos são feitos em aritmética inteira (q7/q31).
 *
 * Pré-requisito: executar train_iris.py para gerar model/iris_weights.h
 * Compilação:    make
 * Execução:      ./iris_mlp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "model/iris_weights.h"
#include "model/iris_parameters.h"

/* ─── Configurações ─────────────────────────────────────────── */
#define MAX_SAMPLES   200
#define DATASET_PATH  "../../../datasets/iris/iris.data"
#define SCRATCH_SIZE  512

/* ─── Nomes das classes ─────────────────────────────────────── */
static const char *class_names[OUTPUT_SIZE] = {
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
};

/* ─── Dados ─────────────────────────────────────────────────── */
static float raw_inputs[MAX_SAMPLES][INPUT_SIZE];
static int   labels[MAX_SAMPLES];
static int   num_samples = 0;

/* ─── Buffers de inferência ─────────────────────────────────── */
static int8_t  input_q[INPUT_SIZE];
static int8_t  hidden_q[HIDDEN_SIZE];
static int8_t  output_q[OUTPUT_SIZE];
static int16_t scratch_buf[SCRATCH_SIZE];

/* ─── Tempo em nanossegundos ────────────────────────────────── */
static double time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ─── Carregar dataset ──────────────────────────────────────── */
static int load_dataset(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Erro: não foi possível abrir '%s'\n", path);
        return -1;
    }

    char line[1024];
    num_samples = 0;

    while (fgets(line, sizeof(line), f) && num_samples < MAX_SAMPLES) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;

        char *tok = strtok(line, ",");
        int j;
        for (j = 0; j < INPUT_SIZE && tok; j++) {
            raw_inputs[num_samples][j] = (float)atof(tok);
            tok = strtok(NULL, ",");
        }
        if (!tok) continue;

        tok[strcspn(tok, "\r\n")] = '\0';

        if (strcmp(tok, class_names[0]) == 0)      labels[num_samples] = 0;
        else if (strcmp(tok, class_names[1]) == 0)  labels[num_samples] = 1;
        else if (strcmp(tok, class_names[2]) == 0)  labels[num_samples] = 2;
        else continue;

        num_samples++;
    }
    fclose(f);

    printf("[CMSIS-NN] %d amostras carregadas de '%s'\n", num_samples, path);
    return 0;
}

/* ─── Quantizar input float → int8 ─────────────────────────── */
static void quantize_input(const float *raw, int8_t *q)
{
    int j;
    for (j = 0; j < INPUT_SIZE; j++) {
        /* Normalizar para [0,1] usando parâmetros do treinamento */
        float range = input_max[j] - input_min[j];
        float norm = (range > 0.0f)
                     ? (raw[j] - input_min[j]) / range
                     : 0.0f;

        /* Mapear para [-128, 127] */
        int val = (int)(norm / INPUT_SCALE) + INPUT_ZERO_POINT;
        if (val < -128) val = -128;
        if (val > 127) val = 127;
        q[j] = (int8_t)val;
    }
}

/* ─── Inferência MLP com CMSIS-NN ───────────────────────────── */
static void run_inference(const int8_t *input, int8_t *output)
{
    cmsis_nn_context ctx;
    ctx.buf = scratch_buf;
    ctx.size = SCRATCH_SIZE * sizeof(int16_t);

    /* ── FC1: input [4] → hidden [16] ─────────────────────── */
    cmsis_nn_fc_params fc1_params;
    fc1_params.input_offset  = -INPUT_ZERO_POINT;
    fc1_params.output_offset = 0;
    fc1_params.filter_offset = 0;
    fc1_params.activation.min = -128;  /* ReLU: min=0 em escala quantizada */
    fc1_params.activation.max = 127;

    cmsis_nn_per_tensor_quant_params fc1_quant;
    fc1_quant.multiplier = (int32_t)(INPUT_SCALE * FC1_WEIGHT_SCALE /
                                      (INPUT_SCALE) * (1 << 20));
    fc1_quant.shift = -20;

    cmsis_nn_dims fc1_input_dims  = {1, 1, 1, INPUT_SIZE};
    cmsis_nn_dims fc1_filter_dims = {1, 1, INPUT_SIZE, HIDDEN_SIZE};
    cmsis_nn_dims fc1_bias_dims   = {1, 1, 1, HIDDEN_SIZE};
    cmsis_nn_dims fc1_output_dims = {1, 1, 1, HIDDEN_SIZE};

    arm_fully_connected_s8(&ctx,
                           &fc1_params, &fc1_quant,
                           &fc1_input_dims, input,
                           &fc1_filter_dims, fc1_weights,
                           &fc1_bias_dims, fc1_bias,
                           &fc1_output_dims, hidden_q);

    /* Aplicar ReLU manualmente */
    int i;
    for (i = 0; i < HIDDEN_SIZE; i++) {
        if (hidden_q[i] < 0) hidden_q[i] = 0;
    }

    /* ── FC2: hidden [16] → output [3] ────────────────────── */
    cmsis_nn_fc_params fc2_params;
    fc2_params.input_offset  = 0;
    fc2_params.output_offset = 0;
    fc2_params.filter_offset = 0;
    fc2_params.activation.min = -128;
    fc2_params.activation.max = 127;

    cmsis_nn_per_tensor_quant_params fc2_quant;
    fc2_quant.multiplier = (int32_t)(FC1_WEIGHT_SCALE * FC2_WEIGHT_SCALE /
                                      FC2_WEIGHT_SCALE * (1 << 20));
    fc2_quant.shift = -20;

    cmsis_nn_dims fc2_input_dims  = {1, 1, 1, HIDDEN_SIZE};
    cmsis_nn_dims fc2_filter_dims = {1, 1, HIDDEN_SIZE, OUTPUT_SIZE};
    cmsis_nn_dims fc2_bias_dims   = {1, 1, 1, OUTPUT_SIZE};
    cmsis_nn_dims fc2_output_dims = {1, 1, 1, OUTPUT_SIZE};

    arm_fully_connected_s8(&ctx,
                           &fc2_params, &fc2_quant,
                           &fc2_input_dims, hidden_q,
                           &fc2_filter_dims, fc2_weights,
                           &fc2_bias_dims, fc2_bias,
                           &fc2_output_dims, output);
}

/* ─── Utilitário: argmax int8 ───────────────────────────────── */
static int argmax_q(const int8_t *v, int n)
{
    int best = 0;
    int i;
    for (i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

/* ─── Main ──────────────────────────────────────────────────── */
int main(void)
{
    printf("========================================\n");
    printf("  Iris MLP — CMSIS-NN (C puro, INT8)\n");
    printf("========================================\n\n");

    /* 1. Carregar dados */
    if (load_dataset(DATASET_PATH) != 0)
        return 1;

    printf("Topologia: %d -> %d (ReLU) -> %d\n",
           INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Aritmética: INT8 quantizado\n\n");

    /* 2. Avaliação */
    int correct = 0;
    double total_infer_ns = 0.0;
    int i;

    for (i = 0; i < num_samples; i++) {
        quantize_input(raw_inputs[i], input_q);

        double t0 = time_ns();
        run_inference(input_q, output_q);
        total_infer_ns += time_ns() - t0;

        int predicted = argmax_q(output_q, OUTPUT_SIZE);
        if (predicted == labels[i]) correct++;
    }

    double accuracy = (double)correct / num_samples * 100.0;
    double avg_infer_us = total_infer_ns / num_samples / 1e3;

    /* 3. Resultados */
    printf("─── Resultados ─────────────────────────\n");
    printf("Acurácia:              %d/%d (%.1f%%)\n",
           correct, num_samples, accuracy);
    printf("Latência inferência:   %.2f µs (média por amostra)\n",
           avg_infer_us);
    printf("Inferência total:      %.2f ms (%d amostras)\n",
           total_infer_ns / 1e6, num_samples);
    printf("─────────────────────────────────────────\n");

    return 0;
}
