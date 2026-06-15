#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "arm_math.h"

#define SEQ_LEN 30
#define FEATURES 14
#define HIDDEN 16

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

void arm_sigmoid_f32(float32_t *pSrc, float32_t *pDst, uint32_t blockSize) {
    for (uint32_t i = 0; i < blockSize; i++) pDst[i] = 1.0f / (1.0f + expf(-pSrc[i]));
}

void arm_tanh_f32_custom(float32_t *pSrc, float32_t *pDst, uint32_t blockSize) {
    for (uint32_t i = 0; i < blockSize; i++) pDst[i] = tanhf(pSrc[i]);
}

int main() {
    printf("========================================\n");
    printf(" CMAPSS GRU — CMSIS-DSP (Série Temporal)\n");
    printf("========================================\n\n");

    float32_t wir[HIDDEN*FEATURES], wiz[HIDDEN*FEATURES], win[HIDDEN*FEATURES];
    float32_t whr[HIDDEN*HIDDEN], whz[HIDDEN*HIDDEN], whn[HIDDEN*HIDDEN];
    float32_t br[HIDDEN], bz[HIDDEN], bn[HIDDEN];
    float32_t w_fc[1*HIDDEN], b_fc[1];

    FILE *fw = fopen("model/cmapss_cmsis_weights.txt", "r");
    if (!fw) {
        printf("Erro ao abrir pesos da GRU.\n");
        return 1;
    }
    
    for(int i=0; i<HIDDEN*FEATURES; i++) fscanf(fw, "%f", &wir[i]);
    for(int i=0; i<HIDDEN*FEATURES; i++) fscanf(fw, "%f", &wiz[i]);
    for(int i=0; i<HIDDEN*FEATURES; i++) fscanf(fw, "%f", &win[i]);
    
    for(int i=0; i<HIDDEN*HIDDEN; i++) fscanf(fw, "%f", &whr[i]);
    for(int i=0; i<HIDDEN*HIDDEN; i++) fscanf(fw, "%f", &whz[i]);
    for(int i=0; i<HIDDEN*HIDDEN; i++) fscanf(fw, "%f", &whn[i]);
    
    for(int i=0; i<HIDDEN; i++) fscanf(fw, "%f", &br[i]);
    for(int i=0; i<HIDDEN; i++) fscanf(fw, "%f", &bz[i]);
    for(int i=0; i<HIDDEN; i++) fscanf(fw, "%f", &bn[i]);
    
    for(int i=0; i<HIDDEN; i++) fscanf(fw, "%f", &w_fc[i]);
    for(int i=0; i<1; i++)      fscanf(fw, "%f", &b_fc[i]);
    fclose(fw);

    arm_matrix_instance_f32 m_wir; arm_mat_init_f32(&m_wir, HIDDEN, FEATURES, wir);
    arm_matrix_instance_f32 m_wiz; arm_mat_init_f32(&m_wiz, HIDDEN, FEATURES, wiz);
    arm_matrix_instance_f32 m_win; arm_mat_init_f32(&m_win, HIDDEN, FEATURES, win);
    arm_matrix_instance_f32 m_whr; arm_mat_init_f32(&m_whr, HIDDEN, HIDDEN, whr);
    arm_matrix_instance_f32 m_whz; arm_mat_init_f32(&m_whz, HIDDEN, HIDDEN, whz);
    arm_matrix_instance_f32 m_whn; arm_mat_init_f32(&m_whn, HIDDEN, HIDDEN, whn);
    arm_matrix_instance_f32 m_fc;  arm_mat_init_f32(&m_fc,  1, HIDDEN, w_fc);

    FILE *fin = fopen("model/cmapss_test_data.txt", "r");
    if (!fin) {
        printf("Erro ao abrir dados de teste do CMAPSS.\n");
        return 1;
    }
    
    int samples, total_feat, out;
    fscanf(fin, "%d %d %d", &samples, &total_feat, &out);
    
    float32_t seq[SEQ_LEN * FEATURES];
    float32_t true_rul;
    
    float32_t h[HIDDEN];
    float32_t r[HIDDEN], z[HIDDEN], n[HIDDEN], t1[HIDDEN], t2[HIDDEN];
    float32_t final_rul[1];
    
    arm_matrix_instance_f32 m_h, m_x, m_t1, m_t2, m_out;
    arm_mat_init_f32(&m_h, HIDDEN, 1, h);
    arm_mat_init_f32(&m_t1, HIDDEN, 1, t1);
    arm_mat_init_f32(&m_t2, HIDDEN, 1, t2);
    arm_mat_init_f32(&m_out, 1, 1, final_rul);

    double total_mse = 0.0;
    double total_mae = 0.0;
    double total_infer_ns = 0.0;

    for(int s = 0; s < samples; s++) {
        // Leitura da sequência temporal (X)
        for (int i = 0; i < SEQ_LEN * FEATURES; i++) {
            fscanf(fin, "%f", &seq[i]);
        }
        // Leitura da RUL real (Y)
        fscanf(fin, "%f", &true_rul);
        
        // Zera o estado oculto para o novo motor
        for(int i=0; i<HIDDEN; i++) h[i] = 0.0f;

        double t0 = time_ns();

        // Loop temporal da sequência (Início da Inferência)
        for(int step = 0; step < SEQ_LEN; step++) {
            float32_t *x = &seq[step * FEATURES];
            arm_mat_init_f32(&m_x, FEATURES, 1, x);

            // 1. Reset Gate
            arm_mat_mult_f32(&m_wir, &m_x, &m_t1);
            arm_mat_mult_f32(&m_whr, &m_h, &m_t2);
            arm_add_f32(t1, t2, r, HIDDEN);
            arm_add_f32(r, br, r, HIDDEN);
            arm_sigmoid_f32(r, r, HIDDEN);

            // 2. Update Gate
            arm_mat_mult_f32(&m_wiz, &m_x, &m_t1);
            arm_mat_mult_f32(&m_whz, &m_h, &m_t2);
            arm_add_f32(t1, t2, z, HIDDEN);
            arm_add_f32(z, bz, z, HIDDEN);
            arm_sigmoid_f32(z, z, HIDDEN);

            // 3. New Gate
            arm_mat_mult_f32(&m_win, &m_x, &m_t1);
            arm_mat_mult_f32(&m_whn, &m_h, &m_t2);
            arm_mult_f32(r, m_t2.pData, m_t2.pData, HIDDEN);
            arm_add_f32(t1, m_t2.pData, n, HIDDEN);
            arm_add_f32(n, bn, n, HIDDEN);
            arm_tanh_f32_custom(n, n, HIDDEN);

            // 4. Update Hidden State
            for(int i=0; i<HIDDEN; i++) {
                h[i] = (1.0f - z[i]) * n[i] + z[i] * h[i];
            }
        }

        // Regressão Linear final
        arm_mat_mult_f32(&m_fc, &m_h, &m_out);
        arm_add_f32(final_rul, b_fc, final_rul, 1);
        
        total_infer_ns += (time_ns() - t0);

        // Cálculo dos Erros
        double error = final_rul[0] - true_rul;
        total_mse += (error * error);
        total_mae += fabs(error);
    }
    fclose(fin);

    double final_mse = total_mse / samples;
    double final_mae = total_mae / samples;
    double avg_infer_us = (total_infer_ns / samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("\n─── Resultados CMAPSS GRU ───────────────\n");
    printf("MSE (Erro Quadratico): %.4f\n", final_mse);
    printf("MAE (Erro Absoluto):   %.4f ciclos\n", final_mae);
    printf("Latencia inferencia:   %.2f us (media por motor)\n", avg_infer_us);
    printf("Inferencia total:      %.2f ms (%d motores)\n", total_infer_ms, samples);
    printf("─────────────────────────────────────────\n");

    return 0;
}