#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "arm_math.h"

#define IMG_SIZE 32
#define KERNEL 3
#define FILTERS 4
#define CONV_OUT 30
#define POOL_OUT 15
#define FLATTEN_SIZE (FILTERS * POOL_OUT * POOL_OUT)

void arm_sigmoid_f32(float32_t *pSrc, float32_t *pDst, uint32_t blockSize) {
    for (uint32_t i = 0; i < blockSize; i++) pDst[i] = 1.0f / (1.0f + expf(-pSrc[i]));
}

int main() {
    printf("========================================\n");
    printf(" VWW CNN 2D — CMSIS-DSP & C Nativo\n");
    printf("========================================\n\n");

    float32_t w_conv[FILTERS * KERNEL * KERNEL], b_conv[FILTERS];
    float32_t w_fc[1 * FLATTEN_SIZE], b_fc[1];

    FILE *fw = fopen("model/vww_cmsis_weights.txt", "r");
    for (int i = 0; i < FILTERS*KERNEL*KERNEL; i++) fscanf(fw, "%f", &w_conv[i]);
    for (int i = 0; i < FILTERS; i++)               fscanf(fw, "%f", &b_conv[i]);
    for (int i = 0; i < FLATTEN_SIZE; i++)          fscanf(fw, "%f", &w_fc[i]);
    for (int i = 0; i < 1; i++)                     fscanf(fw, "%f", &b_fc[i]);
    fclose(fw);

    arm_matrix_instance_f32 mat_w_fc; arm_mat_init_f32(&mat_w_fc, 1, FLATTEN_SIZE, w_fc);

    FILE *fin = fopen("model/vww_test_data.txt", "r");
    int samples, in_size, out_size;
    fscanf(fin, "%d %d %d", &samples, &in_size, &out_size);
    
    float32_t img[IMG_SIZE * IMG_SIZE];
    float32_t conv_out[FILTERS * CONV_OUT * CONV_OUT];
    float32_t pool_out[FLATTEN_SIZE];
    float32_t fc_out[1];
    
    arm_matrix_instance_f32 mat_pool, mat_fc;
    arm_mat_init_f32(&mat_pool, FLATTEN_SIZE, 1, pool_out);
    arm_mat_init_f32(&mat_fc, 1, 1, fc_out);

    for (int s = 0; s < 1; s++) { // Processando a primeira imagem como exemplo
        for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) fscanf(fin, "%f", &img[i]);

        // 1. Convolução 2D (Stride 1, Pad 0) + ReLU
        for (int f = 0; f < FILTERS; f++) {
            for (int y = 0; y < CONV_OUT; y++) {
                for (int x = 0; x < CONV_OUT; x++) {
                    float sum = b_conv[f];
                    for (int ky = 0; ky < KERNEL; ky++) {
                        for (int kx = 0; kx < KERNEL; kx++) {
                            float pixel = img[(y + ky) * IMG_SIZE + (x + kx)];
                            float weight = w_conv[f * KERNEL * KERNEL + ky * KERNEL + kx];
                            sum += pixel * weight;
                        }
                    }
                    // ReLU nativo
                    conv_out[f * CONV_OUT * CONV_OUT + y * CONV_OUT + x] = (sum > 0) ? sum : 0;
                }
            }
        }

        // 2. MaxPool 2D (Stride 2)
        for (int f = 0; f < FILTERS; f++) {
            for (int y = 0; y < POOL_OUT; y++) {
                for (int x = 0; x < POOL_OUT; x++) {
                    float max_val = -1e9;
                    for (int ky = 0; ky < 2; ky++) {
                        for (int kx = 0; kx < 2; kx++) {
                            float val = conv_out[f * CONV_OUT * CONV_OUT + (y*2 + ky) * CONV_OUT + (x*2 + kx)];
                            if (val > max_val) max_val = val;
                        }
                    }
                    pool_out[f * POOL_OUT * POOL_OUT + y * POOL_OUT + x] = max_val;
                }
            }
        }

        // 3. Flatten e Dense (CMSIS-DSP)
        arm_mat_mult_f32(&mat_w_fc, &mat_pool, &mat_fc);
        arm_add_f32(fc_out, b_fc, fc_out, 1);
        arm_sigmoid_f32(fc_out, fc_out, 1);

        printf("Amostra %d -> Predicao VWW: %.4f\n", s, fc_out[0]);
    }
    fclose(fin);
    return 0;
}