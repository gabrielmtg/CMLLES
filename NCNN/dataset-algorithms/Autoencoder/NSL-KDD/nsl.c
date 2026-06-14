#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <c_api.h>

#define TEST_DATA "model/nsl_test_data.txt"
#define PARAM_FILE "model/nsl_model.ncnn.param"
#define BIN_FILE "model/nsl_model.ncnn.bin"

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    printf("=============================================\n");
    printf("  NSL-KDD Autoencoder Monitor — NCNN (C API)\n");
    printf("=============================================\n\n");

    ncnn_net_t net = ncnn_net_create();
    if (ncnn_net_load_param(net, PARAM_FILE) != 0) {
        fprintf(stderr, "Erro ao carregar o parametro do modelo.\n");
        return 1;
    }
    if (ncnn_net_load_model(net, BIN_FILE) != 0) {
        fprintf(stderr, "Erro ao carregar o modelo binario.\n");
        return 1;
    }

    FILE *fin = fopen(TEST_DATA, "r");
    if (!fin) return 1;
    
    int num_samples, num_inputs, num_outputs;
    fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs);

    float* x_data_flat = (float*)malloc(num_samples * num_inputs * sizeof(float));
    float* y_data = (float*)malloc(num_samples * sizeof(float));

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_inputs; j++) {
            fscanf(fin, "%f", &x_data_flat[i * num_inputs + j]);
        }
        fscanf(fin, "%f", &y_data[i]);
    }
    fclose(fin);

    printf("Modelo carregado. A avaliar %d amostras...\n", num_samples);

    int correct = 0;
    double total_infer_ns = 0.0;
    float threshold = 0.008f; 

    for (int i = 0; i < num_samples; i++) {
        ncnn_extractor_t ex = ncnn_extractor_create(net);
        
        ncnn_mat_t in = ncnn_mat_create_external_1d(num_inputs, (void*)&x_data_flat[i * num_inputs], NULL);
        ncnn_extractor_input(ex, "in0", in);

        double t0 = time_ns();
        
        ncnn_mat_t out;
        ncnn_extractor_extract(ex, "out0", &out);
        
        total_infer_ns += (time_ns() - t0);

        float* out_ptr = (float*)ncnn_mat_get_data(out);

        float mse = 0.0f;
        for (int j = 0; j < num_inputs; j++) {
            float diff = x_data_flat[i * num_inputs + j] - out_ptr[j];
            mse += (diff * diff);
        }
        mse /= num_inputs;

        int true_val = (y_data[i] >= 0.5f) ? 1 : 0;
        int pred = (mse >= threshold) ? 1 : 0;

        if (pred == true_val) correct++;

        ncnn_mat_destroy(in);
        ncnn_mat_destroy(out);
        ncnn_extractor_destroy(ex);
    }

    printf("Exatidão: %d/%d (%.1f%%)\n", correct, num_samples, ((double)correct/num_samples)*100);

    ncnn_net_destroy(net);
    free(x_data_flat);
    free(y_data);

    return 0;
}
