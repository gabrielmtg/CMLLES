#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <c_api.h>

#define TEST_DATA "model/nsl_test_data.txt"
#define NCNN_PARAM "model/nsl_model.ncnn.param"
#define NCNN_BIN "model/nsl_model.ncnn.bin"

static double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    ncnn_net_t net = ncnn_net_create();

    if (ncnn_net_load_param(net, NCNN_PARAM) != 0) {
        ncnn_net_destroy(net);
        return 1;
    }
    if (ncnn_net_load_model(net, NCNN_BIN) != 0) {
        ncnn_net_destroy(net);
        return 1;
    }

    FILE *fin = fopen(TEST_DATA, "r");
    if (!fin) {
        ncnn_net_destroy(net);
        return 1;
    }
    
    int num_samples, num_inputs, num_outputs;
    if (fscanf(fin, "%d %d %d", &num_samples, &num_inputs, &num_outputs) != 3) {
        fclose(fin);
        ncnn_net_destroy(net);
        return 1;
    }

    float** x_data = (float**)malloc(num_samples * sizeof(float*));
    float* x_data_flat = (float*)malloc(num_samples * num_inputs * sizeof(float));
    float* y_data = (float*)malloc(num_samples * sizeof(float));

    for (int i = 0; i < num_samples; i++) {
        x_data[i] = &x_data_flat[i * num_inputs];
        for (int j = 0; j < num_inputs; j++) {
            if (fscanf(fin, "%f", &x_data[i][j]) != 1) {}
        }
        if (fscanf(fin, "%f", &y_data[i]) != 1) {}
    }
    fclose(fin);

    int correct = 0;
    double total_infer_ns = 0.0;
    float threshold = 0.008f; 

    for (int i = 0; i < num_samples; i++) {
        ncnn_mat_t in_mat = ncnn_mat_create_1d(num_inputs, NULL);
        
        float* in_ptr = (float*)ncnn_mat_get_data(in_mat);
        for (int j = 0; j < num_inputs; j++) {
            in_ptr[j] = x_data[i][j];
        }

        double t0 = time_ns();
        
        ncnn_extractor_t ex = ncnn_extractor_create(net);
        ncnn_extractor_input(ex, "in0", in_mat);
        
        ncnn_mat_t out_mat;
        ncnn_extractor_extract(ex, "out0", &out_mat);
        
        total_infer_ns += (time_ns() - t0);

        float* out_ptr = (float*)ncnn_mat_get_data(out_mat);
        
        float mse = 0.0f;
        for (int j = 0; j < num_inputs; j++) {
            float diff = x_data[i][j] - out_ptr[j];
            mse += (diff * diff);
        }
        mse /= num_inputs;

        int true_val = (y_data[i] >= 0.5f) ? 1 : 0;
        int pred = (mse >= threshold) ? 1 : 0;

        if (pred == true_val) correct++;

        ncnn_mat_destroy(in_mat);
        ncnn_mat_destroy(out_mat);
        ncnn_extractor_destroy(ex);
    }

    double accuracy = ((double)correct / num_samples) * 100.0;
    double avg_infer_us = (total_infer_ns / num_samples) / 1e3;
    double total_infer_ms = total_infer_ns / 1e6;

    printf("%.1f\n", accuracy);
    printf("%.2f\n", avg_infer_us);
    printf("%.2f\n", total_infer_ms);

    free(x_data_flat);
    free(x_data);
    free(y_data);

    ncnn_net_destroy(net);

    return 0;
}
