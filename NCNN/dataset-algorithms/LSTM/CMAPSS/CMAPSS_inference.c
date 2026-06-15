#include <stdio.h>
#include <stdlib.h>
#include <c_api.h>

int main() {
    ncnn_net_t net = ncnn_net_create();

    if (ncnn_net_load_param(net, "model/cmapss_lstm.ncnn.param") != 0) {
        ncnn_net_destroy(net);
        return -1;
    }
    if (ncnn_net_load_model(net, "model/cmapss_lstm.ncnn.bin") != 0) {
        ncnn_net_destroy(net);
        return -1;
    }

    size_t input_tensor_size = 1 * 30 * 14;
    float* input_tensor_values = (float*)malloc(input_tensor_size * sizeof(float));
    for (size_t i = 0; i < input_tensor_size; i++) {
        input_tensor_values[i] = 0.5f; 
    }

    ncnn_mat_t in_mat = ncnn_mat_create_2d(14, 30, NULL);
    float* in_ptr = (float*)ncnn_mat_get_data(in_mat);
    for(size_t i = 0; i < input_tensor_size; i++) {
        in_ptr[i] = input_tensor_values[i];
    }

    ncnn_extractor_t ex = ncnn_extractor_create(net);
    ncnn_extractor_input(ex, "in0", in_mat);

    ncnn_mat_t out_mat;
    ncnn_extractor_extract(ex, "out0", &out_mat);

    float* output_arr = (float*)ncnn_mat_get_data(out_mat);
    float rul_em_ciclos = output_arr[0] * 125.0f;

    printf("Predicted RUL: %.2f\n", rul_em_ciclos);

    ncnn_mat_destroy(in_mat);
    ncnn_mat_destroy(out_mat);
    ncnn_extractor_destroy(ex);
    ncnn_net_destroy(net);
    free(input_tensor_values);

    return 0;
}
