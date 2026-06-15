#include <stdio.h>
#include <stdlib.h>
#include <c_api.h>

#define BATCH_SIZE 1
#define CHANNELS 3
#define HEIGHT 96
#define WIDTH 96

int main() {
    ncnn_net_t net = ncnn_net_create();

    if (ncnn_net_load_param(net, "model/vww_96_model.ncnn.param") != 0) {
        ncnn_net_destroy(net);
        return 1;
    }
    if (ncnn_net_load_model(net, "model/vww_96_model.ncnn.bin") != 0) {
        ncnn_net_destroy(net);
        return 1;
    }

    size_t input_tensor_size = BATCH_SIZE * CHANNELS * HEIGHT * WIDTH;
    float* input_tensor_values = (float*)malloc(input_tensor_size * sizeof(float));
    
    FILE* file = fopen("imagem_teste.bin", "rb");
    if (file == NULL) {
        ncnn_net_destroy(net);
        free(input_tensor_values);
        return 1;
    }
    
    if (fread(input_tensor_values, sizeof(float), input_tensor_size, file) != input_tensor_size) {}
    fclose(file);
    
    ncnn_mat_t in_mat = ncnn_mat_create_3d(WIDTH, HEIGHT, CHANNELS, NULL);
    float* in_ptr = (float*)ncnn_mat_get_data(in_mat);
    for(size_t i = 0; i < input_tensor_size; i++) {
        in_ptr[i] = input_tensor_values[i];
    }

    ncnn_extractor_t ex = ncnn_extractor_create(net);
    ncnn_extractor_input(ex, "in0", in_mat);

    ncnn_mat_t out_mat;
    ncnn_extractor_extract(ex, "out0", &out_mat);

    float* out_arr = (float*)ncnn_mat_get_data(out_mat);
    
    printf("%.4f\n", out_arr[0]);
    printf("%.4f\n", out_arr[1]);
    
    ncnn_mat_destroy(in_mat);
    ncnn_mat_destroy(out_mat);
    ncnn_extractor_destroy(ex);
    ncnn_net_destroy(net);
    free(input_tensor_values);

    return 0;
}
