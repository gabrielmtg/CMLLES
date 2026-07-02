#include <stdio.h>
#include <fann.h>

static void gen_ae(const char *size_name, const unsigned int *layers, unsigned int num_layers) {
    struct fann *ann = fann_create_standard_array(num_layers, layers);
    fann_set_activation_function_hidden(ann, FANN_RELU);
    fann_set_activation_function_output(ann, FANN_RELU);
    fann_randomize_weights(ann, -1.0, 1.0);

    char path[256];
    snprintf(path, sizeof(path), "model/ae_%s_f32.net", size_name);
    fann_save(ann, path);
    printf("Generated %s\n", path);
    fann_destroy(ann);
}

int main(void) {
    unsigned int small_layers[]  = {64, 32, 16, 32, 64};
    unsigned int medium_layers[] = {128, 64, 32, 64, 128};
    unsigned int large_layers[]  = {256, 128, 64, 128, 256};

    gen_ae("small",  small_layers,  5);
    gen_ae("medium", medium_layers, 5);
    gen_ae("large",  large_layers,  5);
    return 0;
}
