#ifndef MLP_LARGE_RELU_PARAMS_H
#define MLP_LARGE_RELU_PARAMS_H

#define NUM_FC_LAYERS 5
#define INPUT_SCALE_F32 0.0078740157f
static const int LAYER_SIZES[] = {4, 128, 256, 128, 64, 3};
static const int32_t FC_MULTIPLIERS[] = {5799, 10254, 8589, 6321, 6557};
static const int FC_SHIFTS[] = {-20, -20, -20, -20, -20};
#define USE_SIGMOID_OUTPUT 0

#endif
