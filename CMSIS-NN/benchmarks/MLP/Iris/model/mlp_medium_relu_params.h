#ifndef MLP_MEDIUM_RELU_PARAMS_H
#define MLP_MEDIUM_RELU_PARAMS_H

#define NUM_FC_LAYERS 4
#define INPUT_SCALE_F32 0.0078740157f
static const int LAYER_SIZES[] = {4, 64, 128, 64, 3};
static const int32_t FC_MULTIPLIERS[] = {6427, 13886, 21937, 4722};
static const int FC_SHIFTS[] = {-20, -20, -20, -20};
#define USE_SIGMOID_OUTPUT 0

#endif
