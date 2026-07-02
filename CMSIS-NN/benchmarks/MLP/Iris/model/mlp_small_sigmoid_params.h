#ifndef MLP_SMALL_SIGMOID_PARAMS_H
#define MLP_SMALL_SIGMOID_PARAMS_H

#define NUM_FC_LAYERS 3
#define INPUT_SCALE_F32 0.0078740157f
static const int LAYER_SIZES[] = {4, 32, 64, 3};
static const int32_t FC_MULTIPLIERS[] = {36907, 12166, 8464};
static const int FC_SHIFTS[] = {-20, -20, -20};
#define USE_SIGMOID_OUTPUT 1

#endif
