#ifndef IAES_LARGE_RELU_PARAMS_H
#define IAES_LARGE_RELU_PARAMS_H

#define NUM_FC_LAYERS 5
#define INPUT_SCALE_F32 0.0078740157f
static const int LAYER_SIZES[] = {12, 128, 256, 128, 64, 1};
static const int32_t FC_MULTIPLIERS[] = {6474, 10962, 8551, 8291, 5833};
static const int FC_SHIFTS[] = {-20, -20, -20, -20, -20};

#endif
