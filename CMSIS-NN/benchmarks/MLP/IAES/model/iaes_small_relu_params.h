#ifndef IAES_SMALL_RELU_PARAMS_H
#define IAES_SMALL_RELU_PARAMS_H

#define NUM_FC_LAYERS 3
#define INPUT_SCALE_F32 0.0078740157f
static const int LAYER_SIZES[] = {12, 32, 64, 1};
static const int32_t FC_MULTIPLIERS[] = {7715, 12490, 15563};
static const int FC_SHIFTS[] = {-20, -20, -20};

#endif
