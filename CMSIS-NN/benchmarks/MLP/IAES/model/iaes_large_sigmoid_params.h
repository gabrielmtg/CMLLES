#ifndef IAES_LARGE_SIGMOID_PARAMS_H
#define IAES_LARGE_SIGMOID_PARAMS_H

#define NUM_FC_LAYERS 5
#define INPUT_SCALE_F32 0.0078740157f
static const int LAYER_SIZES[] = {12, 128, 256, 128, 64, 1};
static const int32_t FC_MULTIPLIERS[] = {19899, 10939, 22019, 7414, 3297};
static const int FC_SHIFTS[] = {-20, -20, -20, -20, -20};

#endif
