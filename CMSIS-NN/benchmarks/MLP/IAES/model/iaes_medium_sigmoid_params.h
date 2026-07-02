#ifndef IAES_MEDIUM_SIGMOID_PARAMS_H
#define IAES_MEDIUM_SIGMOID_PARAMS_H

#define NUM_FC_LAYERS 4
#define INPUT_SCALE_F32 0.0078740157f
static const int LAYER_SIZES[] = {12, 64, 128, 64, 1};
static const int32_t FC_MULTIPLIERS[] = {20901, 11109, 10524, 5807};
static const int FC_SHIFTS[] = {-20, -20, -20, -20};

#endif
