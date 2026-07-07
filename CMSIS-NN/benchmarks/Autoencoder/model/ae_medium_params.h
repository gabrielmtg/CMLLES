#ifndef AE_MEDIUM_PARAMS_H
#define AE_MEDIUM_PARAMS_H

#define NUM_FC_LAYERS 6
#define INPUT_SCALE_F32 0.0078740157f
static const int LAYER_SIZES[] = {122, 96, 48, 24, 48, 96, 122};
static const int32_t FC_MULTIPLIERS[] = {13891, 12652, 10117, 5756, 9674, 13482};
static const int FC_SHIFTS[] = {-20, -20, -20, -20, -20, -20};

#endif
