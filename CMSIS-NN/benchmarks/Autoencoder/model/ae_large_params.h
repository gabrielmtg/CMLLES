#ifndef AE_LARGE_PARAMS_H
#define AE_LARGE_PARAMS_H

#define NUM_FC_LAYERS 6
#define INPUT_SCALE_F32 0.0078740157f
static const int LAYER_SIZES[] = {122, 128, 64, 32, 64, 128, 122};
static const int32_t FC_MULTIPLIERS[] = {10757, 9972, 7499, 7758, 9992, 14538};
static const int FC_SHIFTS[] = {-20, -20, -20, -20, -20, -20};

#endif
