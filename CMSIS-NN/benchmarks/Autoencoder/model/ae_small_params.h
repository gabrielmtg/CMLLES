#ifndef AE_SMALL_PARAMS_H
#define AE_SMALL_PARAMS_H

#define NUM_FC_LAYERS 6
#define INPUT_SCALE_F32 0.0078740157f
static const int LAYER_SIZES[] = {122, 64, 32, 16, 32, 64, 122};
static const int32_t FC_MULTIPLIERS[] = {12699, 13650, 11943, 8100, 15827, 24402};
static const int FC_SHIFTS[] = {-20, -20, -20, -20, -20, -20};

#endif
