#ifndef CNN_INTERMEDIATE_PARAMS_H
#define CNN_INTERMEDIATE_PARAMS_H

#define IN_CHANNELS 3
#define IN_HEIGHT   32
#define IN_WIDTH    32
#define FC_IN_SIZE  2048
#define FC_OUT_SIZE 2
#define INPUT_SCALE_F32 0.0078740157f
#define NUM_CONV_LAYERS 3

static const int32_t CONV_MULTIPLIERS[] = {3528, 5755, 4675};
static const int CONV_SHIFTS[] = {-20, -20, -20};
#define FC_MULTIPLIER 3256
#define FC_SHIFT -20

#define NUM_LAYER_INFO 3
typedef struct { int type; int in_c, out_c, k, s, p, in_h, in_w, out_h, out_w; } LayerInfo;
static const LayerInfo LAYER_INFO[] = {
    {0, 3, 32, 3, 2, 1, 32, 32, 16, 16},
    {0, 32, 64, 3, 2, 1, 16, 16, 8, 8},
    {0, 64, 128, 3, 2, 1, 8, 8, 4, 4},
};

#endif
