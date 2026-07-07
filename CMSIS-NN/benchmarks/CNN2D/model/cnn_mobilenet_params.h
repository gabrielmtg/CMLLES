#ifndef CNN_MOBILENET_PARAMS_H
#define CNN_MOBILENET_PARAMS_H

#define IN_CHANNELS 3
#define IN_HEIGHT   96
#define IN_WIDTH    96
#define FC_IN_SIZE  4608
#define FC_OUT_SIZE 2
#define INPUT_SCALE_F32 0.0078740157f
#define NUM_CONV_LAYERS 4

static const int32_t CONV_MULTIPLIERS[] = {2984, 6337, 4929, 5147};
static const int CONV_SHIFTS[] = {-20, -20, -20, -20};
#define FC_MULTIPLIER 2967
#define FC_SHIFT -20

#define NUM_LAYER_INFO 4
typedef struct { int type; int in_c, out_c, k, s, p, in_h, in_w, out_h, out_w; } LayerInfo;
static const LayerInfo LAYER_INFO[] = {
    {0, 3, 32, 3, 2, 1, 96, 96, 48, 48},
    {0, 32, 64, 3, 2, 1, 48, 48, 24, 24},
    {0, 64, 128, 3, 2, 1, 24, 24, 12, 12},
    {0, 128, 128, 3, 2, 1, 12, 12, 6, 6},
};

#endif
