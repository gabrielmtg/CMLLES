CMSIS_NN = ../../CMSIS-NN-bak

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(CMSIS_NN)/Include -I.
LDFLAGS = -lm

CMSIS_SRC = $(CMSIS_NN)/Source/FullyConnectedFunctions/arm_fully_connected_s8.c \
            $(CMSIS_NN)/Source/ConvolutionFunctions/arm_convolve_s8.c \
            $(CMSIS_NN)/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c \
            $(CMSIS_NN)/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c \
            $(CMSIS_NN)/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c

CONFIGS = simple intermediate mobilenet
TARGETS = $(foreach c,$(CONFIGS),benchmark_cnn2d_$(c))

.PHONY: all clean train

all: $(TARGETS)

train:
	python3 train.py

benchmark_cnn2d_%: benchmark_cnn2d.c model/cnn_%_weights.h model/cnn_%_params.h
	$(CC) $(CFLAGS) \
		-DMODEL_WEIGHTS_H='"model/cnn_$*_weights.h"' \
		-DMODEL_PARAMS_H='"model/cnn_$*_params.h"' \
		-DMODEL_LABEL='"cnn_$*_int8"' \
		-o $@ benchmark_cnn2d.c $(CMSIS_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGETS) model/*.h model/test_data.txt model/training_metrics.json model/*.bin
	rm -rf latencies/
