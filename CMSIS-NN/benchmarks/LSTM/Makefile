CMSIS_NN = ../../CMSIS-NN

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(CMSIS_NN)/Include -I.
LDFLAGS = -lm

CMSIS_SRC = \
    $(CMSIS_NN)/Source/LSTMFunctions/arm_lstm_unidirectional_s8.c \
    $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_lstm_step_s8.c \
    $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s8_s16.c \
    $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s8_s16.c \
    $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c \
    $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c \
    $(CMSIS_NN)/Source/NNSupportFunctions/arm_nntables.c \
    $(CMSIS_NN)/Source/BasicMathFunctions/arm_elementwise_mul_s16.c \
    $(CMSIS_NN)/Source/BasicMathFunctions/arm_elementwise_mul_acc_s16.c \
    $(CMSIS_NN)/Source/BasicMathFunctions/arm_elementwise_mul_s16_s8.c \
    $(CMSIS_NN)/Source/ActivationFunctions/arm_nn_activation_s16.c \
    $(CMSIS_NN)/Source/FullyConnectedFunctions/arm_fully_connected_s8.c

HIDDEN_SIZES = 32 64 128
TARGETS = $(foreach h,$(HIDDEN_SIZES),benchmark_lstm_h$(h))

.PHONY: all clean train

all: $(TARGETS)

train:
	python3 train.py

benchmark_lstm_h%: benchmark_lstm.c model/lstm_h%_weights.h model/lstm_h%_params.h
	$(CC) $(CFLAGS) \
		-DMODEL_WEIGHTS_H='"model/lstm_h$*_weights.h"' \
		-DMODEL_PARAMS_H='"model/lstm_h$*_params.h"' \
		-DMODEL_LABEL='"lstm_h$*_int8"' \
		-o $@ benchmark_lstm.c $(CMSIS_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGETS) model/*.h model/test_data.txt model/training_metrics.json
	rm -rf latencies/
