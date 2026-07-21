CMSIS_NN = ../../../CMSIS-NN

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(CMSIS_NN)/Include -I.
LDFLAGS = -lm

CMSIS_SRC = $(CMSIS_NN)/Source/FullyConnectedFunctions/arm_fully_connected_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c

SIZES = small medium large
ACTS  = relu sigmoid

TARGETS = $(foreach s,$(SIZES),$(foreach a,$(ACTS),benchmark_iaes_$(s)_$(a)))

.PHONY: all clean train

all: $(TARGETS)

train:
	python3 train.py

benchmark_iaes_%: benchmark_iaes.c model/iaes_%_weights.h model/iaes_%_params.h
	$(CC) $(CFLAGS) \
		-DMODEL_WEIGHTS_H='"model/iaes_$*_weights.h"' \
		-DMODEL_PARAMS_H='"model/iaes_$*_params.h"' \
		-DMODEL_LABEL='"iaes_$*_int8"' \
		-o $@ benchmark_iaes.c $(CMSIS_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGETS) model/*.h model/test_data.txt model/training_metrics.json
	rm -rf latencies/
