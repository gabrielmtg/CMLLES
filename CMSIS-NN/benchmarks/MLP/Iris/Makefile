CMSIS_NN = ../../../CMSIS-NN-bak

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(CMSIS_NN)/Include -I.
LDFLAGS = -lm

CMSIS_SRC = $(CMSIS_NN)/Source/FullyConnectedFunctions/arm_fully_connected_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c

SIZES = small medium large
ACTS  = relu sigmoid

TARGETS = $(foreach s,$(SIZES),$(foreach a,$(ACTS),benchmark_mlp_$(s)_$(a)))

.PHONY: all clean train

all: $(TARGETS)

train:
	python3 train.py

benchmark_mlp_%: benchmark_mlp.c model/mlp_%_weights.h model/mlp_%_params.h
	$(CC) $(CFLAGS) \
		-DMODEL_WEIGHTS_H='"model/mlp_$*_weights.h"' \
		-DMODEL_PARAMS_H='"model/mlp_$*_params.h"' \
		-DMODEL_LABEL='"mlp_$*_int8"' \
		-o $@ benchmark_mlp.c $(CMSIS_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGETS) model/*.h model/test_data.txt model/training_metrics.json
	rm -rf latencies/
