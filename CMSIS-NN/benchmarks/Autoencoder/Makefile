CMSIS_NN = ../../CMSIS-NN-bak

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(CMSIS_NN)/Include -I.
LDFLAGS = -lm

CMSIS_SRC = $(CMSIS_NN)/Source/FullyConnectedFunctions/arm_fully_connected_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c

SIZES = small medium large
TARGETS = $(foreach s,$(SIZES),benchmark_ae_$(s))

.PHONY: all clean train

all: $(TARGETS)

train:
	python3 train.py

benchmark_ae_%: benchmark_ae.c model/ae_%_weights.h model/ae_%_params.h
	$(CC) $(CFLAGS) \
		-DMODEL_WEIGHTS_H='"model/ae_$*_weights.h"' \
		-DMODEL_PARAMS_H='"model/ae_$*_params.h"' \
		-DMODEL_LABEL='"ae_$*_int8"' \
		-DMODEL_AE_SIZE='"$*"' \
		-o $@ benchmark_ae.c $(CMSIS_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGETS) model/*.h model/test_data.txt model/training_metrics.json model/*_threshold.txt
	rm -rf latencies/
