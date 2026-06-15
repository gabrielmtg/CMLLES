# Iris MLP — CMSIS-NN
#
# Pré-requisito: executar 'python3 train_iris.py' para gerar model/iris_weights.h
#
# CMSIS-NN é otimizado para ARM Cortex-M. Para compilar em x86 (teste),
# as funções usam a implementação C de referência (sem SIMD).

CMSIS_NN = ../../CMSIS-NN

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra \
           -I$(CMSIS_NN)/Include \
           -I.
LDFLAGS  = -lm

# Fontes CMSIS-NN necessárias (fully connected + dependências)
CMSIS_SRC = $(CMSIS_NN)/Source/FullyConnectedFunctions/arm_fully_connected_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c

.PHONY: all clean run train

all: iris_mlp

train:
	python3 train_iris.py

iris_mlp: iris_mlp.c model/iris_weights.h model/iris_parameters.h
	$(CC) $(CFLAGS) -o $@ iris_mlp.c $(CMSIS_SRC) $(LDFLAGS)

run: iris_mlp
	./iris_mlp

clean:
	rm -f iris_mlp
