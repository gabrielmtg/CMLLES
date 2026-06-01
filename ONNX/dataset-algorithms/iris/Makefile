# Iris MLP — ONNX Runtime
#
# Pré-requisito: executar 'python3 train_iris.py' para gerar model/iris_mlp.onnx
#
# Ajuste ORT_DIR para apontar para sua instalação do ONNX Runtime.
# Download: https://github.com/microsoft/onnxruntime/releases

ORT_DIR ?= ../../onnxruntime

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(ORT_DIR)/include
LDFLAGS  = -L$(ORT_DIR)/lib -lonnxruntime -lm -Wl,-rpath,$(ORT_DIR)/lib

.PHONY: all clean run train

all: iris_mlp

train:
	python3 train_iris.py

iris_mlp: iris_mlp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: iris_mlp
	./iris_mlp

clean:
	rm -f iris_mlp
