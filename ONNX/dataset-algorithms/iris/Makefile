ORT_DIR ?= ../../onnxruntime-bak

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(ORT_DIR)/include
LDFLAGS  = -L$(ORT_DIR)/lib -lonnxruntime -lm -Wl,-rpath,$(ORT_DIR)/lib

.PHONY: all clean run train

all: train iris_mlp run

train:
	python3 train_iris.py

iris_mlp: iris_mlp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: iris_mlp
	./iris_mlp

clean:
	rm -rf iris_mlp model/iris_mlp.onnx model/iris_mlp.onnx.data
