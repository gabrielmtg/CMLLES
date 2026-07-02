ORT_DIR ?= ../../../onnxruntime-bak

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(ORT_DIR)/include
LDFLAGS = -L$(ORT_DIR)/lib -lonnxruntime -lm -Wl,-rpath,$(ORT_DIR)/lib

.PHONY: all clean train

all: benchmark_mlp

train:
	python3 train.py

benchmark_mlp: benchmark_mlp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_mlp model/*.onnx model/test_data.txt model/training_metrics.json
	rm -rf latencies/
