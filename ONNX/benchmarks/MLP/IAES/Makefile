ORT_DIR ?= ../../../onnxruntime-bak

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(ORT_DIR)/include
LDFLAGS = -L$(ORT_DIR)/lib -lonnxruntime -lm -Wl,-rpath,$(ORT_DIR)/lib

.PHONY: all clean train

all: benchmark_iaes

train:
	python3 train.py

benchmark_iaes: benchmark_iaes.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_iaes model/*.onnx model/test_data.txt model/training_metrics.json
	rm -rf latencies/
