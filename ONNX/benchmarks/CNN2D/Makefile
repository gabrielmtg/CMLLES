ORT_DIR ?= ../../onnxruntime-bak

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(ORT_DIR)/include
LDFLAGS = -L$(ORT_DIR)/lib -lonnxruntime -lm -Wl,-rpath,$(ORT_DIR)/lib

.PHONY: all clean train

all: benchmark_cnn2d

train:
	python3 train.py

benchmark_cnn2d: benchmark_cnn2d.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_cnn2d model/*.onnx model/*.bin model/training_metrics.json
	rm -rf latencies/
