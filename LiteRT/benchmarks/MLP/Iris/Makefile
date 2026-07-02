TF_SRC_DIR  ?= ../../../tensorflow
LITERT_DIR  ?= $(TF_SRC_DIR)/tflite_build

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(TF_SRC_DIR) -I$(LITERT_DIR)/include
LDFLAGS = -L$(LITERT_DIR) -ltensorflowlite_c -lm -Wl,-rpath,$(LITERT_DIR)

.PHONY: all clean train

all: benchmark_mlp

train:
	python3 train.py

benchmark_mlp: benchmark_mlp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_mlp model/*.tflite model/test_data.txt model/training_metrics.json
	rm -rf latencies/
