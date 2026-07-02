TF_SRC_DIR  ?= ../../tensorflow
LITERT_DIR  ?= $(TF_SRC_DIR)/tflite_build

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(TF_SRC_DIR) -I$(LITERT_DIR)/include
LDFLAGS = -L$(LITERT_DIR) -ltensorflowlite_c -lm -Wl,-rpath,$(LITERT_DIR)

.PHONY: all clean train

all: benchmark_ae

train:
	python3 train.py

benchmark_ae: benchmark_ae.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_ae model/*.tflite model/test_data.txt model/training_metrics.json model/*_threshold.txt
	rm -rf latencies/
