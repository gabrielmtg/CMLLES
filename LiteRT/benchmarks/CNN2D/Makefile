TF_SRC_DIR  ?= ../../tensorflow
LITERT_DIR  ?= $(TF_SRC_DIR)/tflite_build

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(TF_SRC_DIR) -I$(LITERT_DIR)/include
LDFLAGS = -L$(LITERT_DIR) -ltensorflowlite_c -lm -Wl,-rpath,$(LITERT_DIR)

.PHONY: all clean train

all: benchmark_cnn2d

train:
	python3 train.py

benchmark_cnn2d: benchmark_cnn2d.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_cnn2d model/*.tflite model/test_data.txt model/training_metrics.json model/*.bin
	rm -rf latencies/
