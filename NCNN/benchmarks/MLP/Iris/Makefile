NCNN_DIR ?= ../../../ncnn/build/install

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(NCNN_DIR)/include/ncnn -fopenmp
LDFLAGS = -L$(NCNN_DIR)/lib -lncnn -fopenmp -lstdc++ -lm -Wl,-rpath,$(NCNN_DIR)/lib

.PHONY: all clean train

all: benchmark_mlp

train:
	python3 train.py

benchmark_mlp: benchmark_mlp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_mlp model/*.param model/*.bin model/*.pt model/test_data.txt model/training_metrics.json
	rm -rf latencies/
