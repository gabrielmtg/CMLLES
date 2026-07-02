NCNN_DIR ?= ../../ncnn/build/install

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(NCNN_DIR)/include/ncnn -fopenmp
LDFLAGS = -L$(NCNN_DIR)/lib -lncnn -fopenmp -lstdc++ -lm -Wl,-rpath,$(NCNN_DIR)/lib

.PHONY: all clean train

all: benchmark_ae

train:
	python3 train.py

benchmark_ae: benchmark_ae.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_ae model/*.param model/*.bin model/*.pt model/test_data.txt model/training_metrics.json model/*_threshold.txt
	rm -rf latencies/
