NCNN_DIR ?= ../../ncnn/build/install

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(NCNN_DIR)/include/ncnn -fopenmp
LDFLAGS = -L$(NCNN_DIR)/lib -lncnn -fopenmp -lstdc++ -lm -Wl,-rpath,$(NCNN_DIR)/lib

.PHONY: all clean train

all: benchmark_cnn2d

train:
	python3 train.py

benchmark_cnn2d: benchmark_cnn2d.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_cnn2d model/*.param model/*.bin model/*.pt model/test_data.txt model/training_metrics.json model/*.bin
	rm -rf latencies/
