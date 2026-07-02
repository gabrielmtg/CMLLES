FANN_DIR ?= ../../../fann/build/install

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I$(FANN_DIR)/include
LDFLAGS = -L$(FANN_DIR)/lib -lfann -lm -Wl,-rpath,$(FANN_DIR)/lib

.PHONY: all clean train prepare

all: benchmark_iaes

prepare:
	python3 prepare_data.py

train: prepare train_models
	./train_models

train_models: train_models.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

benchmark_iaes: benchmark_iaes.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_iaes train_models model/*.net model/fann_train.data
	rm -f model/test_data.txt model/training_metrics.json
	rm -rf latencies/
