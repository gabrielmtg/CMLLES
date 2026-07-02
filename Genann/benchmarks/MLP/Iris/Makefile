CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lm
GENANN  = ../../../genann

.PHONY: all clean train prepare

all: benchmark_mlp

prepare:
	python3 prepare_data.py

train: prepare train_models
	./train_models

train_models: train_models.c $(GENANN)/genann.c
	$(CC) $(CFLAGS) -I$(GENANN) -o $@ train_models.c $(GENANN)/genann.c $(LDFLAGS)

benchmark_mlp: benchmark_mlp.c $(GENANN)/genann.c
	$(CC) $(CFLAGS) -I$(GENANN) -o $@ benchmark_mlp.c $(GENANN)/genann.c $(LDFLAGS)

clean:
	rm -f benchmark_mlp train_models model/*.genann model/iris_genann_train.data
	rm -f model/test_data.txt model/training_metrics.json
	rm -rf latencies/
