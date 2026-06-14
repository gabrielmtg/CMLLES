ORT_DIR ?= ../../../onnxruntime-bak

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(ORT_DIR)/include
LDFLAGS  = -L$(ORT_DIR)/lib -lonnxruntime -lm -Wl,-rpath,$(ORT_DIR)/lib

.PHONY: all clean run train

all: train IAES_monitor run

train:
	python3 train_IAES.py

IAES_monitor: IAES_monitor.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: IAES_monitor
	./IAES_monitor

validate: IAES_monitor
	python3 validate_IAES.py

clean:
	rm -rf IAES_monitor data/iaes_model.onnx data/iaes_model.onnx.data data/iaes_test_data.txt

