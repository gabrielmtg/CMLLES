ORT_DIR ?= ../../../onnxruntime-bak

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(ORT_DIR)/include
LDFLAGS  = -L$(ORT_DIR)/lib -lonnxruntime -lm -Wl,-rpath,$(ORT_DIR)/lib

.PHONY: all clean run train

all: train nsl run

train:
	python3 train_nsl.py

nsl: nsl.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: nsl
	./nsl

clean:
	rm -rf nsl model/nsl_model.onnx model/nsl_model.onnx.data model/nsl_test_data.txt
