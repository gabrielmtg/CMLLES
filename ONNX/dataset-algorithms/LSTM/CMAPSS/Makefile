ORT_DIR ?= ../../../onnxruntime-bak

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(ORT_DIR)/include
LDFLAGS  = -L$(ORT_DIR)/lib -lonnxruntime -lm -Wl,-rpath,$(ORT_DIR)/lib

.PHONY: all clean run train

all: train CMAPSS_inference run

train:
	python3 train_CMAPSS.py

CMAPSS_inference: CMAPSS_inference.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: CMAPSS_inference
	./CMAPSS_inference

validate: CMAPSS_inference
	python3 validate_CMAPSS.py

clean:
	rm -rf CMAPSS_inference model/cmapss_lstm.onnx model/cmapss_lstm.onnx.data
