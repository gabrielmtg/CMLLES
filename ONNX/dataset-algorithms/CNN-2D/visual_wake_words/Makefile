ORT_DIR ?= ../../../onnxruntime-bak

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(ORT_DIR)/include
LDFLAGS  = -L$(ORT_DIR)/lib -lonnxruntime -lm -Wl,-rpath,$(ORT_DIR)/lib

.PHONY: all clean run train

all: train vww_inference run

train:
	python3 train_vww.py

vww_inference: vww_inference.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: vww_inference
	./vww_inference

validate: vww_inference
	python3 validate_vww.py	

clean:
	rm -rf vww_inference vww_96_model.onnx
