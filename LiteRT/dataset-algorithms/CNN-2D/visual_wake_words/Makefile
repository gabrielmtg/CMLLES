TF_SRC_DIR ?= ../../../tensorflow
LITERT_DIR ?= ../../../tensorflow/tflite_build

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(TF_SRC_DIR) -I$(LITERT_DIR)/include
LDFLAGS  = -L$(LITERT_DIR) -ltensorflowlite_c -lm -Wl,-rpath,$(LITERT_DIR)

.PHONY: all clean run train validate

all: train vww_inference run

train:
	python3 train_vww.py
	python3 export_image.py

vww_inference: vww_inference.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run: vww_inference
	./vww_inference

validate: vww_inference
	python3 validate_vww.py	

clean:
	rm -rf vww_inference model/vww_96_model.tflite imagem_teste.bin
