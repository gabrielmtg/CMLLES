TF_SRC_DIR ?= ../../../tensorflow
LITERT_DIR ?= ../../../tensorflow/tflite_build

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(TF_SRC_DIR) -I$(LITERT_DIR)/include
LDFLAGS  = -L$(LITERT_DIR) -ltensorflowlite_c -lm -Wl,-rpath,$(LITERT_DIR)

.PHONY: all clean run train validate

all: train CMAPSS_inference run

train:
	python3 train_CMAPSS.py

CMAPSS_inference: CMAPSS_inference.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run: CMAPSS_inference
	./CMAPSS_inference

validate: CMAPSS_inference
	python3 validate_CMAPSS.py

clean:
	rm -rf CMAPSS_inference model/cmapss_lstm.tflite
