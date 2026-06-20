TF_SRC_DIR ?= ../../../tensorflow

LITERT_DIR ?= ../../../tensorflow/tflite_build

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(TF_SRC_DIR) -I$(LITERT_DIR)/include
LDFLAGS  = -L$(LITERT_DIR) -ltensorflowlite_c -lm -Wl,-rpath,$(LITERT_DIR)

.PHONY: all clean run train

all: train IAES_monitor run

train:
	python3 train_IAES.py

IAES_monitor: IAES_monitor.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run: IAES_monitor
	./IAES_monitor

clean:
	rm -f IAES_monitor model/IAES_monitor.tflite model/iaes_model.tflite model/iaes_test_data.txt

