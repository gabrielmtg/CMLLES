TF_SRC_DIR ?= ../../../tensorflow
LITERT_DIR ?= ../../../tensorflow/tflite_build

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(TF_SRC_DIR) -I$(LITERT_DIR)/include
LDFLAGS  = -L$(LITERT_DIR) -ltensorflowlite_c -lm -Wl,-rpath,$(LITERT_DIR)

.PHONY: all clean run train validate

all: train nsl run validate

train:
	python3 train_nsl.py

nsl: nsl.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run: nsl
	./nsl

validate: nsl
	python3 validate_nsl.py

clean:
	rm -rf nsl model/nsl_model.tflite model/nsl_test_data.txt
