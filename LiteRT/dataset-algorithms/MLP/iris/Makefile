TF_SRC_DIR ?= ../../../tensorflow

LITERT_DIR ?= $(TF_SRC_DIR)/tflite_build

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(TF_SRC_DIR) -I$(LITERT_DIR)/include
LDFLAGS  = -L$(LITERT_DIR) -ltensorflowlite_c -lm -Wl,-rpath,$(LITERT_DIR)

.PHONY: all clean run train

all: train iris_mlp run

train:
	python3 train_iris.py

iris_mlp: iris_mlp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: iris_mlp
	./iris_mlp

clean:
	rm -f iris_mlp model/iris_mlp.tflite
