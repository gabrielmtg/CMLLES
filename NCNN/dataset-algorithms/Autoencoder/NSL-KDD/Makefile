NCNN_DIR ?= ../../../ncnn/build/install

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(NCNN_DIR)/include/ncnn -fopenmp
LDFLAGS  = -L$(NCNN_DIR)/lib -lncnn -fopenmp -lstdc++ -lm -Wl,-rpath,$(NCNN_DIR)/lib

.PHONY: all clean run train

all: train nsl run

train:
	python3 train_nsl.py

nsl: nsl.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: nsl
	./nsl

clean:
	rm -rf nsl model/nsl_model.* model/nsl_test_data.txt model/nsl_model_ncnn.py  model/nsl_model_pnnx.py

