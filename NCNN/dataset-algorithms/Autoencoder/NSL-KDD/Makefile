NCNN_DIR ?= ../../../ncnn/build/install

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(NCNN_DIR)/include/ncnn -fopenmp
LDFLAGS  = -L$(NCNN_DIR)/lib -lncnn -fopenmp -lstdc++ -lm -Wl,-rpath,$(NCNN_DIR)/lib

.PHONY: all clean run train validate

all: train nsl run

train:
	python3 train_nsl.py

nsl: nsl.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run: nsl
	./nsl

validate: nsl
	python3 validate_nsl.py

clean:
	rm -rf nsl model/nsl_test_data.txt model/nsl_model.ncnn.bin model/nsl_model_ncnn.py \
		model/nsl_model.pnnx.onnx model/nsl_model_pnnx.py model/nsl_model.ncnn.param \
		model/nsl_model.pnnx.bin model/nsl_model.pnnx.param model/nsl_model.pt model/__pycache__
