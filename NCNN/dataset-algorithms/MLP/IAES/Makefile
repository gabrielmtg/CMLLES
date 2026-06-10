NCNN_DIR ?= ../../../ncnn/build/install

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(NCNN_DIR)/include/ncnn -fopenmp
LDFLAGS  = -L$(NCNN_DIR)/lib -lncnn -fopenmp -lstdc++ -lm -Wl,-rpath,$(NCNN_DIR)/lib

.PHONY: all clean run train

all: train IAES_monitor run

train:
	python3 train_IAES_monitor.py

IAES_monitor: IAES_monitor.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: IAES_monitor
	./IAES_monitor

clean:
	rm -rf IAES_monitor model/iaes_model.ncnn.bin model/iaes_model_ncnn.py model/iaes_model.pnnx.onnx model/iaes_model_pnnx.py model/iaes_test_data.txt \
		model/iaes_model.ncnn.param model/iaes_model.pnnx.bin model/iaes_model.pnnx.param model/iaes_model.pt model/__pycache__

