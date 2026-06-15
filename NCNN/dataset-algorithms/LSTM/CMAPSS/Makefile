NCNN_DIR ?= ../../../ncnn/build/install

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(NCNN_DIR)/include/ncnn -fopenmp
LDFLAGS  = -L$(NCNN_DIR)/lib -lncnn -fopenmp -lstdc++ -lm -Wl,-rpath,$(NCNN_DIR)/lib

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
	rm -rf CMAPSS_inference model/cmapss_lstm.ncnn.bin model/cmapss_lstm_ncnn.py \
		model/cmapss_lstm.pnnx.onnx model/cmapss_lstm_pnnx.py model/cmapss_lstm.ncnn.param \
		model/cmapss_lstm.pnnx.bin model/cmapss_lstm.pnnx.param model/cmapss_lstm.pt model/__pycache__
