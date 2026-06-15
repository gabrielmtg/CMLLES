NCNN_DIR ?= ../../../ncnn/build/install

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(NCNN_DIR)/include/ncnn -fopenmp
LDFLAGS  = -L$(NCNN_DIR)/lib -lncnn -fopenmp -lstdc++ -lm -Wl,-rpath,$(NCNN_DIR)/lib

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
	rm -rf vww_inference imagem_teste.bin model/vww_96_model.ncnn.bin model/vww_96_model_ncnn.py \
		model/vww_96_model.pnnx.onnx model/vww_96_model_pnnx.py model/vww_96_model.ncnn.param \
		model/vww_96_model.pnnx.bin model/vww_96_model.pnnx.param model/vww_96_model.pt model/__pycache__
