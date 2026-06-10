NCNN_DIR ?= ../../../ncnn/build/install

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -I$(NCNN_DIR)/include/ncnn -fopenmp
LDFLAGS  = -L$(NCNN_DIR)/lib -lncnn -fopenmp -lstdc++ -lm -Wl,-rpath,$(NCNN_DIR)/lib

#CXX      = g++
#CXXFLAGS = -O2 -Wall -Wextra -I$(NCNN_DIR)/include/ncnn -fopenmp
#LDFLAGS  = -L$(NCNN_DIR)/lib -lncnn -fopenmp -lstdc++ -lm -Wl,-rpath,$(NCNN_DIR)/lib

.PHONY: all clean run train

all: train iris_mlp run

train:
	python3 train_iris.py

iris_mlp: iris_mlp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: iris_mlp
	./iris_mlp

clean:
	rm -rf iris_mlp model/iris_mlp.bin model/iris_mlp.onnx.data model/iris_model.ncnn.bin \
		model/iris_model_ncnn.py model/iris_model.pnnx.onnx model/iris_model_pnnx.py \
		model/__pycache__ model/iris_mlp.onnx model/iris_mlp.param model/iris_model.ncnn.param \
		model/iris_model.pnnx.bin model/iris_model.pnnx.param model/iris_model.pt \
		model/iris_mlp.ncnn.bin model/iris_mlp.ncnn.param model/iris_mlp_ncnn.py model/iris_mlp.pnnx.bin \
		model/iris_mlp.pnnx.onnx model/iris_mlp.pnnx.param model/iris_mlp_pnnx.py model/iris_mlp.pt


