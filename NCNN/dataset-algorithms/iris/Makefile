# Iris MLP — NCNN
#
# Pré-requisito: executar 'python3 train_iris.py' para gerar model/iris_mlp.param e .bin
#
# Ajuste NCNN_DIR para apontar para a instalação do NCNN compilado.

NCNN_DIR ?= ../../ncnn/build

CXX      = g++
CXXFLAGS = -O2 -Wall -Wextra -std=c++17 \
           -I$(NCNN_DIR)/install/include/ncnn \
           -I$(NCNN_DIR)/install/include
LDFLAGS  = -L$(NCNN_DIR)/install/lib -lncnn -lpthread -lm

.PHONY: all clean run train

all: iris_mlp

train:
	python3 train_iris.py

iris_mlp: iris_mlp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: iris_mlp
	./iris_mlp

clean:
	rm -f iris_mlp
