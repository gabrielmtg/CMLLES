# Iris MLP — LiteRT (TFLite Micro)
#
# Pré-requisito: executar 'python3 train_iris.py' para gerar model/iris_model_data.h
#
# Ajuste TFLM_DIR para apontar para o repositório tflite-micro.
# O build system do TFLite Micro é complexo; este Makefile assume
# que a lib foi compilada previamente.

TFLM_DIR  = ../../tflite-micro
TFLM_INC  = $(TFLM_DIR) \
             $(TFLM_DIR)/tensorflow/lite/micro \
             $(TFLM_DIR)/third_party/flatbuffers/include \
             $(TFLM_DIR)/third_party/gemmlowp \
             $(TFLM_DIR)/third_party/ruy

CXX       = g++
CXXFLAGS  = -O2 -Wall -Wextra -std=c++17 \
            $(addprefix -I, $(TFLM_INC)) \
            -I.
LDFLAGS   = -lm -lpthread

# Nota: a compilação completa do TFLite Micro requer o build system
# do projeto (Bazel ou CMake). Este Makefile é um template.
# Para compilar, siga as instruções em:
# https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro

.PHONY: all clean run train

all:
	@echo "NOTA: TFLite Micro requer build via Bazel ou CMake do projeto."
	@echo "Este Makefile é um template. Use:"
	@echo "  1. python3 train_iris.py  (gerar modelo)"
	@echo "  2. Integrar iris_mlp.cpp no build do tflite-micro"

train:
	python3 train_iris.py

clean:
	rm -f iris_mlp
