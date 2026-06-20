CMSIS_DIR = ../../../CMSIS-NN-bak

CMSIS_INC = $(CMSIS_DIR)/Include
CMSIS_LIB = $(CMSIS_DIR)/libcmsis-nn.a

CMSIS_DSP_DIR = ../../../CMSIS-DSP
CMSIS_DSP_INC = $(CMSIS_DSP_DIR)/Include

CMSIS_CORE_DIR = ../../../CMSIS_5/CMSIS/Core
CMSIS_CORE_INC = $(CMSIS_CORE_DIR)/Include

CC        = gcc
CFLAGS    = -O3 -Wall -Wextra
LDFLAGS_M = -lm

.PHONY: all clean run_cmapss train

all: train cmapss_gru run_cmapss

train:
	python3 train_cmapss.py

cmapss_gru: cmapss_gru.c
	$(CC) $(CFLAGS) -I$(CMSIS_INC) -I$(CMSIS_DSP_INC) -I$(CMSIS_CORE_INC) -o $@ cmapss_gru.c $(CMSIS_LIB) $(LDFLAGS_M)

run_cmapss: cmapss_gru
	./cmapss_gru

clean:
	rm -rf cmapss_gru model/cmapss_cmsis_weights.txt  model/cmapss_test_data.txt




