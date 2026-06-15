# ORT_DIR ?= ../../onnxruntime-bak

# CC       = gcc
# CFLAGS   = -O2 -Wall -Wextra -I$(ORT_DIR)/include
# LDFLAGS  = -L$(ORT_DIR)/lib -lonnxruntime -lm -Wl,-rpath,$(ORT_DIR)/lib

# .PHONY: all clean run train

# all: train IAES_monitor run

# train:
# 	python3 train_IAES.py

# IAES_monitor: IAES_monitor.c
# 	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# run: IAES_monitor
# 	./IAES_monitor

# clean:
# 	rm -rf IAES_monitor data/iaes_model.onnx data/iaes_model.onnx.data data/iaes_test_data.txt

# Makefile para o IAES no Genann
CC      = gcc
CFLAGS  = -O3 -Wall -Wextra
LDFLAGS = -lm
GENANN  = ../../genann

.PHONY: all clean run

all: iaes_genann

iaes_genann: iaes_genann.c $(GENANN)/genann.c $(GENANN)/genann.h
	$(CC) $(CFLAGS) -I$(GENANN) -o $@ iaes_genann.c $(GENANN)/genann.c $(LDFLAGS)

run: iaes_genann
	./iaes_genann

clean:
	rm -f iaes_genann