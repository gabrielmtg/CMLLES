CC      = gcc
CFLAGS  = -O3 -Wall -Wextra
LDFLAGS = -lm
GENANN  = ../../../genann

.PHONY: all clean run

all: train iaes_genann run

train:
	python3 train_IAES.py

iaes_genann: iaes_genann.c $(GENANN)/genann.c $(GENANN)/genann.h
	$(CC) $(CFLAGS) -I$(GENANN) -o $@ iaes_genann.c $(GENANN)/genann.c $(LDFLAGS)

run: iaes_genann
	./iaes_genann

clean:
	rm -f iaes_genann model/iaes_genann_weights.txt model/iaes_model.onnx \
		model/iaes_model.onnx.data model/iaes_test_data.txt
