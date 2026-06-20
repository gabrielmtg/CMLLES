CC      = gcc
CFLAGS  = -O3 -Wall -Wextra
LDFLAGS = -lm
GENANN  = ../../../genann

.PHONY: all clean run

all: train nsl_kdd_genann run

train:
	python3 train_nslkdd.py

nsl_kdd_genann: nsl_kdd_genann.c $(GENANN)/genann.c $(GENANN)/genann.h
	$(CC) $(CFLAGS) -I$(GENANN) -o $@ nsl_kdd_genann.c $(GENANN)/genann.c $(LDFLAGS)

run: nsl_kdd_genann
	./nsl_kdd_genann

clean:
	rm -f nsl_kdd_genann model/nsl_autoencoder.onnx model/nsl_autoencoder.onnx.data \
		model/nsl_genann_weights.txt model/nsl_test_data.txt
