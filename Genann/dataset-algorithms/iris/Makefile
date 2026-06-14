# Iris MLP — Genann
CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lm
GENANN  = ../../genann

.PHONY: all clean run

all: iris_mlp

iris_mlp: iris_mlp.c $(GENANN)/genann.c $(GENANN)/genann.h
	$(CC) $(CFLAGS) -I$(GENANN) -o $@ iris_mlp.c $(GENANN)/genann.c $(LDFLAGS)

run: iris_mlp
	./iris_mlp

clean:
	rm -f iris_mlp
