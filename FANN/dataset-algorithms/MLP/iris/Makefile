CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lfann -lm

.PHONY: all clean run

all: iris_mlp run

iris_mlp: iris_mlp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: iris_mlp
	./iris_mlp

clean:
	rm -f iris_mlp model/iris_fann.train model/iris_model.net
