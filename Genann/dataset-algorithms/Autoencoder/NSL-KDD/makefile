CC      = gcc
CFLAGS  = -O3 -Wall -Wextra
LDFLAGS = -lm
GENANN  = ../../genann

.PHONY: all clean run

all: nsl_kdd_genann

nsl_kdd_genann: nsl_kdd_genann.c $(GENANN)/genann.c $(GENANN)/genann.h
	$(CC) $(CFLAGS) -I$(GENANN) -o $@ nsl_kdd_genann.c $(GENANN)/genann.c $(LDFLAGS)

run: nsl_kdd_genann
	./nsl_kdd_genann

clean:
	rm -f nsl_kdd_genann