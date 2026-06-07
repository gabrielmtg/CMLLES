CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lfann -lm

.PHONY: all clean run

all: IAES-monitor-MLP

IAES-monitor-MLP: IAES_monitor.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: IAES-monitor-MLP
	./IAES-monitor-MLP

clean:
	rm -f IAES-monitor-MLP IAES_fann.train IAES_model.net
