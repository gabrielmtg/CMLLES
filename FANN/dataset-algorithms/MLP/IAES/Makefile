CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lfann -lm

.PHONY: all clean run train_data

all: train_data IAES-monitor-MLP run

IAES-monitor-MLP: IAES_monitor.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

train_data:
	python3 generate_train_data.py

run: IAES-monitor-MLP
	./IAES-monitor-MLP

clean:
	rm -f IAES-monitor-MLP model/IAES_fann.train model/IAES_model.net
