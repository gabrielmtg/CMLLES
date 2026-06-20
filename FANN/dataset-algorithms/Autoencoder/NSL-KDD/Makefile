CC = gcc
CFLAGS = -Wall -O3
LDFLAGS = -lfann -lm

TARGET = nsl_fann
SRCS = nsl_fann.c

all: data $(TARGET) run

data:
	python3 prepare_data.py

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

run:
	./$(TARGET)

clean:
	rm -f $(TARGET) model/nsl_model.net model/nsl_test.txt model/nsl_train.data
