CC = gcc
CFLAGS = -Wall -O3
LDFLAGS = -lfann -lm

TARGET = nsl_fann
SRCS = nsl_fann.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
