CC = gcc
CFLAGS = -Wall -Wextra -std=c11
TARGET = network

SRCS = main.c

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) -lm

.PHONY: clean

clean:
	rm -f $(TARGET)
