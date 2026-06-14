CMSIS_DIR      = /home/dudu/teste/CMLLES/CMSIS-DSP
CMSIS_CORE_DIR = /home/dudu/teste/CMLLES/CMSIS_5/CMSIS/Core

CMSIS_INC      = $(CMSIS_DIR)/Include
CMSIS_CORE_INC = $(CMSIS_CORE_DIR)/Include
CMSIS_LIB      = $(CMSIS_DIR)/build/Source/libCMSISDSP.a 

CC        = gcc
CFLAGS    = -O3 -Wall -Wextra
LDFLAGS_M = -lm

cmapss_gru: cmapss_gru.c
	$(CC) $(CFLAGS) -I$(CMSIS_INC) -I$(CMSIS_CORE_INC) -o $@ cmapss_gru.c $(CMSIS_LIB) $(LDFLAGS_M)

run_cmapss: cmapss_gru
	./cmapss_gru

.PHONY: all clean run_cmapss train

all: cmapss_gru