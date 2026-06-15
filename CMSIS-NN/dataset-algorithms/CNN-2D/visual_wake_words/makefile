CMSIS_DIR      = /home/dudu/teste/CMLLES/CMSIS-DSP
CMSIS_CORE_DIR = /home/dudu/teste/CMLLES/CMSIS_5/CMSIS/Core

CMSIS_INC      = $(CMSIS_DIR)/Include
CMSIS_CORE_INC = $(CMSIS_CORE_DIR)/Include
CMSIS_LIB      = $(CMSIS_DIR)/build/Source/libCMSISDSP.a 

CC        = gcc
CFLAGS    = -O3 -Wall -Wextra
LDFLAGS_M = -lm

vww_cnn: vww_cnn.c
	$(CC) $(CFLAGS) -I$(CMSIS_INC) -I$(CMSIS_CORE_INC) -o $@ vww_cnn.c $(CMSIS_LIB) $(LDFLAGS_M)

	