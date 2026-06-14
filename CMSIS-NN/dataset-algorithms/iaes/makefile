CMSIS_DIR      = /home/dudu/teste/CMLLES/CMSIS-DSP
CMSIS_CORE_DIR = /home/dudu/teste/CMLLES/CMSIS_5/CMSIS/Core

# Rotas de inclusão
CMSIS_INC      = $(CMSIS_DIR)/Include
CMSIS_CORE_INC = $(CMSIS_CORE_DIR)/Include

# Caminho da biblioteca estática recém-compilada
CMSIS_LIB      = $(CMSIS_DIR)/build/Source/libCMSISDSP.a 

# Regras de compilação
iaes_cmsis: iaes_cmsis.c
	$(CC) $(CFLAGS) -I$(CMSIS_INC) -I$(CMSIS_CORE_INC) -o $@ iaes_cmsis.c $(CMSIS_LIB) -lm

run_cmsis: iaes_cmsis
	./iaes_cmsis