# Backend CMSIS-DSP
CMSIS_DIR      = /home/dudu/teste/CMLLES/CMSIS-DSP
CMSIS_CORE_DIR = /home/dudu/teste/CMLLES/CMSIS_5/CMSIS/Core
# Rotas de inclusão
CMSIS_INC      = $(CMSIS_DIR)/Include
CMSIS_CORE_INC = $(CMSIS_CORE_DIR)/Include

# Caminho da biblioteca estática recém-compilada
CMSIS_LIB      = $(CMSIS_DIR)/build/Source/libCMSISDSP.a 
nsl_kdd_cmsis: nsl_kdd_cmsis.c
	$(CC) $(CFLAGS) -I$(CMSIS_INC) -I$(CMSIS_CORE_INC) -o $@ nsl_kdd_cmsis.c $(CMSIS_LIB) -lm

run_cmsis: nsl_kdd_cmsis
	./nsl_kdd_cmsis

clean:
	rm -f nsl_kdd_genann nsl_kdd_cmsis
	rm -rf model/nsl_autoencoder.onnx model/nsl_test_data.txt model/nsl_genann_weights.txt model/nsl_cmsis_weights.txt \
		model/nsl_autoencoder.onnx.data
