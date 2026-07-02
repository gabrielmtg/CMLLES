.PHONY: all benchmarks train \
        bench-fann bench-genann bench-ncnn bench-litert bench-onnx bench-cmsisnn \
        train-fann train-genann train-ncnn train-litert train-onnx train-cmsisnn

all: benchmarks

benchmarks: bench-fann bench-genann bench-ncnn bench-litert bench-onnx bench-cmsisnn
	@echo "All benchmarks built!"

train: train-fann train-genann train-ncnn train-litert train-onnx train-cmsisnn
	@echo "All models trained!"

bench-fann:
	@echo "--- Building FANN Benchmarks ---"
	$(MAKE) -C FANN/benchmarks/MLP all
	$(MAKE) -C FANN/benchmarks/Autoencoder all

bench-genann:
	@echo "--- Building Genann Benchmarks ---"
	$(MAKE) -C Genann/benchmarks/MLP all
	$(MAKE) -C Genann/benchmarks/Autoencoder all

bench-ncnn:
	@echo "--- Building NCNN Benchmarks ---"
	$(MAKE) -C NCNN/benchmarks/MLP all
	$(MAKE) -C NCNN/benchmarks/Autoencoder all
	$(MAKE) -C NCNN/benchmarks/CNN2D all
	$(MAKE) -C NCNN/benchmarks/LSTM all

bench-litert:
	@echo "--- Building LiteRT Benchmarks ---"
	$(MAKE) -C LiteRT/benchmarks/MLP all
	$(MAKE) -C LiteRT/benchmarks/Autoencoder all
	$(MAKE) -C LiteRT/benchmarks/CNN2D all
	$(MAKE) -C LiteRT/benchmarks/LSTM all

bench-onnx:
	@echo "--- Building ONNX Benchmarks ---"
	$(MAKE) -C ONNX/benchmarks/MLP all
	$(MAKE) -C ONNX/benchmarks/Autoencoder all
	$(MAKE) -C ONNX/benchmarks/CNN2D all
	$(MAKE) -C ONNX/benchmarks/LSTM all

bench-cmsisnn:
	@echo "--- Building CMSIS-NN Benchmarks ---"
	$(MAKE) -C CMSIS-NN/benchmarks/MLP all
	$(MAKE) -C CMSIS-NN/benchmarks/Autoencoder all
	$(MAKE) -C CMSIS-NN/benchmarks/CNN2D all
	$(MAKE) -C CMSIS-NN/benchmarks/LSTM all

train-fann:
	@echo "--- Training FANN Models ---"
	$(MAKE) -C FANN/benchmarks/MLP train
	$(MAKE) -C FANN/benchmarks/Autoencoder train

train-genann:
	@echo "--- Training Genann Models ---"
	$(MAKE) -C Genann/benchmarks/MLP train
	$(MAKE) -C Genann/benchmarks/Autoencoder train

train-ncnn:
	@echo "--- Training NCNN Models ---"
	$(MAKE) -C NCNN/benchmarks/MLP train
	$(MAKE) -C NCNN/benchmarks/Autoencoder train
	$(MAKE) -C NCNN/benchmarks/CNN2D train
	$(MAKE) -C NCNN/benchmarks/LSTM train

train-litert:
	@echo "--- Training LiteRT Models ---"
	$(MAKE) -C LiteRT/benchmarks/MLP train
	$(MAKE) -C LiteRT/benchmarks/Autoencoder train
	$(MAKE) -C LiteRT/benchmarks/CNN2D train
	$(MAKE) -C LiteRT/benchmarks/LSTM train

train-onnx:
	@echo "--- Training ONNX Models ---"
	$(MAKE) -C ONNX/benchmarks/MLP train
	$(MAKE) -C ONNX/benchmarks/Autoencoder train
	$(MAKE) -C ONNX/benchmarks/CNN2D train
	$(MAKE) -C ONNX/benchmarks/LSTM train

train-cmsisnn:
	@echo "--- Training CMSIS-NN Models ---"
	$(MAKE) -C CMSIS-NN/benchmarks/MLP train
	$(MAKE) -C CMSIS-NN/benchmarks/Autoencoder train
	$(MAKE) -C CMSIS-NN/benchmarks/CNN2D train
	$(MAKE) -C CMSIS-NN/benchmarks/LSTM train
