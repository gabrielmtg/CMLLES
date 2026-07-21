#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/../env_riscv.sh"

echo "============================================================="
echo " Cross-Compiling CMLLES Benchmarks for RISC-V (riscv64)"
echo "============================================================="


GLIBC_COMPAT_OBJ="${LIBS_PREFIX}/glibc_compat.o"
"${RV_CROSS_CC}" -fPIC -c -U_GNU_SOURCE -U_ISOC23_SOURCE -D_POSIX_C_SOURCE=200809L \
    "${SCRIPT_DIR}/glibc_compat.c" -o "${GLIBC_COMPAT_OBJ}"

export CC="$RV_CROSS_CC -I${LIBS_PREFIX}/include -I${LIBS_PREFIX}/include/ncnn -L${LIBS_PREFIX}/lib ${GLIBC_COMPAT_OBJ}"
export CXX="$RV_CROSS_CXX -I${LIBS_PREFIX}/include -I${LIBS_PREFIX}/include/ncnn -L${LIBS_PREFIX}/lib ${GLIBC_COMPAT_OBJ}"

mkdir -p "$EXEC_DIR"

compile_and_copy() {
    local dir=$1
    local target=$2
    local output_name=$3

    echo "Compiling $output_name in $dir..."
    rm -f "${dir}/${target}"
    make -C "$dir" CC="$CC" CXX="$CXX" "$target"

    if [ -f "${dir}/${target}" ]; then
        cp "${dir}/${target}" "${EXEC_DIR}/${output_name}"
    else
        echo "Error: ${target} was not built in ${dir}!"
    fi
}

cd "${CMLLES_SOURCE_DIR}"

compile_and_copy "FANN/benchmarks/MLP/Iris"     "benchmark_mlp"  "fann_bench_mlp"
compile_and_copy "FANN/benchmarks/MLP/IAES"     "benchmark_iaes" "fann_bench_iaes"
compile_and_copy "FANN/benchmarks/Autoencoder"  "benchmark_ae"   "fann_bench_ae"

compile_and_copy "Genann/benchmarks/MLP/Iris"    "benchmark_mlp"  "genann_bench_mlp"
compile_and_copy "Genann/benchmarks/MLP/IAES"    "benchmark_iaes" "genann_bench_iaes"
compile_and_copy "Genann/benchmarks/Autoencoder" "benchmark_ae"   "genann_bench_ae"

compile_and_copy "NCNN/benchmarks/MLP/Iris"     "benchmark_mlp"  "ncnn_bench_mlp"
compile_and_copy "NCNN/benchmarks/MLP/IAES"     "benchmark_iaes" "ncnn_bench_iaes"
compile_and_copy "NCNN/benchmarks/Autoencoder"  "benchmark_ae"   "ncnn_bench_ae"
compile_and_copy "NCNN/benchmarks/CNN2D" "benchmark_cnn2d" "ncnn_bench_cnn2d"
compile_and_copy "NCNN/benchmarks/LSTM" "benchmark_lstm" "ncnn_bench_lstm"

compile_and_copy "LiteRT/benchmarks/MLP/Iris"    "benchmark_mlp"  "litert_bench_mlp"
compile_and_copy "LiteRT/benchmarks/MLP/IAES"    "benchmark_iaes" "litert_bench_iaes"
compile_and_copy "LiteRT/benchmarks/Autoencoder" "benchmark_ae"   "litert_bench_ae"
compile_and_copy "LiteRT/benchmarks/CNN2D" "benchmark_cnn2d" "litert_bench_cnn2d"
compile_and_copy "LiteRT/benchmarks/LSTM" "benchmark_lstm" "litert_bench_lstm"

if [ -f "${LIBS_PREFIX}/lib/libonnxruntime.so" ]; then
    compile_and_copy "ONNX/benchmarks/MLP/Iris"    "benchmark_mlp"  "onnx_bench_mlp"
    compile_and_copy "ONNX/benchmarks/MLP/IAES"    "benchmark_iaes" "onnx_bench_iaes"
    compile_and_copy "ONNX/benchmarks/Autoencoder" "benchmark_ae"   "onnx_bench_ae"
    compile_and_copy "ONNX/benchmarks/CNN2D" "benchmark_cnn2d" "onnx_bench_cnn2d"
    compile_and_copy "ONNX/benchmarks/LSTM" "benchmark_lstm" "onnx_bench_lstm"
else
    echo "Skipping ONNX benchmarks: ${LIBS_PREFIX}/lib/libonnxruntime.so not found."
fi


echo "============================================================="
echo " Cross-Compilation Complete! Executables are in ${EXEC_DIR}"
echo "============================================================="
