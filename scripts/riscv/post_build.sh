#!/bin/bash
#
# Buildroot BR2_ROOTFS_POST_BUILD_SCRIPT hook for the VisionFive2/CMLLES image.
# Invoked by Buildroot as: post_build.sh $TARGET_DIR $(BR2_ROOTFS_POST_SCRIPT_ARGS)
# NOTE: BR2_ROOTFS_POST_SCRIPT_ARGS is shared with the genimage post-image hook
# ("-c board/visionfive2/genimage.cfg") - only $1 (TARGET_DIR) is ours to use.
#
set -e

TARGET_DIR="$1"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/../env_riscv.sh"

echo "============================================================="
echo " CMLLES post-build: injecting libs/benchmarks into rootfs"
echo "============================================================="

echo "Installing shared libraries..."
mkdir -p "${TARGET_DIR}/usr/lib"
for f in "${LIBS_PREFIX}"/lib/*.so*; do
    [ -e "$f" ] && cp -P "$f" "${TARGET_DIR}/usr/lib/"
done

echo "Injecting executables..."
mkdir -p "${TARGET_DIR}/usr/bin/cmlles_tests"

deploy_bench() {
    local src_dir=$1
    local src_name=$2
    local dest_name=$3
    local dest="${TARGET_DIR}/usr/bin/cmlles_tests/${src_dir}"

    if [ ! -f "${EXEC_DIR}/${src_name}" ]; then
        echo "  Warning: ${EXEC_DIR}/${src_name} not found, skipping ${src_dir}"
        return
    fi

    mkdir -p "$dest"
    cp "${EXEC_DIR}/${src_name}" "${dest}/${dest_name}"
    if [ -d "${CMLLES_SOURCE_DIR}/${src_dir}/model" ]; then
        mkdir -p "${dest}/model"
        find "${CMLLES_SOURCE_DIR}/${src_dir}/model" -maxdepth 1 -type f \
            ! -name "*_train.data" | \
            while read -r f; do cp "$f" "${dest}/model/"; done
    fi
}

deploy_bench "FANN/benchmarks/MLP/Iris"     "fann_bench_mlp"   "benchmark_mlp"
deploy_bench "FANN/benchmarks/MLP/IAES"     "fann_bench_iaes"  "benchmark_iaes"
deploy_bench "FANN/benchmarks/Autoencoder"  "fann_bench_ae"    "benchmark_ae"

deploy_bench "Genann/benchmarks/MLP/Iris"    "genann_bench_mlp"  "benchmark_mlp"
deploy_bench "Genann/benchmarks/MLP/IAES"    "genann_bench_iaes" "benchmark_iaes"
deploy_bench "Genann/benchmarks/Autoencoder" "genann_bench_ae"   "benchmark_ae"

deploy_bench "NCNN/benchmarks/MLP/Iris"    "ncnn_bench_mlp"  "benchmark_mlp"
deploy_bench "NCNN/benchmarks/MLP/IAES"    "ncnn_bench_iaes" "benchmark_iaes"
deploy_bench "NCNN/benchmarks/Autoencoder" "ncnn_bench_ae"   "benchmark_ae"
deploy_bench "NCNN/benchmarks/CNN2D" "ncnn_bench_cnn2d" "benchmark_cnn2d"
deploy_bench "NCNN/benchmarks/LSTM" "ncnn_bench_lstm" "benchmark_lstm"

deploy_bench "LiteRT/benchmarks/MLP/Iris"    "litert_bench_mlp"  "benchmark_mlp"
deploy_bench "LiteRT/benchmarks/MLP/IAES"   "litert_bench_iaes" "benchmark_iaes"
deploy_bench "LiteRT/benchmarks/Autoencoder" "litert_bench_ae"   "benchmark_ae"
deploy_bench "LiteRT/benchmarks/CNN2D" "litert_bench_cnn2d" "benchmark_cnn2d"
deploy_bench "LiteRT/benchmarks/LSTM" "litert_bench_lstm" "benchmark_lstm"

deploy_bench "ONNX/benchmarks/MLP/Iris"    "onnx_bench_mlp"  "benchmark_mlp"
deploy_bench "ONNX/benchmarks/MLP/IAES"   "onnx_bench_iaes" "benchmark_iaes"
deploy_bench "ONNX/benchmarks/Autoencoder" "onnx_bench_ae"   "benchmark_ae"
deploy_bench "ONNX/benchmarks/CNN2D" "onnx_bench_cnn2d" "benchmark_cnn2d"
deploy_bench "ONNX/benchmarks/LSTM" "onnx_bench_lstm" "benchmark_lstm"

echo "Installing benchmark runner..."
cp "${CMLLES_SOURCE_DIR}/scripts/image/run_benchmarks_riscv.sh" \
    "${TARGET_DIR}/usr/bin/cmlles_tests/run_benchmarks.sh"
chmod +x "${TARGET_DIR}/usr/bin/cmlles_tests/run_benchmarks.sh"

echo "Installing BusyBox init.d autorun script..."
mkdir -p "${TARGET_DIR}/etc/init.d"
cp "${CMLLES_SOURCE_DIR}/scripts/image/S99cmlles" "${TARGET_DIR}/etc/init.d/S99cmlles"
chmod +x "${TARGET_DIR}/etc/init.d/S99cmlles"

echo "============================================================="
echo " CMLLES post-build complete"
echo "============================================================="
