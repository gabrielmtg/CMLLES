#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

source env_riscv.sh

echo "============================================================="
echo " Building Buildroot Image for StarFive VisionFive 2 (RISC-V)"
echo " (With Cross-Compiled CMLLES Benchmarks on Boot)"
echo "============================================================="

mkdir -p "$BUILDROOT_OUTPUT_DIR"

if [ ! -f "${BUILDROOT_OUTPUT_DIR}/.config" ]; then
    echo "Configuring Buildroot (${BUILDROOT_DEFCONFIG})..."
    make -C "${BUILDROOT_SRC_DIR}" O="${BUILDROOT_OUTPUT_DIR}" "${BUILDROOT_DEFCONFIG}"
else
    echo "Buildroot already configured, skipping defconfig."
fi

if ! { command -v "${RV_CROSS_CC}" && command -v "${RV_CROSS_CXX}"; } >/dev/null 2>&1; then
    echo "Building Buildroot's internal riscv64 toolchain (this can take a while)..."
    make -C "${BUILDROOT_OUTPUT_DIR}" toolchain
else
    echo "riscv64 toolchain already available, skipping toolchain-only build."
fi

benchmarks_compiled() {
    local names=(
        fann_bench_mlp fann_bench_iaes fann_bench_ae
        genann_bench_mlp genann_bench_iaes genann_bench_ae
        ncnn_bench_mlp ncnn_bench_iaes ncnn_bench_ae ncnn_bench_cnn2d ncnn_bench_lstm
        litert_bench_mlp litert_bench_iaes litert_bench_ae litert_bench_cnn2d litert_bench_lstm
    )
    if [ -f "${LIBS_PREFIX}/lib/libonnxruntime.so" ]; then
        names+=(onnx_bench_mlp onnx_bench_iaes onnx_bench_ae onnx_bench_cnn2d onnx_bench_lstm)
    fi
    for name in "${names[@]}"; do
        [ -f "${EXEC_DIR}/${name}" ] || return 1
    done
    return 0
}

# cross_compile_libs_riscv.sh gates each library (and the ONNX Runtime
# attempt) on its own installed artifact internally, so it's always safe -
# and fast, when everything's already built - to call it unconditionally.
# This also ensures ONNX gets retried on the next run if it previously
# failed, instead of being silently skipped forever once the other three
# libraries already exist.
echo "Cross-compiling ML Libraries..."
./cross/cross_compile_libs_riscv.sh

if benchmarks_compiled; then
    echo "Benchmarks already compiled, skipping."
else
    echo "Cross-compiling CMLLES Benchmarks..."
    ./cross/cross_compile_tests_riscv.sh
fi

echo "Building full Buildroot image (kernel, rootfs, sdcard.img)..."
make -C "${BUILDROOT_OUTPUT_DIR}"

echo "============================================================="
echo " Build Complete!"
echo " The SD Card image is located at:"
echo "   ${RV_IMAGES_DIR}/sdcard.img"
echo " Deploy it with:"
echo "   scripts/deploy_to_sdcard.sh /dev/sdX \"${RV_IMAGES_DIR}/sdcard.img\""
echo "============================================================="
