#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/../env_riscv.sh"

echo "============================================================="
echo " Cross-Compiling ML Libraries for RISC-V (riscv64)"
echo "============================================================="

if ! command -v "${RV_CROSS_CC}" >/dev/null 2>&1; then
    echo "ERROR: ${RV_CROSS_CC} not found on PATH."
    echo "Build the Buildroot toolchain first: make -C \"${BUILDROOT_OUTPUT_DIR}\" toolchain"
    exit 1
fi

TOOLCHAIN_FILE="${SCRIPT_DIR}/riscv64-toolchain.cmake"

mkdir -p "${LIBS_PREFIX}"


if [ -f "${LIBS_PREFIX}/lib/libfann.so" ]; then
    echo "--- FANN already built, skipping ---"
else
    echo "--- Compiling FANN ---"
    cd "${CMLLES_SOURCE_DIR}/FANN/fann"
    rm -rf build_riscv64
    mkdir -p build_riscv64 && cd build_riscv64
    cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" -DCMAKE_INSTALL_PREFIX="${LIBS_PREFIX}" ..
    make -j${JOBS}
    make install
fi

if [ -f "${LIBS_PREFIX}/lib/libncnn.a" ]; then
    echo "--- NCNN already built, skipping ---"
else
    echo "--- Compiling NCNN ---"
    cd "${CMLLES_SOURCE_DIR}/NCNN/ncnn"
    rm -rf build_riscv64
    mkdir -p build_riscv64 && cd build_riscv64

    cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
          -DNCNN_VULKAN=OFF \
          -DNCNN_RVV=OFF \
          -DNCNN_XTHEADVECTOR=OFF \
          -DNCNN_ZFH=OFF \
          -DNCNN_ZVFH=OFF \
          -DCMAKE_INSTALL_PREFIX="${LIBS_PREFIX}" ..
    make -j${JOBS}
    make install
fi

if [ -f "${LIBS_PREFIX}/lib/libtensorflowlite_c.so" ]; then
    echo "--- LiteRT already built, skipping ---"
else
    echo "--- Compiling LiteRT ---"

    HWPROBE_COMPAT_INCLUDE="${SCRIPT_DIR}/riscv_hwprobe_compat_include"
    HWPROBE_COMPAT_OBJ="${LIBS_PREFIX}/riscv_hwprobe_compat.o"
    "${RV_CROSS_CC}" -fPIC -c "${SCRIPT_DIR}/riscv_hwprobe_compat.c" -o "${HWPROBE_COMPAT_OBJ}"

    GLIBC_COMPAT_OBJ="${LIBS_PREFIX}/glibc_compat.o"
    "${RV_CROSS_CC}" -fPIC -c -U_GNU_SOURCE -U_ISOC23_SOURCE -D_POSIX_C_SOURCE=200809L \
        "${SCRIPT_DIR}/glibc_compat.c" -o "${GLIBC_COMPAT_OBJ}"

    cd "${CMLLES_SOURCE_DIR}/LiteRT/tensorflow"
    rm -rf tflite_build_riscv64
    mkdir -p tflite_build_riscv64 && cd tflite_build_riscv64

    cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
          -DTFLITE_ENABLE_XNNPACK=OFF \
          -DCMAKE_C_FLAGS="-I${HWPROBE_COMPAT_INCLUDE}" \
          -DCMAKE_CXX_FLAGS="-I${HWPROBE_COMPAT_INCLUDE}" \
          -DCMAKE_SHARED_LINKER_FLAGS="${HWPROBE_COMPAT_OBJ} ${GLIBC_COMPAT_OBJ} -static-libstdc++ -static-libgcc" \
          ../tensorflow/lite/c
    make -j${JOBS}
    mkdir -p "${LIBS_PREFIX}/lib"
    cp libtensorflowlite_c.* "${LIBS_PREFIX}/lib/" 2>/dev/null || true
    mkdir -p "${LIBS_PREFIX}/include/tensorflow/lite"
    cp -r "${CMLLES_SOURCE_DIR}/LiteRT/tensorflow/tensorflow/lite/c" \
          "${LIBS_PREFIX}/include/tensorflow/lite/c"
fi

if [ -f "${LIBS_PREFIX}/lib/libonnxruntime.so" ]; then
    echo "--- ONNX Runtime already built, skipping ---"
else
    echo "--- Building ONNX Runtime from source (best-effort) ---"
    "${SCRIPT_DIR}/build_onnxruntime_riscv.sh" || true
    if [ -f "${LIBS_PREFIX}/lib/libonnxruntime.so" ]; then
        echo "ONNX Runtime available for riscv64."
    else
        echo "ONNX Runtime NOT available for riscv64 - ONNX benchmarks will be skipped."
    fi
fi

echo "============================================================="
echo " Cross-Compilation of Libraries Complete!"
echo "============================================================="
