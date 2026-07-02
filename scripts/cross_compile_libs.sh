#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
source env.sh

echo "============================================================="
echo " Cross-Compiling ML Libraries for AArch64"
echo "============================================================="

TOOLCHAIN_FILE="${SCRIPT_DIR}/aarch64-toolchain.cmake"

export CC="$CROSS_CC"
export CXX="$CROSS_CXX"
export AR="$CROSS_AR"
export LD="$CROSS_LD"

echo "--- Compiling FANN ---"
cd "${CMLLES_SOURCE_DIR}/FANN/fann"
rm -rf build_aarch64
mkdir -p build_aarch64 && cd build_aarch64
cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" -DCMAKE_INSTALL_PREFIX="${LIBS_PREFIX}" ..
make -j${JOBS}
make install

echo "--- Compiling NCNN ---"
cd "${CMLLES_SOURCE_DIR}/NCNN/ncnn"
rm -rf build_aarch64
mkdir -p build_aarch64 && cd build_aarch64
cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" -DNCNN_VULKAN=OFF -DCMAKE_INSTALL_PREFIX="${LIBS_PREFIX}" ..
make -j${JOBS}
make install

echo "--- Compiling LiteRT ---"
cd "${CMLLES_SOURCE_DIR}/LiteRT/tensorflow"
rm -rf tflite_build_aarch64
mkdir -p tflite_build_aarch64 && cd tflite_build_aarch64
cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
      ../tensorflow/lite/c
make -j${JOBS}
mkdir -p "${LIBS_PREFIX}/lib"
cp libtensorflowlite_c.* "${LIBS_PREFIX}/lib/" 2>/dev/null || true
mkdir -p "${LIBS_PREFIX}/include/tensorflow/lite"
cp -r "${CMLLES_SOURCE_DIR}/LiteRT/tensorflow/tensorflow/lite/c" \
      "${LIBS_PREFIX}/include/tensorflow/lite/c"
echo "--- Compiling ONNX Runtime ---"
cd "${CMLLES_SOURCE_DIR}/ONNX/onnxruntime-bak" 2>/dev/null || cd "${CMLLES_SOURCE_DIR}/ONNX"
if [ -f "build.sh" ]; then
    ./build.sh --config Release --build_shared_lib --arm64 \
        --cmake_extra_defines CMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
        --parallel ${JOBS}
else
    echo "WARNING: ONNX Runtime build.sh not found."
fi

echo "============================================================="
echo " Cross-Compilation of Libraries Complete!"
echo "============================================================="
