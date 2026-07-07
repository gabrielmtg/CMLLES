#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/../env.sh"

echo "============================================================="
echo " Cross-Compiling ML Libraries for AArch64"
echo "============================================================="

TOOLCHAIN_FILE="${SCRIPT_DIR}/aarch64-toolchain.cmake"
BOOKWORM_SYSROOT="${OUTPUT_DIR}/bookworm-sysroot"
BOOKWORM_LDFLAGS="-L${BOOKWORM_SYSROOT}/lib/aarch64-linux-gnu"
COMPAT_OBJ="${OUTPUT_DIR}/glibc_compat.o"

create_bookworm_sysroot() {
    local tmpmt="${OUTPUT_DIR}/mnt_sysroot_tmp"
    local ldev

    sudo mkdir -p "$tmpmt"
    ldev=$(sudo losetup --find --show -P "${IMAGES_DIR}/${RPIOS_FILENAME}")
    sudo mount "${ldev}p2" "$tmpmt"

    mkdir -p "${BOOKWORM_SYSROOT}/lib/aarch64-linux-gnu"

    for f in libm.so.6; do
        [ -f "${tmpmt}/lib/aarch64-linux-gnu/${f}" ] && \
            sudo cp "${tmpmt}/lib/aarch64-linux-gnu/${f}" "${BOOKWORM_SYSROOT}/lib/aarch64-linux-gnu/"
    done
    ln -sf libm.so.6 "${BOOKWORM_SYSROOT}/lib/aarch64-linux-gnu/libm.so"

    sudo umount "$tmpmt"
    sudo losetup -d "$ldev"
    sudo rmdir "$tmpmt"

    sudo chown -R "$(id -u):$(id -g)" "${BOOKWORM_SYSROOT}"
}

if [ ! -d "${BOOKWORM_SYSROOT}/lib/aarch64-linux-gnu" ]; then
    echo "Creating Bookworm sysroot from RPi OS image..."
    create_bookworm_sysroot
fi

echo "--- Compiling glibc compatibility object ---"
aarch64-linux-gnu-gcc -fPIC -c \
    -U_GNU_SOURCE -U_ISOC23_SOURCE -D_POSIX_C_SOURCE=200809L \
    "${SCRIPT_DIR}/glibc_compat.c" -o "${COMPAT_OBJ}"

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
cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
      -DNCNN_VULKAN=OFF \
      -DCMAKE_C_FLAGS="-fno-tree-vectorize" \
      -DCMAKE_CXX_FLAGS="-fno-tree-vectorize" \
      -DCMAKE_INSTALL_PREFIX="${LIBS_PREFIX}" ..
make -j${JOBS}
make install

echo "--- Compiling LiteRT ---"
cd "${CMLLES_SOURCE_DIR}/LiteRT/tensorflow"
rm -rf tflite_build_aarch64
mkdir -p tflite_build_aarch64 && cd tflite_build_aarch64
cmake -DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/aarch64-gcc12-toolchain.cmake" \
      -DCMAKE_SHARED_LINKER_FLAGS="${BOOKWORM_LDFLAGS} ${COMPAT_OBJ}" \
      ../tensorflow/lite/c
make -j${JOBS}
mkdir -p "${LIBS_PREFIX}/lib"
cp libtensorflowlite_c.* "${LIBS_PREFIX}/lib/" 2>/dev/null || true
mkdir -p "${LIBS_PREFIX}/include/tensorflow/lite"
cp -r "${CMLLES_SOURCE_DIR}/LiteRT/tensorflow/tensorflow/lite/c" \
      "${LIBS_PREFIX}/include/tensorflow/lite/c"
echo "--- Installing ONNX Runtime (prebuilt) ---"
ONNX_PREBUILT="${CMLLES_SOURCE_DIR}/ONNX/onnxruntime-bak"
if [ ! -f "${ONNX_PREBUILT}/lib/libonnxruntime.so.1.17.1" ]; then
    echo "ERROR: prebuilt ONNX Runtime not found at ${ONNX_PREBUILT}/lib/"
    exit 1
fi
cp "${ONNX_PREBUILT}/lib/libonnxruntime.so.1.17.1" "${LIBS_PREFIX}/lib/"
ln -sf libonnxruntime.so.1.17.1 "${LIBS_PREFIX}/lib/libonnxruntime.so"
mkdir -p "${LIBS_PREFIX}/include/onnxruntime"
cp "${ONNX_PREBUILT}/include/"* "${LIBS_PREFIX}/include/onnxruntime/" 2>/dev/null || true

echo "============================================================="
echo " Cross-Compilation of Libraries Complete!"
echo "============================================================="
