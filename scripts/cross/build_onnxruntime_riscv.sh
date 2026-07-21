#!/bin/bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/../env_riscv.sh"

ORT_REPO="https://github.com/microsoft/onnxruntime.git"
ORT_REF="${ORT_RISCV_REF:-main}"
ORT_SRC_DIR="${OUTPUT_DIR}/onnxruntime-src"
ORT_BUILD_DIR="${OUTPUT_DIR}/onnxruntime-build"
SHIM_DIR="${OUTPUT_DIR}/onnxruntime-riscv-toolchain-shim"

fail() {
    echo "Warning: ONNX Runtime riscv64 build failed (${1}) - skipping ONNX for this run." >&2
    exit 0
}

command -v "${RV_CROSS_CC}" >/dev/null 2>&1 || \
    fail "riscv64 toolchain (${RV_CROSS_CC}) not found on PATH - build the Buildroot toolchain first (make toolchain)"
[ -d "${RV_STAGING_DIR}" ] || \
    fail "Buildroot staging dir not found at ${RV_STAGING_DIR} - build the Buildroot toolchain first (make toolchain)"

echo "============================================================="
echo " Building ONNX Runtime from source for riscv64 (best-effort)"
echo "============================================================="

echo "--- Preparing ORT<->Buildroot toolchain shim ---"
mkdir -p "${SHIM_DIR}/bin"

ln -sf "$(command -v "${RV_CROSS_CC}").br_real"  "${SHIM_DIR}/bin/riscv64-unknown-linux-gnu-gcc"
ln -sf "$(command -v "${RV_CROSS_CXX}").br_real" "${SHIM_DIR}/bin/riscv64-unknown-linux-gnu-g++"
ln -sfn "${RV_STAGING_DIR}" "${SHIM_DIR}/sysroot"

if [ ! -d "${ORT_SRC_DIR}/.git" ]; then
    echo "--- Cloning ONNX Runtime (${ORT_REF}) ---"
    git clone --recursive --depth 1 --branch "${ORT_REF}" "${ORT_REPO}" "${ORT_SRC_DIR}" || \
        fail "git clone of ${ORT_REPO}@${ORT_REF} failed"
fi

echo "--- Checking for riscv64 build support (--rv64) in this ORT checkout ---"
python3 "${ORT_SRC_DIR}/tools/ci_build/build.py" --help 2>/dev/null | grep -q -- "--rv64" || \
    fail "this ORT checkout (${ORT_REF}) has no --rv64 flag - try a newer ref via ORT_RISCV_REF"


HWPROBE_COMPAT_INCLUDE="${SCRIPT_DIR}/riscv_hwprobe_compat_include"
HWPROBE_COMPAT_OBJ="${LIBS_PREFIX}/riscv_hwprobe_compat.o"
mkdir -p "${LIBS_PREFIX}"
"${RV_CROSS_CC}" -fPIC -c "${SCRIPT_DIR}/riscv_hwprobe_compat.c" -o "${HWPROBE_COMPAT_OBJ}"

GLIBC_COMPAT_OBJ="${LIBS_PREFIX}/glibc_compat.o"
"${RV_CROSS_CC}" -fPIC -c -U_GNU_SOURCE -U_ISOC23_SOURCE -D_POSIX_C_SOURCE=200809L \
    "${SCRIPT_DIR}/glibc_compat.c" -o "${GLIBC_COMPAT_OBJ}"

echo "--- Building ONNX Runtime (this can take a long time) ---"
"${ORT_SRC_DIR}/build.sh" \
    --build_dir "${ORT_BUILD_DIR}" \
    --config Release \
    --parallel \
    --skip_tests \
    --build_shared_lib \
    --allow_running_as_root \
    --compile_no_warning_as_error \
    --rv64 \
    --riscv_toolchain_root="${SHIM_DIR}" \
    --cmake_extra_defines "CMAKE_C_FLAGS=-I${HWPROBE_COMPAT_INCLUDE} -D__NR_riscv_hwprobe=258" "CMAKE_CXX_FLAGS=-I${HWPROBE_COMPAT_INCLUDE} -D__NR_riscv_hwprobe=258" "CMAKE_SHARED_LINKER_FLAGS=${HWPROBE_COMPAT_OBJ} ${GLIBC_COMPAT_OBJ} -static-libstdc++ -static-libgcc -latomic" "CMAKE_EXE_LINKER_FLAGS=${HWPROBE_COMPAT_OBJ} ${GLIBC_COMPAT_OBJ} -static-libstdc++ -static-libgcc -latomic" \
    || fail "ONNX Runtime build.sh invocation failed"

ORT_LIB=$(find "${ORT_BUILD_DIR}" -maxdepth 3 -name "libonnxruntime.so*" -print -quit)
[ -n "${ORT_LIB}" ] || fail "build finished but libonnxruntime.so was not found under ${ORT_BUILD_DIR}"

echo "--- Installing ONNX Runtime into ${LIBS_PREFIX} ---"

mkdir -p "${LIBS_PREFIX}/lib" "${LIBS_PREFIX}/include"
cp -P "${ORT_BUILD_DIR}"/Release/libonnxruntime.so* "${LIBS_PREFIX}/lib/" 2>/dev/null || \
    cp -P "${ORT_BUILD_DIR}"/*/libonnxruntime.so* "${LIBS_PREFIX}/lib/"
ln -sf "$(basename "$(ls "${LIBS_PREFIX}/lib/libonnxruntime.so".* | head -1)")" "${LIBS_PREFIX}/lib/libonnxruntime.so"
cp "${ORT_SRC_DIR}/include/onnxruntime/core/session/"*.h "${LIBS_PREFIX}/include/" 2>/dev/null || true

echo "============================================================="
echo " ONNX Runtime riscv64 build complete"
echo "============================================================="
