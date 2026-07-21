#!/bin/bash
#
# Overwrites all 21 benchmark executables (FANN/Genann/NCNN/LiteRT/ONNX) and
# the two shared libraries (libtensorflowlite_c.so, libonnxruntime.so*)
# already deployed under usr/bin/cmlles_tests and usr/lib on a target
# rootfs, with freshly rebuilt riscv64 artifacts from
# output_rv_image/{executables,riscv64-libs}. Everything else on the target
# is left alone.
#
# Must be run as root (writes into a root-owned rootfs mount).
#
# Usage: sudo ./redeploy_riscv_fix.sh [SDCARD_ROOT]
#   SDCARD_ROOT defaults to /media/gabriel/root

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/env_riscv.sh"

SDCARD_ROOT="${1:-/media/gabriel/root}"
TESTS_ROOT="${SDCARD_ROOT}/usr/bin/cmlles_tests"
LIB_ROOT="${SDCARD_ROOT}/usr/lib"

if [ "$(id -u)" -ne 0 ]; then
    echo "Erro: precisa rodar como root (sudo)." >&2
    exit 1
fi

if [ ! -d "${TESTS_ROOT}" ]; then
    echo "Erro: ${TESTS_ROOT} nao existe - o cartao certo esta montado em ${SDCARD_ROOT}?" >&2
    exit 1
fi

deploy_bin() {
    local framework=$1 dir=$2 exec_name=$3
    local src="${EXEC_DIR}/${exec_name}"
    local dst="${TESTS_ROOT}/${framework}/benchmarks/${dir}"

    if [ ! -f "${src}" ]; then
        echo "Aviso: ${src} nao existe, pulando." >&2
        return
    fi
    if [ ! -d "${dst}" ]; then
        echo "Aviso: ${dst} nao existe no cartao, pulando." >&2
        return
    fi

    local bin_name
    bin_name=$(find "${dst}" -maxdepth 1 -type f -executable -printf '%f\n' | head -1)
    if [ -z "${bin_name}" ]; then
        echo "Aviso: nenhum binario existente encontrado em ${dst}, pulando." >&2
        return
    fi

    cp "${src}" "${dst}/${bin_name}"
    chmod 755 "${dst}/${bin_name}"
    echo "  ${dst}/${bin_name}"
}

echo "=== Reimplantando executaveis corrigidos em ${TESTS_ROOT} ==="

deploy_bin "FANN"   "MLP/Iris"     "fann_bench_mlp"
deploy_bin "FANN"   "MLP/IAES"     "fann_bench_iaes"
deploy_bin "FANN"   "Autoencoder"  "fann_bench_ae"

deploy_bin "Genann" "MLP/Iris"     "genann_bench_mlp"
deploy_bin "Genann" "MLP/IAES"     "genann_bench_iaes"
deploy_bin "Genann" "Autoencoder"  "genann_bench_ae"

deploy_bin "NCNN"   "MLP/Iris"     "ncnn_bench_mlp"
deploy_bin "NCNN"   "MLP/IAES"     "ncnn_bench_iaes"
deploy_bin "NCNN"   "Autoencoder"  "ncnn_bench_ae"
deploy_bin "NCNN"   "CNN2D"        "ncnn_bench_cnn2d"
deploy_bin "NCNN"   "LSTM"         "ncnn_bench_lstm"

deploy_bin "LiteRT" "MLP/Iris"     "litert_bench_mlp"
deploy_bin "LiteRT" "MLP/IAES"     "litert_bench_iaes"
deploy_bin "LiteRT" "Autoencoder"  "litert_bench_ae"
deploy_bin "LiteRT" "CNN2D"        "litert_bench_cnn2d"
deploy_bin "LiteRT" "LSTM"         "litert_bench_lstm"

deploy_bin "ONNX"   "MLP/Iris"     "onnx_bench_mlp"
deploy_bin "ONNX"   "MLP/IAES"     "onnx_bench_iaes"
deploy_bin "ONNX"   "Autoencoder"  "onnx_bench_ae"
deploy_bin "ONNX"   "CNN2D"        "onnx_bench_cnn2d"
deploy_bin "ONNX"   "LSTM"         "onnx_bench_lstm"

echo "=== Reimplantando bibliotecas corrigidas em ${LIB_ROOT} ==="

if [ -f "${LIBS_PREFIX}/lib/libtensorflowlite_c.so" ]; then
    cp -P "${LIBS_PREFIX}"/lib/libtensorflowlite_c.so* "${LIB_ROOT}/"
    chmod 755 "${LIB_ROOT}"/libtensorflowlite_c.so*
    echo "  ${LIB_ROOT}/libtensorflowlite_c.so"
else
    echo "Aviso: ${LIBS_PREFIX}/lib/libtensorflowlite_c.so nao existe, pulando." >&2
fi

if [ -f "${LIBS_PREFIX}/lib/libonnxruntime.so" ]; then
    cp -P "${LIBS_PREFIX}"/lib/libonnxruntime.so* "${LIB_ROOT}/"
    chmod 755 "${LIB_ROOT}"/libonnxruntime.so*
    echo "  ${LIB_ROOT}/libonnxruntime.so*"
else
    echo "Aviso: ${LIBS_PREFIX}/lib/libonnxruntime.so nao existe, pulando." >&2
fi

sync
echo "=== Concluido ==="
