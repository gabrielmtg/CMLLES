#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/../env_riscv.sh"

RESULTS_OUT="${OUTPUT_DIR}/cmlles_results"
LATENCIES_OUT="${OUTPUT_DIR}/latencies_riscv"

echo "Dispositivos disponíveis:"
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT | grep -v loop
echo ""
read -rp "Digite o dispositivo do SD card (ex: sda, sdb): " DEV
DEV="/dev/${DEV}"

# VisionFive2's genimage.cfg defines 3 partitions (two small dummy ones, then
# rootfs) - u-boot is hard-coded to look at the 3rd partition, confirmed by
# extlinux.conf's `root=/dev/mmcblk1p3`. This is partition 2 on the RPi image,
# so it's genuinely 3 here, not an off-by-one.
if [ ! -b "${DEV}3" ]; then
    echo "Erro: partição ${DEV}3 não encontrada."
    exit 1
fi

MOUNT_DIR=$(mktemp -d)
echo "Montando ${DEV}3..."
sudo mount -o ro "${DEV}3" "$MOUNT_DIR"

echo "Copiando resultados dos benchmarks..."
if [ -d "${MOUNT_DIR}/var/log/cmlles_results" ]; then
    rm -rf "$RESULTS_OUT"
    sudo cp -r "${MOUNT_DIR}/var/log/cmlles_results" "$RESULTS_OUT"
    sudo chown -R "$(id -u):$(id -g)" "$RESULTS_OUT"
    echo "  $(ls "$RESULTS_OUT" | wc -l) arquivos copiados para ${RESULTS_OUT}"
else
    echo "  Aviso: /var/log/cmlles_results não encontrado no SD card."
fi

echo "Copiando latências individuais..."
mkdir -p "$LATENCIES_OUT"
find "${MOUNT_DIR}/usr/bin/cmlles_tests" -type d -name "latencies" 2>/dev/null | \
while read -r lat_dir; do
    rel=$(echo "$lat_dir" | sed "s|${MOUNT_DIR}/usr/bin/cmlles_tests/||")
    dest="${LATENCIES_OUT}/${rel}"
    mkdir -p "$dest"
    sudo cp "${lat_dir}"/*.csv "$dest/" 2>/dev/null || true
done
sudo chown -R "$(id -u):$(id -g)" "$LATENCIES_OUT" 2>/dev/null || true
echo "  $(find "$LATENCIES_OUT" -name "*.csv" | wc -l) CSVs copiados para ${LATENCIES_OUT}"

sudo umount "$MOUNT_DIR"
rmdir "$MOUNT_DIR"

echo ""
echo "Verificando dados de perf..."
perf_count=$(grep -rl "instructions" "$RESULTS_OUT" 2>/dev/null | wc -l)
if [ "$perf_count" -gt 0 ]; then
    echo "  Perf coletado em ${perf_count}/$(ls "$RESULTS_OUT" | wc -l) benchmarks."
else
    echo "  Aviso: nenhum dado de perf encontrado. Perf estava instalado na imagem?"
fi

echo ""
echo "Copiando para ${RESULTS_DIR}..."
mkdir -p "$RESULTS_DIR"
rm -rf "${RESULTS_DIR}/cmlles_results_riscv" "${RESULTS_DIR}/latencies_riscv"
cp -r "$RESULTS_OUT"   "${RESULTS_DIR}/cmlles_results_riscv"
cp -r "$LATENCIES_OUT" "${RESULTS_DIR}/latencies_riscv"
echo "  Cópia salva em ${RESULTS_DIR}"
