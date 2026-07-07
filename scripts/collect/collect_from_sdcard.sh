#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/../env.sh"

RESULTS_OUT="${OUTPUT_DIR}/cmlles_results"
LATENCIES_OUT="${OUTPUT_DIR}/latencies_rpi"

echo "Dispositivos disponíveis:"
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT | grep -v loop
echo ""
read -rp "Digite o dispositivo do SD card (ex: sda, sdb): " DEV
DEV="/dev/${DEV}"

if [ ! -b "${DEV}2" ]; then
    echo "Erro: partição ${DEV}2 não encontrada."
    exit 1
fi

MOUNT_DIR=$(mktemp -d)
echo "Montando ${DEV}2..."
sudo mount -o ro "${DEV}2" "$MOUNT_DIR"

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
rm -rf "${RESULTS_DIR}/cmlles_results" "${RESULTS_DIR}/latencies_rpi"
cp -r "$RESULTS_OUT"   "${RESULTS_DIR}/cmlles_results"
cp -r "$LATENCIES_OUT" "${RESULTS_DIR}/latencies_rpi"
echo "  Cópia salva em ${RESULTS_DIR}"
