#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/../env.sh"

LATENCIES_OUT="${OUTPUT_DIR}/latencies_rpi"

echo "Detectando SD card..."
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
echo "Montando ${DEV}2 em ${MOUNT_DIR}..."
sudo mount -o ro "${DEV}2" "$MOUNT_DIR"

SRC="${MOUNT_DIR}/usr/bin/cmlles_tests"
if [ ! -d "$SRC" ]; then
    sudo umount "$MOUNT_DIR"
    rmdir "$MOUNT_DIR"
    echo "Erro: ${SRC} não encontrado no SD card."
    exit 1
fi

echo "Copiando latencies/..."
mkdir -p "$LATENCIES_OUT"

count=0
find "$SRC" -type d -name "latencies" | while read -r lat_dir; do
    rel=$(echo "$lat_dir" | sed "s|${SRC}/||")
    dest="${LATENCIES_OUT}/${rel}"
    mkdir -p "$dest"
    sudo cp "${lat_dir}"/*.csv "$dest/" 2>/dev/null || true
    echo "  copiado: $rel"
    count=$((count + 1))
done

echo ""
sudo umount "$MOUNT_DIR"
rmdir "$MOUNT_DIR"

echo "Latências salvas em: ${LATENCIES_OUT}"
echo "Arquivos copiados:"
find "$LATENCIES_OUT" -name "*.csv" | wc -l
