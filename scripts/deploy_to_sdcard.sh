#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

source env.sh

SD_DEVICE="${1:-/dev/sda}"
WORK_IMG="${2:-${IMAGES_DIR}/cmlles-raspios.img}"

if [ ! -f "$WORK_IMG" ]; then
    echo "Erro: imagem não encontrada em ${WORK_IMG}"
    echo "Execute ./build.sh primeiro para gerar a imagem."
    exit 1
fi

if [ ! -b "$SD_DEVICE" ]; then
    echo "Erro: dispositivo ${SD_DEVICE} não encontrado."
    echo "Use: $0 /dev/sdX"
    exit 1
fi

IMG_SIZE=$(stat -c%s "$WORK_IMG")
IMG_SIZE_MB=$((IMG_SIZE / 1024 / 1024))

echo "============================================================="
echo " Deploy CMLLES -> SD Card"
echo "============================================================="
echo " Imagem : $WORK_IMG (${IMG_SIZE_MB} MB)"
echo " Destino: $SD_DEVICE"
echo ""
echo " ATENÇÃO: todos os dados em $SD_DEVICE serão apagados!"
echo "============================================================="
read -rp " Confirmar gravação? (s/N): " confirm
if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
    echo "Cancelado."
    exit 0
fi

echo "Desmontando partições de ${SD_DEVICE}..."
for mp in $(lsblk -ln -o MOUNTPOINT "$SD_DEVICE" 2>/dev/null | grep -v '^$'); do
    sudo umount "$mp" && echo "  desmontado: $mp" || true
done

echo "Gravando imagem no SD card (pode demorar alguns minutos)..."
sudo dd if="$WORK_IMG" of="$SD_DEVICE" bs=4M status=progress conv=fsync

echo "Sincronizando..."
sync

echo ""
echo "============================================================="
echo " Pronto! SD card gravado com sucesso."
echo " Remova o cartão e insira na placa de destino."
echo "============================================================="
