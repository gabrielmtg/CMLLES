#!/bin/bash

export RPIOS_VERSION="2024-11-19"
export RPIOS_DATE="2024-11-19"
export RPIOS_FILENAME="${RPIOS_DATE}-raspios-bookworm-arm64-lite.img"
export RPIOS_URL="https://downloads.raspberrypi.com/raspios_lite_arm64/images/raspios_lite_arm64-${RPIOS_VERSION}/${RPIOS_FILENAME}.xz"

export JOBS=$(nproc)

export CMLLES_SOURCE_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"

export OUTPUT_DIR="${CMLLES_SOURCE_DIR}/output_rpi_image"
export OVERLAY_DIR="${OUTPUT_DIR}/rootfs_overlay"
export IMAGES_DIR="${OUTPUT_DIR}/images"
export LIBS_PREFIX="${OUTPUT_DIR}/aarch64-sysroot"
export EXEC_DIR="${OUTPUT_DIR}/executables"

export RESULTS_DIR="${CMLLES_SOURCE_DIR}/results"

export CROSS_CC="aarch64-linux-gnu-gcc"
export CROSS_CXX="aarch64-linux-gnu-g++"
export CROSS_AR="aarch64-linux-gnu-ar"
export CROSS_LD="aarch64-linux-gnu-ld"
