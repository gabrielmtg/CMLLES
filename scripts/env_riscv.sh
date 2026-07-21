#!/bin/bash

export JOBS=$(nproc)

export CMLLES_SOURCE_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"

export BUILDROOT_SRC_DIR="${CMLLES_SOURCE_DIR}/scripts/buildroot-2024.02.2"
export BUILDROOT_DEFCONFIG="visionfive2_cmlles_defconfig"

export OUTPUT_DIR="${CMLLES_SOURCE_DIR}/output_rv_image"
export BUILDROOT_OUTPUT_DIR="${OUTPUT_DIR}/buildroot"
export BR2_DL_DIR="${OUTPUT_DIR}/buildroot-dl"

export RV_HOST_DIR="${BUILDROOT_OUTPUT_DIR}/host"
export RV_IMAGES_DIR="${BUILDROOT_OUTPUT_DIR}/images"

export RV_CROSS_TUPLE="riscv64-buildroot-linux-gnu"
export RV_CROSS_CC="${RV_CROSS_TUPLE}-gcc"
export RV_CROSS_CXX="${RV_CROSS_TUPLE}-g++"
export RV_CROSS_AR="${RV_CROSS_TUPLE}-ar"
export RV_CROSS_LD="${RV_CROSS_TUPLE}-ld"

# NOTE: intentionally NOT ${BUILDROOT_OUTPUT_DIR}/staging - that's a symlink
# Buildroot only creates during the *full* package build sequence, well
# after `make toolchain` finishes. The internal toolchain's own sysroot
# (which staging/ eventually points at anyway) is fully populated with
# glibc headers/libs immediately once the toolchain itself is built, so
# scripts that only need `make toolchain` (e.g. the ONNX Runtime build)
# can rely on this path being ready without waiting on the full build.
export RV_STAGING_DIR="${RV_HOST_DIR}/${RV_CROSS_TUPLE}/sysroot"

export PATH="${RV_HOST_DIR}/bin:${PATH}"

export LIBS_PREFIX="${OUTPUT_DIR}/riscv64-libs"
export EXEC_DIR="${OUTPUT_DIR}/executables"

export RESULTS_DIR="${CMLLES_SOURCE_DIR}/results"
