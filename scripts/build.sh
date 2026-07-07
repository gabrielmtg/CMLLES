#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

source env.sh

echo "============================================================="
echo " Building Raspberry Pi OS Lite Image for Raspberry Pi 4 B"
echo " (With Cross-Compiled CMLLES Benchmarks on Boot)"
echo "============================================================="

mkdir -p "$IMAGES_DIR"
mkdir -p "$LIBS_PREFIX"

if [ ! -f "${IMAGES_DIR}/${RPIOS_FILENAME}" ]; then
    echo "Downloading Raspberry Pi OS Lite..."
    if [ ! -f "${IMAGES_DIR}/${RPIOS_FILENAME}.xz" ]; then
        wget -O "${IMAGES_DIR}/${RPIOS_FILENAME}.xz" "$RPIOS_URL"
    fi
    echo "Decompressing image..."
    xz -dk "${IMAGES_DIR}/${RPIOS_FILENAME}.xz"
fi

libs_compiled() {
    for f in libfann.so libncnn.a libonnxruntime.so libtensorflowlite_c.so; do
        [ -f "${LIBS_PREFIX}/lib/${f}" ] || return 1
    done
    return 0
}

benchmarks_compiled() {
    local names=(
        fann_bench_mlp fann_bench_iaes fann_bench_ae
        genann_bench_mlp genann_bench_iaes genann_bench_ae
        ncnn_bench_mlp ncnn_bench_iaes ncnn_bench_ae ncnn_bench_cnn2d ncnn_bench_lstm
        litert_bench_mlp litert_bench_iaes litert_bench_ae litert_bench_cnn2d litert_bench_lstm
        onnx_bench_mlp onnx_bench_iaes onnx_bench_ae onnx_bench_cnn2d onnx_bench_lstm
    )
    for s in small medium large; do
        for a in relu sigmoid; do
            names+=(cmsisnn_bench_mlp_${s}_${a} cmsisnn_bench_iaes_${s}_${a})
        done
        names+=(cmsisnn_bench_ae_${s})
    done
    for cfg in simple intermediate mobilenet; do names+=(cmsisnn_bench_cnn2d_${cfg}); done
    for h in 32 64 128; do names+=(cmsisnn_bench_lstm_h${h}); done
    for name in "${names[@]}"; do
        [ -f "${EXEC_DIR}/${name}" ] || return 1
    done
    return 0
}

if libs_compiled; then
    echo "ML Libraries already compiled, skipping."
else
    echo "Cross-compiling ML Libraries..."
    ./cross/cross_compile_libs.sh
fi

if benchmarks_compiled; then
    echo "Benchmarks already compiled, skipping."
else
    echo "Cross-compiling CMLLES Benchmarks..."
    ./cross/cross_compile_tests.sh
fi

WORK_IMG="${IMAGES_DIR}/cmlles-raspios.img"
echo "Creating working copy of the image..."
cp "${IMAGES_DIR}/${RPIOS_FILENAME}" "$WORK_IMG"

echo "Expanding rootfs partition by 1.5 GB..."
truncate -s +1536M "$WORK_IMG"
sudo parted -s "$WORK_IMG" resizepart 2 100%
EXPAND_LOOP=$(sudo losetup --find --show -P "$WORK_IMG")
sudo e2fsck -fy "${EXPAND_LOOP}p2" || true
sudo resize2fs "${EXPAND_LOOP}p2"
sudo losetup -d "$EXPAND_LOOP"

MOUNT_DIR="${OUTPUT_DIR}/mnt_rootfs"
mkdir -p "$MOUNT_DIR"

echo "Mounting partitions..."
LOOP_DEV=$(sudo losetup --find --show -P "$WORK_IMG")
sudo mount "${LOOP_DEV}p2" "$MOUNT_DIR"

BOOT_DIR="${OUTPUT_DIR}/mnt_boot"
mkdir -p "$BOOT_DIR"
sudo mount "${LOOP_DEV}p1" "$BOOT_DIR"

echo "Enabling UART serial console..."
if ! sudo grep -q "^enable_uart=1" "$BOOT_DIR/config.txt"; then
    echo "enable_uart=1" | sudo tee -a "$BOOT_DIR/config.txt" > /dev/null
fi
if ! sudo grep -q "^dtoverlay=disable-bt" "$BOOT_DIR/config.txt"; then
    echo "dtoverlay=disable-bt" | sudo tee -a "$BOOT_DIR/config.txt" > /dev/null
fi

echo "Creating default user (pi/raspberry)..."
echo 'pi:$6$VzNVnqU8fq.2uCt2$ejKJaMAi5Hx8UkQxZy9LHSO/8j3fd.6lMz5qpM5dSp13XRbVkyB40xuNCsYSVDFZmkMNVFD9waDdcWPcHSycD1' \
    | sudo tee "$BOOT_DIR/userconf.txt" > /dev/null

sudo umount "$BOOT_DIR"
rmdir "$BOOT_DIR" 2>/dev/null || true

PERF_INSTALLED=false
PERF_PARANOID_SET=false
[ -f "${MOUNT_DIR}/usr/bin/perf" ] && PERF_INSTALLED=true
sudo grep -q "^kernel.perf_event_paranoid = -1" "${MOUNT_DIR}/etc/sysctl.conf" 2>/dev/null && PERF_PARANOID_SET=true

if $PERF_INSTALLED && $PERF_PARANOID_SET; then
    echo "linux-perf already installed and PMU access already configured, skipping."
else
    echo "Installing linux-perf and configuring PMU access..."
    if ! dpkg -l qemu-user-static &>/dev/null; then
        echo "  Installing qemu-user-static (only used here to run apt inside the arm64 rootfs)..."
        sudo apt-get install -y qemu-user-static >/dev/null
    fi
    sudo cp /usr/bin/qemu-aarch64-static "${MOUNT_DIR}/usr/bin/"
    sudo mount --bind /proc    "${MOUNT_DIR}/proc"
    sudo mount --bind /sys     "${MOUNT_DIR}/sys"
    sudo mount --bind /dev     "${MOUNT_DIR}/dev"
    sudo mount --bind /dev/pts "${MOUNT_DIR}/dev/pts"
    sudo cp /etc/resolv.conf "${MOUNT_DIR}/etc/resolv.conf"
    if ! $PERF_INSTALLED; then
        sudo chroot "${MOUNT_DIR}" apt-get update -qq
        sudo chroot "${MOUNT_DIR}" apt-get install -y linux-perf || \
            echo "Warning: linux-perf not installed — PMU data will not be collected"
    fi
    sudo umount "${MOUNT_DIR}/dev/pts"
    sudo umount "${MOUNT_DIR}/dev"
    sudo umount "${MOUNT_DIR}/sys"
    sudo umount "${MOUNT_DIR}/proc"
    sudo rm -f "${MOUNT_DIR}/usr/bin/qemu-aarch64-static"
    if ! $PERF_PARANOID_SET; then
        echo "kernel.perf_event_paranoid = -1" \
            | sudo tee -a "${MOUNT_DIR}/etc/sysctl.conf" > /dev/null
    fi
fi

echo "Installing shared libraries..."
sudo mkdir -p "${MOUNT_DIR}/usr/lib/aarch64-linux-gnu"
for lib in \
    libfann.so libfann.so.2 libfann.so.2.2.0 \
    libfloatfann.so libfloatfann.so.2 libfloatfann.so.2.2.0 \
    libdoublefann.so libdoublefann.so.2 libdoublefann.so.2.2.0 \
    libfixedfann.so libfixedfann.so.2 libfixedfann.so.2.2.0 \
    libonnxruntime.so libonnxruntime.so.1.17.1 \
    libtensorflowlite_c.so; do
    sudo cp "${LIBS_PREFIX}/lib/${lib}" "${MOUNT_DIR}/usr/lib/aarch64-linux-gnu/${lib}"
done
echo "Injecting executables..."
sudo mkdir -p "${MOUNT_DIR}/usr/bin/cmlles_tests"

deploy_bench() {
    local src_dir=$1
    local src_name=$2
    local dest_name=$3
    local dest="${MOUNT_DIR}/usr/bin/cmlles_tests/${src_dir}"
    sudo mkdir -p "$dest"
    sudo cp "${EXEC_DIR}/${src_name}" "${dest}/${dest_name}"
    if [ -d "${CMLLES_SOURCE_DIR}/${src_dir}/model" ]; then
        sudo mkdir -p "${dest}/model"
        find "${CMLLES_SOURCE_DIR}/${src_dir}/model" -maxdepth 1 -type f \
            ! -name "*_train.data" | \
            while read -r f; do sudo cp "$f" "${dest}/model/"; done
    fi
}

deploy_bench "FANN/benchmarks/MLP/Iris"     "fann_bench_mlp"   "benchmark_mlp"
deploy_bench "FANN/benchmarks/MLP/IAES"    "fann_bench_iaes"  "benchmark_iaes"
deploy_bench "FANN/benchmarks/Autoencoder" "fann_bench_ae"    "benchmark_ae"

deploy_bench "Genann/benchmarks/MLP/Iris"    "genann_bench_mlp"  "benchmark_mlp"
deploy_bench "Genann/benchmarks/MLP/IAES"    "genann_bench_iaes" "benchmark_iaes"
deploy_bench "Genann/benchmarks/Autoencoder" "genann_bench_ae"   "benchmark_ae"

deploy_bench "NCNN/benchmarks/MLP/Iris"    "ncnn_bench_mlp"  "benchmark_mlp"
deploy_bench "NCNN/benchmarks/MLP/IAES"    "ncnn_bench_iaes" "benchmark_iaes"
deploy_bench "NCNN/benchmarks/Autoencoder" "ncnn_bench_ae"   "benchmark_ae"
deploy_bench "NCNN/benchmarks/CNN2D" "ncnn_bench_cnn2d" "benchmark_cnn2d"
deploy_bench "NCNN/benchmarks/LSTM" "ncnn_bench_lstm" "benchmark_lstm"

deploy_bench "LiteRT/benchmarks/MLP/Iris"    "litert_bench_mlp"  "benchmark_mlp"
deploy_bench "LiteRT/benchmarks/MLP/IAES"   "litert_bench_iaes" "benchmark_iaes"
deploy_bench "LiteRT/benchmarks/Autoencoder" "litert_bench_ae"   "benchmark_ae"
deploy_bench "LiteRT/benchmarks/CNN2D" "litert_bench_cnn2d" "benchmark_cnn2d"
deploy_bench "LiteRT/benchmarks/LSTM" "litert_bench_lstm" "benchmark_lstm"

deploy_bench "ONNX/benchmarks/MLP/Iris"    "onnx_bench_mlp"  "benchmark_mlp"
deploy_bench "ONNX/benchmarks/MLP/IAES"   "onnx_bench_iaes" "benchmark_iaes"
deploy_bench "ONNX/benchmarks/Autoencoder" "onnx_bench_ae"   "benchmark_ae"
deploy_bench "ONNX/benchmarks/CNN2D" "onnx_bench_cnn2d" "benchmark_cnn2d"
deploy_bench "ONNX/benchmarks/LSTM" "onnx_bench_lstm" "benchmark_lstm"

for s in small medium large; do
    for a in relu sigmoid; do
        deploy_bench "CMSIS-NN/benchmarks/MLP/Iris"  "cmsisnn_bench_mlp_${s}_${a}"  "benchmark_mlp_${s}_${a}"
        deploy_bench "CMSIS-NN/benchmarks/MLP/IAES"  "cmsisnn_bench_iaes_${s}_${a}" "benchmark_iaes_${s}_${a}"
    done
done

for s in small medium large; do
    deploy_bench "CMSIS-NN/benchmarks/Autoencoder" "cmsisnn_bench_ae_${s}" "benchmark_ae_${s}"
done

for cfg in simple intermediate mobilenet; do
    deploy_bench "CMSIS-NN/benchmarks/CNN2D" "cmsisnn_bench_cnn2d_${cfg}" "benchmark_cnn2d_${cfg}"
done

for h in 32 64 128; do
    deploy_bench "CMSIS-NN/benchmarks/LSTM" "cmsisnn_bench_lstm_h${h}" "benchmark_lstm_h${h}"
done

echo "Installing benchmark runner..."
sudo cp image/run_benchmarks.sh "${MOUNT_DIR}/usr/bin/cmlles_tests/"
sudo chmod +x "${MOUNT_DIR}/usr/bin/cmlles_tests/run_benchmarks.sh"

echo "Installing systemd service..."
sudo cp image/cmlles-bench.service "${MOUNT_DIR}/etc/systemd/system/"
sudo ln -sf /etc/systemd/system/cmlles-bench.service "${MOUNT_DIR}/etc/systemd/system/multi-user.target.wants/cmlles-bench.service"

echo "Unmounting..."
sudo umount "$MOUNT_DIR"
sudo losetup -d "$LOOP_DEV"
rmdir "$MOUNT_DIR" 2>/dev/null || true

echo "============================================================="
echo " Build Complete!"
echo " The SD Card image is located at:"
echo "   ${WORK_IMG}"
echo " You can flash it using: sudo dd if=${WORK_IMG} of=/dev/sdX bs=4M status=progress"
echo "============================================================="
