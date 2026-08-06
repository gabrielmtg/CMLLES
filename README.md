# CMLLES — Comparison of Machine Learning Libraries on Embedded Systems

Reproducible benchmarking infrastructure for evaluating neural-network inference across six ML libraries on heterogeneous embedded platforms. This repository accompanies the paper:

> **A Comparison of Machine Learning Libraries Across Heterogeneous Embedded Platforms**
> Gabriel Martins, Eduardo Bischoff Grasel, Rafael Canal, Giovani Gracioli
> *Software/Hardware Integration Lab (LISHA) — Federal University of Santa Catarina (UFSC), Brazil*

## Overview

The evaluation combines software metrics (inference latency, static memory footprint, and model accuracy) with hardware-level measurements from the processor Performance Monitoring Unit (PMU), correlating inference performance with microarchitectural behavior (CPU cycles, IPC, L1 data-cache miss rate, and branch misses).

**Libraries:** CMSIS-NN · NCNN · Genann · ONNX Runtime · LiteRT · FANN

**Workloads:** MLP · Autoencoder · CNN-2D · LSTM — each at three complexity levels (small, medium, large)

**Platforms:**
| Platform | SoC / Core | ISA | RAM | OS |
|---|---|---|---|---|
| Raspberry Pi 4 Model B | Cortex-A72 @ 1.5 GHz | ARMv8-A (NEON SIMD) | 8 GiB | Raspberry Pi OS Lite arm64 (Bookworm, kernel 6.6) |
| StarFive VisionFive 2 | JH7110 / SiFive U74 @ 1.5 GHz | RV64GC (no RVV) | 7.7 GiB | StarFive Debian riscv64 (kernel 6.12) |

**Library × Workload coverage:**
| Library | MLP | Autoencoder | CNN-2D | LSTM | Precision |
|---|:---:|:---:|:---:|:---:|---|
| CMSIS-NN | ✔ | ✔ | ✔ | ✔ | INT8 |
| NCNN | ✔ | ✔ | ✔ | ✔ | Float32 |
| Genann | ✔ | ✔ | — | — | Float32 |
| ONNX Runtime | ✔ | ✔ | ✔ | ✔ | Float32, INT8 |
| LiteRT | ✔ | ✔ | ✔ | ✔ | Float32, INT8 |
| FANN | ✔ | ✔ | — | — | Float32 |

> CMSIS-NN is ARM-only (not evaluated on RISC-V). ONNX Runtime on RISC-V required a best-effort source build with static GLIBC compatibility.

---

## Datasets

Place datasets under `datasets/` before training:

| Workload | Dataset | Path |
|---|---|---|
| MLP (baseline) | [Iris](https://archive.ics.uci.edu/ml/datasets/iris) | `datasets/MLP/iris/iris.data` |
| MLP (anomaly detection) | [IAES Dataset](https://github.com/rafaelcanalp/IAES-Dataset) | `datasets/MLP/IAES-dataset/` |
| Autoencoder | [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) | `datasets/Autoencoder/NSL_KDD_Dataset/` |
| CNN-2D | [Visual Wake Words (VWW)](https://arxiv.org/abs/1906.05721) | `datasets/CNN-2D/coco/` |
| LSTM | [NASA C-MAPSS](https://data.nasa.gov/dataset/Turbofan-Engine-Degradation-Simulation/vrks-gjie) | `datasets/LSTM/CMAPSSData/` |

---

## Prerequisites

**Host machine (Ubuntu/Debian x86-64):**

```bash
# ARM cross-compiler (Raspberry Pi 4)
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
                 cmake make python3 python3-pip minicom

# Python dependencies (training & export)
pip install torch torchvision scikit-learn pandas numpy
```

> The RISC-V toolchain is built automatically via Buildroot (see Step 2b below).

---

## Workflow

### Step 1 — Train all models

```bash
make train
```

Trains every framework/workload combination in sequence. Models are saved to `<FRAMEWORK>/benchmarks/<WORKLOAD>/model/`.

Individual frameworks can be trained separately:

```bash
make train-fann       # FANN only (MLP, Autoencoder)
make train-genann     # Genann only (MLP, Autoencoder)
make train-ncnn       # NCNN (MLP, Autoencoder, CNN2D, LSTM)
make train-litert     # LiteRT (MLP, Autoencoder, CNN2D, LSTM)
make train-onnx       # ONNX Runtime (MLP, Autoencoder, CNN2D, LSTM)
make train-cmsisnn    # CMSIS-NN (MLP, Autoencoder, CNN2D, LSTM) — quantizes to INT8
```

> FANN/Genann use `prepare_data.py` + `train_models.c` instead of `train.py`.
> CMSIS-NN's `train.py` quantizes to INT8 and generates `model/*_weights.h` + `model/*_params.h`.

---

### Step 2a — Build SD card image (Raspberry Pi 4)

```bash
./scripts/build.sh
```

The script automatically:
1. Downloads Raspberry Pi OS Lite (arm64, Bookworm) if needed
2. Cross-compiles the ML libraries for AArch64 using `aarch64-linux-gnu-gcc-12`
3. Cross-compiles all benchmark executables
4. Expands the rootfs, mounts the image, and injects executables into `/usr/bin/cmlles_tests/`
5. Copies trained models for each framework
6. Installs `linux-perf` and configures PMU access (`perf_event_paranoid = -1`)
7. Enables UART serial (`enable_uart=1`, `dtoverlay=disable-bt`)
8. Creates the default user (`pi`/`raspberry`)
9. Installs the `cmlles-bench.service` systemd unit (runs benchmarks at boot)

> If libraries and benchmarks are already compiled, steps 2–3 are skipped automatically.

Output image:
```
output_rpi_image/images/cmlles-raspios.img
```

### Step 2b — Cross-compile and deploy (VisionFive 2 — RISC-V)

The VisionFive 2 runs the **official StarFive Debian riscv64 image**, pre-flashed to the SD card. Buildroot is used **only** to build the `riscv64-buildroot-linux-gnu` cross-compilation toolchain — no custom OS image is generated.

```bash
./scripts/build_riscv.sh
```

The script automatically:
1. Configures Buildroot with the `visionfive2_cmlles_defconfig`
2. Builds the internal `riscv64-buildroot-linux-gnu` toolchain
3. Cross-compiles ML libraries (FANN, NCNN, LiteRT, and ONNX Runtime — best-effort)
4. Cross-compiles all benchmark executables (no CMSIS-NN on RISC-V)

> ONNX Runtime has no official riscv64 release and is compiled from source. A GLIBC compatibility shim (`scripts/cross/glibc_compat.c`) and static linking of `libstdc++`/`libgcc` are used to work around ABI mismatches between the Buildroot toolchain (GLIBC 2.38) and the target's Debian rootfs (GLIBC 2.36).

Once compiled, mount the VisionFive 2's SD card and deploy the binaries:

```bash
sudo ./scripts/redeploy_riscv_fix.sh [SDCARD_ROOT]   # default: /media/gabriel/root
```

This copies all benchmark executables and shared libraries (`libtensorflowlite_c.so`, `libonnxruntime.so`) onto the existing rootfs without modifying anything else. The benchmark runner script (`run_benchmarks_riscv.sh`) and the SysV init script (`S99cmlles`) are also deployed.

---

### Step 3 — Flash the SD card (Raspberry Pi only)

```bash
./scripts/deploy_to_sdcard.sh /dev/sdX
```

> The VisionFive 2 uses the official StarFive Debian image pre-flashed to the SD card — no flashing step is needed. Benchmarks are deployed directly via `redeploy_riscv_fix.sh` (Step 2b).

Identify the correct device:
```bash
lsblk -d -o NAME,SIZE,MODEL,TRAN | grep -v loop
```

---

### Step 4 — Connect via UART serial

**Raspberry Pi 4:**

| USB-Serial Adapter | RPi 4B (GPIO) |
|---|---|
| TX | Pin 10 — RXD (GPIO 15) |
| RX | Pin 8  — TXD (GPIO 14) |
| GND | Pin 6  — GND |

> Do not connect VCC from the adapter — power comes from the RPi's USB-C supply.

**VisionFive 2:** Use the board's built-in 40-pin GPIO UART header (same pinout convention).

```bash
minicom -D /dev/ttyUSB0 -b 115200
```

---

### Step 5 — Run benchmarks

Insert the SD card and power on the board. After boot (~20–40 s), the benchmark service starts automatically. Results appear progressively on the serial console.

**Raspberry Pi 4:**
```bash
journalctl -u cmlles-bench.service        # check logs
systemctl status cmlles-bench.service      # check status
```

**VisionFive 2 (Buildroot/BusyBox):** The SysV init script `S99cmlles` runs benchmarks at boot. Output goes to the serial console and is logged to `/tmp/cmlles_results/`.

Each benchmark configuration executes 1,000 timed single-sample inferences, collecting latency and PMU counters. On the Raspberry Pi, PMU data is collected via an external `perf stat` wrapper. On the VisionFive 2, PMU counters are read in-process via `perf_event_open` using raw SiFive U74 event codes.

---

### Step 6 — Collect results

After benchmarks complete, power off the board, insert the SD card into the host, and run:

**Raspberry Pi 4:**

```bash
# 1. Copy cmlles_results/ and latencies_rpi/ from the SD card → results/
./scripts/collect/collect_from_sdcard.sh

# 2. Aggregate per-inference PMU data (1000 measurements) → results/per_inference_perf_summary.csv
./scripts/collect/collect_per_inference_perf.sh

# 3. Parse external perf stat totals → results/perf_metrics_all.csv
./scripts/collect/collect_perf_metrics.sh

# 4. Training metrics (training_metrics.json) → results/training_metrics_all.csv
./scripts/collect/collect_training_metrics.sh

# 5. Binary size (size, objdump) → results/binary_metrics.csv
./scripts/collect/collect_binary_metrics.sh
```

**VisionFive 2 (RISC-V):**

```bash
./scripts/collect/collect_from_sdcard_riscv.sh
./scripts/collect/collect_per_inference_perf_riscv.sh
./scripts/collect/collect_perf_metrics_riscv.sh
./scripts/collect/collect_training_metrics.sh          # shared (host-side)
./scripts/collect/collect_binary_metrics_riscv.sh
```

---

### Step 7 — Generate analysis plots

The `analysis/` directory contains Python scripts to produce the paper's figures:

```bash
cd analysis/analysis_combined        # combined RPi + RISC-V plots
python plot_latency.py               # Fig. 1 — latency overview
python plot_int8.py                  # Fig. 2 — INT8 speedup over Float32
python plot_hardware.py              # Fig. 3 — IPC, L1-D miss rate, branch misses
python plot_memory.py                # memory footprint analysis
```

Platform-specific analyses are in `analysis/analysis_rasp4/` and `analysis/analysis_riscv/`.

---

## Project Structure

```
CMLLES/
├── CMSIS-NN/                         # CMSIS-NN library (ARM-only)
│   ├── CMSIS-DSP/                    #   git submodule: ARM CMSIS-DSP
│   ├── CMSIS-NN/                     #   git submodule: ARM CMSIS-NN
│   ├── CMSIS_5/                      #   git submodule: ARM CMSIS_5
│   └── benchmarks/{MLP,Autoencoder,CNN2D,LSTM}/
├── NCNN/                             # NCNN library
│   ├── ncnn/                         #   git submodule: Tencent/ncnn
│   └── benchmarks/{MLP,Autoencoder,CNN2D,LSTM}/
├── Genann/                           # Genann library
│   ├── genann/                       #   git submodule: codeplea/genann
│   └── benchmarks/{MLP,Autoencoder}/
├── ONNX/                             # ONNX Runtime
│   ├── onnxruntime-bak/              #   pre-built / source backup
│   └── benchmarks/{MLP,Autoencoder,CNN2D,LSTM}/
├── LiteRT/                           # LiteRT (TensorFlow Lite)
│   ├── tensorflow/                   #   git submodule: tensorflow/tensorflow
│   └── benchmarks/{MLP,Autoencoder,CNN2D,LSTM}/
├── FANN/                             # FANN library
│   ├── fann/                         #   git submodule: libfann/fann
│   └── benchmarks/{MLP,Autoencoder}/
│
├── datasets/                         # Input datasets (not tracked in git)
│   ├── MLP/{iris,IAES-dataset}/
│   ├── Autoencoder/NSL_KDD_Dataset/
│   ├── CNN-2D/coco/
│   └── LSTM/CMAPSSData/
│
├── scripts/
│   ├── build.sh                      # Full RPi image build pipeline
│   ├── build_riscv.sh                # Full VisionFive 2 image build pipeline
│   ├── env.sh                        # RPi environment variables
│   ├── env_riscv.sh                  # RISC-V environment variables
│   ├── deploy_to_sdcard.sh           # Flash image to SD card
│   ├── redeploy_riscv_fix.sh         # Redeploy binaries to RISC-V SD card
│   ├── clear.sh                      # Clean build artifacts
│   ├── cross/                        # Cross-compilation scripts & toolchain files
│   │   ├── cross_compile_libs.sh     #   RPi: compile ML libraries
│   │   ├── cross_compile_libs_riscv.sh  # RISC-V: compile ML libraries
│   │   ├── cross_compile_tests.sh    #   RPi: compile benchmark executables
│   │   ├── cross_compile_tests_riscv.sh # RISC-V: compile benchmark executables
│   │   ├── aarch64-gcc12-toolchain.cmake
│   │   ├── riscv64-toolchain.cmake
│   │   ├── build_onnxruntime_riscv.sh   # ONNX Runtime RISC-V source build
│   │   ├── glibc_compat.c            #   GLIBC compatibility shim for RISC-V
│   │   └── perf_probe.c              #   In-process PMU reader (VisionFive 2)
│   ├── collect/                      # Post-experiment data collection scripts
│   │   ├── collect_from_sdcard.sh / _riscv.sh
│   │   ├── collect_per_inference_perf.sh / _riscv.sh
│   │   ├── collect_perf_metrics.sh / _riscv.sh
│   │   ├── collect_training_metrics.sh
│   │   ├── collect_binary_metrics.sh / _riscv.sh
│   │   └── collect_latencies.sh
│   ├── image/                        # Boot-time benchmark runners
│   │   ├── run_benchmarks.sh         #   RPi runner
│   │   ├── run_benchmarks_riscv.sh   #   RISC-V runner
│   │   ├── cmlles-bench.service      #   systemd unit (RPi)
│   │   └── S99cmlles                 #   SysV init script (RISC-V / Buildroot)
│   ├── riscv/                        # RISC-V Buildroot post-build hooks
│   └── buildroot-2024.02.2/          # Buildroot source tree (RISC-V toolchain)
│
├── analysis/                         # Result analysis & plotting
│   ├── analysis_combined/            #   Cross-platform comparison plots
│   │   ├── plot_latency.py
│   │   ├── plot_int8.py
│   │   ├── plot_hardware.py
│   │   ├── plot_memory.py
│   │   └── utils.py
│   ├── analysis_rasp4/               #   RPi-only analysis
│   └── analysis_riscv/               #   RISC-V-only analysis
│
├── results/                          # Collected experimental results
│   ├── latencies_rpi/                #   1000 per-inference latencies (RPi)
│   ├── latencies_riscv/              #   1000 per-inference latencies (RISC-V)
│   ├── cmlles_results/               #   Raw perf stat output (RPi)
│   ├── cmlles_results_riscv/         #   Raw perf stat output (RISC-V)
│   ├── per_inference_perf_summary.csv      # PMU summary — RPi
│   ├── per_inference_perf_summary_riscv.csv # PMU summary — RISC-V
│   ├── perf_metrics_all.csv          #   Aggregated perf metrics — RPi
│   ├── perf_metrics_all_riscv.csv    #   Aggregated perf metrics — RISC-V
│   ├── training_metrics_all.csv      #   Training metrics (all frameworks)
│   ├── binary_metrics.csv            #   Binary sizes — RPi
│   └── binary_metrics_riscv.csv      #   Binary sizes — RISC-V
│
├── output_rpi_image/                 # RPi build output (images, sysroot, executables)
├── output_rv_image/                  # VisionFive 2 build output (Buildroot)
├── output_rt_image/                  # (legacy) RT image output
│
├── Makefile                          # Top-level: train / benchmarks targets
├── pessoal/paper/                    # Paper LaTeX source
└── .gitmodules                       # Git submodule definitions
```

### Benchmark directory layout

Each `<FRAMEWORK>/benchmarks/<WORKLOAD>/` follows a common structure:

```
train.py              # Trains and exports the model (Python)
benchmark_<algo>.c    # C inference harness (1000 timed inferences)
Makefile              # Targets: train, all (cross-compile benchmark), clean
model/                # Trained models, test_data.txt, training_metrics.json
latencies/            # Generated on-device: latencies_<label>.csv
```

**Exceptions:**
- **FANN / Genann:** Use `prepare_data.py` + `train_models.c` instead of `train.py`.
- **CMSIS-NN:** `train.py` quantizes to INT8 and generates `model/*_weights.h` + `model/*_params.h`. Each size/activation variant produces a separate executable.

---

## Results Summary

In total, **279 library/workload/variant configurations** run across both platforms (150 on the Raspberry Pi 4, 129 on the VisionFive 2), each executing 1,000 timed inferences, yielding **279,000 latency samples**.

Key findings from the paper:

- **No single library dominates** across all metrics and platforms.
- **CMSIS-NN and Genann** are the fastest and smallest-footprint options for feed-forward models (MLP, Autoencoder), with ≤ 102 KB static footprint.
- **NCNN** offers the best overall balance for CNN and LSTM workloads.
- **INT8 quantization** benefits are platform-dependent: LiteRT gains 1.1–1.5× speedup on the Cortex-A72 but slows down on RISC-V (as low as 0.30×) due to the lack of RVV SIMD.
- **VisionFive 2 is 7–18× slower** than the Raspberry Pi 4 across all workloads.
- **Library rankings do not transfer across ISAs**: NCNN loses its NEON IPC advantage on RISC-V, and CMSIS-NN is entirely unavailable.

---

## License

This repository is released for academic reproducibility. Please cite the accompanying paper if you use this infrastructure in your work.
