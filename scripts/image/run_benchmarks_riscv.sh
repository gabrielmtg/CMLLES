#!/bin/sh

BASE="/usr/bin/cmlles_tests"
RESULTS="/var/log/cmlles_results"
mkdir -p "$RESULTS"

# Unlike the RPi image (where this is persisted via /etc/sysctl.conf at build
# time), there is no static sysctl config being maintained here - set it at
# runtime instead.
sysctl -w kernel.perf_event_paranoid=-1 2>/dev/null || \
    echo -1 > /proc/sys/kernel/perf_event_paranoid 2>/dev/null || true

PERF_EVENTS="cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
branch-misses,context-switches"
# Fallback list if the full event set above isn't supported by the JH7110's
# PMU driver (unverified without real hardware - cache/branch events may not
# be wired up, unlike cycles/instructions which are always available).
PERF_EVENTS_FALLBACK="cycles,instructions"

HAVE_PERF=0
PERF_CMD=""
if command -v perf >/dev/null 2>&1; then
    PERF_CMD="perf"
    HAVE_PERF=1
else
    PERF_CMD=$(ls /usr/bin/perf_* 2>/dev/null | sort -V | tail -1)
    [ -n "$PERF_CMD" ] && HAVE_PERF=1
fi

run_perf() {
    local dir="$1"
    local exec="$2"
    local args="$3"
    local label="$4"
    cd "${BASE}/${dir}"
    printf "=== %s ===\n" "$label" | tee -a "${RESULTS}/${label}.txt"
    if [ "$HAVE_PERF" -eq 1 ]; then
        if ! "$PERF_CMD" stat -e "$PERF_EVENTS" ./"$exec" $args 2>&1 \
            | tee -a "${RESULTS}/${label}.txt"; then
            echo "  (full event set failed, retrying with fallback: ${PERF_EVENTS_FALLBACK})" \
                | tee -a "${RESULTS}/${label}.txt"
            "$PERF_CMD" stat -e "$PERF_EVENTS_FALLBACK" ./"$exec" $args 2>&1 \
                | tee -a "${RESULTS}/${label}.txt"
        fi
    else
        ./"$exec" $args 2>&1 \
            | tee -a "${RESULTS}/${label}.txt"
    fi
}

HAVE_ONNX=0
[ -x "${BASE}/ONNX/benchmarks/MLP/Iris/benchmark_mlp" ] && HAVE_ONNX=1

sleep 5

for size in small medium large; do
    for act in relu sigmoid; do
        run_perf "FANN/benchmarks/MLP/Iris"     "benchmark_mlp"                "$size $act"      "fann_mlp_${size}_${act}"
        run_perf "NCNN/benchmarks/MLP/Iris"     "benchmark_mlp"                "$size $act"      "ncnn_mlp_${size}_${act}"
        run_perf "LiteRT/benchmarks/MLP/Iris"   "benchmark_mlp"                "$size $act"      "litert_mlp_${size}_${act}"
        run_perf "LiteRT/benchmarks/MLP/Iris"   "benchmark_mlp"                "$size $act int8" "litert_mlp_${size}_${act}_int8"
        run_perf "FANN/benchmarks/MLP/IAES"     "benchmark_iaes"               "$size $act"      "fann_iaes_${size}_${act}"
        run_perf "NCNN/benchmarks/MLP/IAES"     "benchmark_iaes"               "$size $act"      "ncnn_iaes_${size}_${act}"
        run_perf "LiteRT/benchmarks/MLP/IAES"   "benchmark_iaes"               "$size $act"      "litert_iaes_${size}_${act}"
        run_perf "LiteRT/benchmarks/MLP/IAES"   "benchmark_iaes"               "$size $act int8" "litert_iaes_${size}_${act}_int8"
        if [ "$HAVE_ONNX" -eq 1 ]; then
            run_perf "ONNX/benchmarks/MLP/Iris"  "benchmark_mlp"  "$size $act"      "onnx_mlp_${size}_${act}"
            run_perf "ONNX/benchmarks/MLP/Iris"  "benchmark_mlp"  "$size $act int8" "onnx_mlp_${size}_${act}_int8"
            run_perf "ONNX/benchmarks/MLP/IAES"  "benchmark_iaes" "$size $act"      "onnx_iaes_${size}_${act}"
            run_perf "ONNX/benchmarks/MLP/IAES"  "benchmark_iaes" "$size $act int8" "onnx_iaes_${size}_${act}_int8"
        fi
    done
    run_perf "Genann/benchmarks/MLP/Iris" "benchmark_mlp"  "$size sigmoid" "genann_mlp_${size}_sigmoid"
    run_perf "Genann/benchmarks/MLP/IAES" "benchmark_iaes" "$size"         "genann_iaes_${size}_sigmoid"
done

for size in small medium large; do
    run_perf "FANN/benchmarks/Autoencoder"     "benchmark_ae"         "$size"      "fann_ae_${size}"
    run_perf "Genann/benchmarks/Autoencoder"   "benchmark_ae"         "$size"      "genann_ae_${size}"
    run_perf "NCNN/benchmarks/Autoencoder"     "benchmark_ae"         "$size"      "ncnn_ae_${size}"
    run_perf "LiteRT/benchmarks/Autoencoder"   "benchmark_ae"         "$size"      "litert_ae_${size}"
    run_perf "LiteRT/benchmarks/Autoencoder"   "benchmark_ae"         "$size int8" "litert_ae_${size}_int8"
    if [ "$HAVE_ONNX" -eq 1 ]; then
        run_perf "ONNX/benchmarks/Autoencoder" "benchmark_ae" "$size"      "onnx_ae_${size}"
        run_perf "ONNX/benchmarks/Autoencoder" "benchmark_ae" "$size int8" "onnx_ae_${size}_int8"
    fi
done

for cfg in simple intermediate mobilenet; do
    run_perf "NCNN/benchmarks/CNN2D"     "benchmark_cnn2d"       "$cfg"      "ncnn_cnn2d_${cfg}"
    run_perf "LiteRT/benchmarks/CNN2D"   "benchmark_cnn2d"       "$cfg"      "litert_cnn2d_${cfg}"
    run_perf "LiteRT/benchmarks/CNN2D"   "benchmark_cnn2d"       "$cfg int8" "litert_cnn2d_${cfg}_int8"
    if [ "$HAVE_ONNX" -eq 1 ]; then
        run_perf "ONNX/benchmarks/CNN2D" "benchmark_cnn2d" "$cfg"      "onnx_cnn2d_${cfg}"
        run_perf "ONNX/benchmarks/CNN2D" "benchmark_cnn2d" "$cfg int8" "onnx_cnn2d_${cfg}_int8"
    fi
done

for h in 32 64 128; do
    run_perf "NCNN/benchmarks/LSTM"     "benchmark_lstm"       "$h"      "ncnn_lstm_h${h}"
    run_perf "LiteRT/benchmarks/LSTM"   "benchmark_lstm"       "$h"      "litert_lstm_h${h}"
    run_perf "LiteRT/benchmarks/LSTM"   "benchmark_lstm"       "$h int8" "litert_lstm_h${h}_int8"
    if [ "$HAVE_ONNX" -eq 1 ]; then
        run_perf "ONNX/benchmarks/LSTM" "benchmark_lstm" "$h"      "onnx_lstm_h${h}"
        run_perf "ONNX/benchmarks/LSTM" "benchmark_lstm" "$h int8" "onnx_lstm_h${h}_int8"
    fi
done

echo "Benchmarks complete. Results in $RESULTS"
