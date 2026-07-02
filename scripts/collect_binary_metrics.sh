#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

source env.sh

EXEC_DIR="${OUTPUT_DIR}/executables"
OUT="${OUTPUT_DIR}/binary_metrics.csv"

if [ ! -d "$EXEC_DIR" ]; then
    echo "Error: ${EXEC_DIR} not found. Run cross_compile_tests.sh first."
    exit 1
fi

echo "executable,text_bytes,data_bytes,bss_bytes,total_bytes" > "$OUT"

for f in "$EXEC_DIR"/*; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    read -r text data bss dec _ <<< "$(aarch64-linux-gnu-size "$f" | tail -1)"
    echo "${name},${text},${data},${bss},${dec}" >> "$OUT"
done

echo "Binary metrics written to ${OUT}"
echo ""
echo "Section details per executable:"
for f in "$EXEC_DIR"/*; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    echo "--- ${name} ---"
    aarch64-linux-gnu-objdump -h "$f" 2>/dev/null \
        | awk '/^\s+[0-9]/ {printf "  %-20s %s bytes\n", $2, strtonum("0x"$3)}' \
        | grep -E 'text|data|bss|rodata' || true
done
