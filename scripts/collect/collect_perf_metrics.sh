#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/../env.sh"

BENCH_RESULTS="${OUTPUT_DIR}/cmlles_results"
mkdir -p "$RESULTS_DIR"
OUTPUT="${RESULTS_DIR}/perf_metrics_all.csv"

if [ ! -d "$BENCH_RESULTS" ]; then
    echo "Erro: ${BENCH_RESULTS} não encontrado. Execute collect_from_sdcard.sh primeiro."
    exit 1
fi

printf "benchmark,cycles,instructions,ipc,cache_references,cache_misses,cache_miss_rate_pct,\
l1_dcache_loads,l1_dcache_load_misses,l1_miss_rate_pct,\
branch_misses,context_switches,elapsed_time_s\n" > "$OUTPUT"

find "$BENCH_RESULTS" -name "*.txt" | sort | while read -r f; do
    label=$(basename "$f" .txt)
    python3 - "$f" "$label" "$OUTPUT" <<'PYEOF'
import sys, re

path, label, out = sys.argv[1:]

with open(path) as fh:
    content = fh.read()

def parse_counter(name, text):
    pat = r'([\d,]+)\s+' + re.escape(name) + r'(?:\s|$)'
    m = re.search(pat, text)
    if not m:
        return None
    return int(m.group(1).replace(',', ''))

def safe_rate(a, b):
    if a is None or b is None or b == 0:
        return ""
    return f"{100.0 * a / b:.2f}"

cycles       = parse_counter("cycles", content)
instructions = parse_counter("instructions", content)
cache_ref    = parse_counter("cache-references", content)
cache_miss   = parse_counter("cache-misses", content)
l1_loads     = parse_counter("L1-dcache-loads", content)
l1_misses    = parse_counter("L1-dcache-load-misses", content)
branch_miss  = parse_counter("branch-misses", content)
ctx_sw       = parse_counter("context-switches", content)

ipc = ""
if cycles and instructions and cycles > 0:
    ipc = f"{instructions / cycles:.4f}"

m = re.search(r'([\d.]+) seconds time elapsed', content)
elapsed = m.group(1) if m else ""

if cycles is None and instructions is None:
    sys.exit(0)

with open(out, "a") as fh:
    fh.write(f"{label},"
             f"{cycles or ''},"
             f"{instructions or ''},"
             f"{ipc},"
             f"{cache_ref or ''},"
             f"{cache_miss or ''},"
             f"{safe_rate(cache_miss, cache_ref)},"
             f"{l1_loads or ''},"
             f"{l1_misses or ''},"
             f"{safe_rate(l1_misses, l1_loads)},"
             f"{branch_miss or ''},"
             f"{ctx_sw or ''},"
             f"{elapsed}\n")
PYEOF
done

lines=$(($(wc -l < "$OUTPUT") - 1))
echo "Saved ${OUTPUT} (${lines} benchmarks com dados de perf)"
