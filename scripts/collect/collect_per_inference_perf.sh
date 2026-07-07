#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/../env.sh"

LATENCIES_DIR="${OUTPUT_DIR}/latencies_rpi"
mkdir -p "$RESULTS_DIR"
OUTPUT="${RESULTS_DIR}/per_inference_perf_summary.csv"

if [ ! -d "$LATENCIES_DIR" ]; then
    echo "Erro: ${LATENCIES_DIR} não encontrado. Execute collect_from_sdcard.sh primeiro."
    exit 1
fi

printf "benchmark,n_runs,\
cycles_sum,cycles_mean,cycles_std,cycles_p95,cycles_p99,\
instructions_sum,instructions_mean,instructions_std,\
l1_loads_sum,l1_loads_mean,\
l1_misses_sum,l1_misses_mean,l1_miss_rate_pct,\
branch_misses_sum,branch_misses_mean,\
ipc\n" > "$OUTPUT"

find "$LATENCIES_DIR" -name "latencies_*.csv" | sort | while read -r f; do
    python3 - "$f" "$OUTPUT" <<'PYEOF'
import sys, csv
import statistics

path, out = sys.argv[1:]

rows = []
with open(path, newline='') as fh:
    reader = csv.DictReader(fh)
    if 'cycles' not in (reader.fieldnames or []):
        sys.exit(0)
    for row in reader:
        try:
            if int(row['run']) == 1:
                continue
            rows.append({
                'cycles':        int(row['cycles']),
                'instructions':  int(row['instructions']),
                'l1_loads':      int(row['l1_loads']),
                'l1_misses':     int(row['l1_misses']),
                'branch_misses': int(row['branch_misses']),
                'label':         row['label'],
            })
        except (ValueError, KeyError):
            pass

if not rows:
    sys.exit(0)

def col(key):
    return [r[key] for r in rows]

def pct(arr, p):
    arr_s = sorted(arr)
    return arr_s[int(p * len(arr_s))]

cyc   = col('cycles')
ins   = col('instructions')
l1l   = col('l1_loads')
l1m   = col('l1_misses')
brm   = col('branch_misses')
label = rows[0]['label']
n     = len(rows)

cyc_sum  = sum(cyc)
ins_sum  = sum(ins)
l1l_sum  = sum(l1l)
l1m_sum  = sum(l1m)
brm_sum  = sum(brm)

ipc = f"{ins_sum / cyc_sum:.4f}" if cyc_sum > 0 else ""
l1_miss_rate = f"{100.0 * l1m_sum / l1l_sum:.2f}" if l1l_sum > 0 else ""

with open(out, 'a') as fh:
    fh.write(
        f"{label},{n},"
        f"{cyc_sum},{sum(cyc)/n:.1f},{statistics.stdev(cyc):.1f},"
        f"{pct(cyc,0.95):.0f},{pct(cyc,0.99):.0f},"
        f"{ins_sum},{sum(ins)/n:.1f},{statistics.stdev(ins):.1f},"
        f"{l1l_sum},{sum(l1l)/n:.1f},"
        f"{l1m_sum},{sum(l1m)/n:.1f},{l1_miss_rate},"
        f"{brm_sum},{sum(brm)/n:.1f},"
        f"{ipc}\n"
    )
PYEOF
done

lines=$(($(wc -l < "$OUTPUT") - 1))
echo "Saved ${OUTPUT} (${lines} benchmarks)"
