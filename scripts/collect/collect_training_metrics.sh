#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/../.."
source scripts/env.sh

mkdir -p "$RESULTS_DIR"
OUTPUT="${RESULTS_DIR}/training_metrics_all.csv"
printf "framework,algorithm,model,training_time_s,epochs,final_train_loss,final_test_accuracy_pct,test_samples,threshold\n" > "$OUTPUT"

find . -path "./output_rpi_image" -prune -o \
        -path "./output_rv_image" -prune -o \
        -path "*/benchmarks/*/model/training_metrics.json" -print | sort | while read -r f; do
    parts=$(echo "$f" | awk -F'/' '{print $2, $4}')
    framework=$(echo "$parts" | cut -d' ' -f1)
    algorithm=$(echo "$parts" | cut -d' ' -f2)
    python3 - "$f" "$framework" "$algorithm" "$OUTPUT" <<'PYEOF'
import sys, json

path, fw, algo, out = sys.argv[1:]
with open(path) as fh:
    data = json.load(fh)
with open(out, "a") as fh:
    for model, m in data.items():
        t    = m.get("training_time_s", "")
        ep   = m.get("epochs", m.get("epochs_trained", ""))
        loss = m.get("final_train_loss", m.get("final_train_mse", m.get("best_val_loss", "")))
        acc  = m.get("final_test_accuracy_pct", "")
        ns   = m.get("test_samples", "")
        thr  = m.get("threshold", "")
        fh.write(f"{fw},{algo},{model},{t},{ep},{loss},{acc},{ns},{thr}\n")
PYEOF
done

echo "Saved $OUTPUT"
