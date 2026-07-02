#!/bin/sh

OUTPUT="training_metrics_all.csv"
printf "framework,algorithm,model,training_time_s,final_test_accuracy_pct,final_test_mae_cycles,final_test_rmse_cycles,test_samples\n" > "$OUTPUT"

find . -path "*/benchmarks/*/model/training_metrics.json" | sort | while read f; do
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
        t  = m.get("training_time_s", "")
        ac = m.get("final_test_accuracy_pct", "")
        mae  = m.get("final_test_mae_cycles", "")
        rmse = m.get("final_test_rmse_cycles", "")
        ns = m.get("test_samples", "")
        fh.write(f"{fw},{algo},{model},{t},{ac},{mae},{rmse},{ns}\n")
PYEOF
done

echo "Saved $OUTPUT"
