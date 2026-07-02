import os
import numpy as np
import math

SEED = 42
np.random.seed(SEED)

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

HIDDEN_SIZES = [32, 64, 128]
INPUT_SIZE = 10
BATCH_SIZE = 1

INPUT_SCALE = 1.0 / 128.0
HIDDEN_SCALE = 1.0 / 128.0
INTERMEDIATE_SCALE = math.pow(2, -12)
CELL_SCALE_POWER = -15


def quantize_weights(W):
    abs_max = np.max(np.abs(W))
    if abs_max == 0:
        return np.zeros_like(W, dtype=np.int8), 1.0
    scale = abs_max / 127.0
    return np.clip(np.round(W / scale), -128, 127).astype(np.int8), scale


def quantize_bias(b, scale):
    return np.clip(np.round(b / scale), -(2**31), 2**31 - 1).astype(np.int32)


def quantize_scale(scale):
    significand, exp = math.frexp(scale)
    q31 = int(round(significand * (1 << 31)))
    return q31, exp


def array_to_c(name, arr, dtype):
    lines = [f"static const {dtype} {name}[] = {{"]
    flat = arr.flatten()
    row = "    "
    for i, v in enumerate(flat):
        row += f"{int(v)}, "
        if (i + 1) % 12 == 0:
            lines.append(row.rstrip())
            row = "    "
    if row.strip():
        lines.append(row.rstrip())
    lines.append("};")
    return "\n".join(lines)


GATE_KEYS = ["forget", "input", "cell", "output"]
GATE_PREFIX = {"forget": "I2F", "input": "I2I", "cell": "I2C", "output": "I2O"}
GATE_RPREFIX = {"forget": "R2F", "input": "R2I", "cell": "R2C", "output": "R2O"}

for hidden in HIDDEN_SIZES:
    weights_path = os.path.join(MODEL_DIR, f"lstm_h{hidden}_weights.h")
    params_path = os.path.join(MODEL_DIR, f"lstm_h{hidden}_params.h")

    gate_data = {}
    for g in GATE_KEYS:
        W = np.random.uniform(-1.0, 1.0, (hidden, INPUT_SIZE)).astype(np.float32)
        R = np.random.uniform(-1.0, 1.0, (hidden, hidden)).astype(np.float32)
        b = np.random.uniform(-0.1, 0.1, (hidden,)).astype(np.float32)

        W_q, w_scale = quantize_weights(W)
        R_q, r_scale = quantize_weights(R)
        b_q = quantize_bias(b, INPUT_SCALE * w_scale)

        eff_in = b_q.copy()
        eff_hid = np.zeros(hidden, dtype=np.int32)

        i2g_eff = INPUT_SCALE * w_scale / INTERMEDIATE_SCALE
        r2g_eff = HIDDEN_SCALE * r_scale / INTERMEDIATE_SCALE
        i_mult, i_shift = quantize_scale(i2g_eff)
        r_mult, r_shift = quantize_scale(r2g_eff)

        gate_data[g] = {
            "W_q": W_q, "R_q": R_q,
            "eff_in": eff_in, "eff_hid": eff_hid,
            "i_mult": i_mult, "i_shift": i_shift,
            "r_mult": r_mult, "r_shift": r_shift,
        }

    forget_to_cell_scale = math.pow(2, -15)
    input_to_cell_scale = math.pow(2, -15)
    ftc_mult, ftc_shift = quantize_scale(forget_to_cell_scale)
    itc_mult, itc_shift = quantize_scale(input_to_cell_scale)

    eff_hidden_scale = math.pow(2, -15) / HIDDEN_SCALE * math.pow(2, -15)
    out_mult, out_shift = quantize_scale(eff_hidden_scale)

    with open(weights_path, "w") as f:
        guard = f"LSTM_H{hidden}_WEIGHTS_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n\n")
        for g in GATE_KEYS:
            d = gate_data[g]
            f.write(array_to_c(f"{g}_input_weights", d["W_q"], "int8_t") + "\n\n")
            f.write(array_to_c(f"{g}_hidden_weights", d["R_q"], "int8_t") + "\n\n")
            f.write(array_to_c(f"{g}_eff_bias_input", d["eff_in"], "int32_t") + "\n\n")
            f.write(array_to_c(f"{g}_eff_bias_hidden", d["eff_hid"], "int32_t") + "\n\n")
        f.write("#endif\n")

    with open(params_path, "w") as f:
        guard = f"LSTM_H{hidden}_PARAMS_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"#define LSTM_INPUT_SIZE {INPUT_SIZE}\n")
        f.write(f"#define LSTM_HIDDEN_SIZE {hidden}\n")
        f.write(f"#define LSTM_BATCH_SIZE {BATCH_SIZE}\n")
        f.write(f"#define LSTM_CELL_SCALE_POWER {CELL_SCALE_POWER}\n")
        f.write(f"#define LSTM_CELL_CLIP 32767\n")
        f.write(f"#define LSTM_INPUT_OFFSET 0\n")
        f.write(f"#define LSTM_OUTPUT_OFFSET 0\n")
        f.write(f"#define LSTM_FORGET_TO_CELL_MULT {ftc_mult}\n")
        f.write(f"#define LSTM_FORGET_TO_CELL_SHIFT {ftc_shift}\n")
        f.write(f"#define LSTM_INPUT_TO_CELL_MULT {itc_mult}\n")
        f.write(f"#define LSTM_INPUT_TO_CELL_SHIFT {itc_shift}\n")
        f.write(f"#define LSTM_OUTPUT_MULT {out_mult}\n")
        f.write(f"#define LSTM_OUTPUT_SHIFT {out_shift}\n")
        for g in GATE_KEYS:
            d = gate_data[g]
            ip = GATE_PREFIX[g]
            rp = GATE_RPREFIX[g]
            f.write(f"#define LSTM_{ip}_MULT {d['i_mult']}\n")
            f.write(f"#define LSTM_{ip}_SHIFT {d['i_shift']}\n")
            f.write(f"#define LSTM_{rp}_MULT {d['r_mult']}\n")
            f.write(f"#define LSTM_{rp}_SHIFT {d['r_shift']}\n")
        f.write("#endif\n")

    print(f"Generated {weights_path}")
    print(f"Generated {params_path}")
