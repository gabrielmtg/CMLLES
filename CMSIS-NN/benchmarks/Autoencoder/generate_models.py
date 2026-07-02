import os
import numpy as np

SEED = 42
np.random.seed(SEED)

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

AE_CONFIGS = {
    "small":  [64, 32, 16, 32, 64],
    "medium": [128, 64, 32, 64, 128],
    "large":  [256, 128, 64, 128, 256],
}
INPUT_SCALE = 1.0 / 127.0


def quantize_weights(W):
    abs_max = np.max(np.abs(W))
    if abs_max == 0:
        return np.zeros_like(W, dtype=np.int8), 1.0
    scale = abs_max / 127.0
    return np.clip(np.round(W / scale), -128, 127).astype(np.int8), scale


def quantize_bias(b, in_scale, w_scale):
    bias_scale = in_scale * w_scale
    if bias_scale == 0:
        return np.zeros_like(b, dtype=np.int32), 1.0
    return np.clip(np.round(b / bias_scale), -(2**31), 2**31 - 1).astype(np.int32), bias_scale


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


def gen_model(size_name, dims):
    n_fc = len(dims) - 1
    weights = []
    biases  = []
    w_scales = []

    in_scale = INPUT_SCALE
    for i in range(n_fc):
        W = np.random.uniform(-1.0, 1.0, (dims[i + 1], dims[i])).astype(np.float32)
        b = np.zeros(dims[i + 1], dtype=np.float32)
        Wq, w_scale = quantize_weights(W)
        bq, _       = quantize_bias(b, in_scale, w_scale)
        weights.append(Wq)
        biases.append(bq)
        w_scales.append(w_scale)
        in_scale = w_scale

    variant = f"ae_{size_name}"
    weights_path = os.path.join(MODEL_DIR, f"{variant}_weights.h")
    params_path  = os.path.join(MODEL_DIR, f"{variant}_params.h")

    with open(weights_path, "w") as f:
        guard = variant.upper() + "_WEIGHTS_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n\n")
        for i, (Wq, bq) in enumerate(zip(weights, biases)):
            f.write(array_to_c(f"fc{i}_weights", Wq, "int8_t") + "\n\n")
            f.write(array_to_c(f"fc{i}_bias", bq, "int32_t") + "\n\n")
        f.write(f"static const int8_t * const FC_WEIGHTS[] = {{")
        f.write(", ".join(f"fc{i}_weights" for i in range(n_fc)))
        f.write("};\n")
        f.write(f"static const int32_t * const FC_BIAS[] = {{")
        f.write(", ".join(f"fc{i}_bias" for i in range(n_fc)))
        f.write("};\n\n")
        f.write(f"#endif\n")

    with open(params_path, "w") as f:
        guard = variant.upper() + "_PARAMS_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"#define NUM_FC_LAYERS {n_fc}\n")
        f.write(f"static const int LAYER_SIZES[] = {{")
        f.write(", ".join(str(d) for d in dims))
        f.write("};\n")
        mults = [int(round(ws * (1 << 20))) for ws in w_scales]
        f.write(f"static const int32_t FC_MULTIPLIERS[] = {{")
        f.write(", ".join(str(m) for m in mults))
        f.write("};\n")
        f.write(f"static const int FC_SHIFTS[] = {{")
        f.write(", ".join("-20" for _ in mults))
        f.write("};\n")
        f.write(f"#define USE_SIGMOID_OUTPUT 0\n")
        f.write(f"\n#endif\n")

    print(f"Generated {weights_path}")
    print(f"Generated {params_path}")


for size, dims in AE_CONFIGS.items():
    gen_model(size, dims)
