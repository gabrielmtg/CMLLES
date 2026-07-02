import os
import numpy as np

SEED = 42
np.random.seed(SEED)

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_SCALE = 1.0 / 127.0

CNN_CONFIGS = {
    "simple": {
        "input_shape": (1, 32, 32),
        "conv_layers": [
            {"filters": 16, "kernel": 3, "stride": 1, "padding": 1},
            {"filters": 32, "kernel": 3, "stride": 1, "padding": 1},
        ],
        "pool_size": 4,
        "fc_out": 10,
    },
    "intermediate": {
        "input_shape": (1, 32, 32),
        "conv_layers": [
            {"filters": 32, "kernel": 3, "stride": 1, "padding": 1},
            {"filters": 64, "kernel": 3, "stride": 1, "padding": 1},
            {"filters": 128, "kernel": 3, "stride": 1, "padding": 1},
        ],
        "pool_size": 4,
        "fc_out": 10,
    },
    "mobilenet": {
        "input_shape": (3, 96, 96),
        "conv_layers": [
            {"filters": 32, "kernel": 3, "stride": 2, "padding": 1, "dw": False},
            {"filters": 64, "kernel": 3, "stride": 1, "padding": 1, "dw": True},
            {"filters": 128, "kernel": 3, "stride": 2, "padding": 1, "dw": True},
            {"filters": 128, "kernel": 3, "stride": 1, "padding": 1, "dw": True},
        ],
        "pool_size": 1,
        "fc_out": 10,
    },
}


def quantize_weights(W):
    abs_max = np.max(np.abs(W))
    if abs_max == 0:
        return np.zeros_like(W, dtype=np.int8), 1.0
    scale = abs_max / 127.0
    return np.clip(np.round(W / scale), -128, 127).astype(np.int8), scale


def quantize_bias(b, in_scale, w_scale):
    bias_scale = in_scale * w_scale
    if bias_scale == 0:
        return np.zeros_like(b, dtype=np.int32)
    return np.clip(np.round(b / bias_scale), -(2**31), 2**31 - 1).astype(np.int32)


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


def gen_model(cfg_name, cfg):
    in_c, in_h, in_w = cfg["input_shape"]
    conv_layers = cfg["conv_layers"]
    pool_size   = cfg["pool_size"]
    fc_out      = cfg["fc_out"]

    weights_path = os.path.join(MODEL_DIR, f"cnn_{cfg_name}_weights.h")
    params_path  = os.path.join(MODEL_DIR, f"cnn_{cfg_name}_params.h")

    all_weights = []
    all_biases  = []
    all_w_scales = []
    layer_info  = []

    cur_c, cur_h, cur_w = in_c, in_h, in_w
    in_scale = INPUT_SCALE

    for i, cl in enumerate(conv_layers):
        is_dw = cl.get("dw", False)
        f      = cl["filters"]
        k      = cl["kernel"]
        s      = cl["stride"]
        p      = cl["padding"]

        if is_dw:
            W_dw = np.random.uniform(-1.0, 1.0, (cur_c, 1, k, k)).astype(np.float32)
            b_dw = np.zeros(cur_c, dtype=np.float32)
            Wdw_q, wdw_scale = quantize_weights(W_dw)
            bdw_q = quantize_bias(b_dw, in_scale, wdw_scale)

            W_pw = np.random.uniform(-1.0, 1.0, (f, cur_c, 1, 1)).astype(np.float32)
            b_pw = np.zeros(f, dtype=np.float32)
            Wpw_q, wpw_scale = quantize_weights(W_pw)
            bpw_q = quantize_bias(b_pw, wdw_scale, wpw_scale)

            all_weights.extend([Wdw_q, Wpw_q])
            all_biases.extend([bdw_q, bpw_q])
            all_w_scales.extend([wdw_scale, wpw_scale])

            out_h = (cur_h + 2 * p - k) // s + 1
            out_w = (cur_w + 2 * p - k) // s + 1
            layer_info.append({"type": "dw_sep", "in_c": cur_c, "out_c": f,
                                "k": k, "s": s, "p": p,
                                "in_h": cur_h, "in_w": cur_w,
                                "out_h": out_h, "out_w": out_w})
            cur_c, cur_h, cur_w = f, out_h, out_w
            in_scale = wpw_scale
        else:
            W = np.random.uniform(-1.0, 1.0, (f, cur_c, k, k)).astype(np.float32)
            b = np.zeros(f, dtype=np.float32)
            Wq, w_scale = quantize_weights(W)
            bq = quantize_bias(b, in_scale, w_scale)

            all_weights.append(Wq)
            all_biases.append(bq)
            all_w_scales.append(w_scale)

            out_h = (cur_h + 2 * p - k) // s + 1
            out_w = (cur_w + 2 * p - k) // s + 1
            layer_info.append({"type": "conv", "in_c": cur_c, "out_c": f,
                                "k": k, "s": s, "p": p,
                                "in_h": cur_h, "in_w": cur_w,
                                "out_h": out_h, "out_w": out_w})
            cur_c, cur_h, cur_w = f, out_h, out_w
            in_scale = w_scale

    pooled_h = pool_size
    pooled_w = pool_size
    fc_in = cur_c * pooled_h * pooled_w

    W_fc = np.random.uniform(-1.0, 1.0, (fc_out, fc_in)).astype(np.float32)
    b_fc = np.zeros(fc_out, dtype=np.float32)
    Wfc_q, wfc_scale = quantize_weights(W_fc)
    bfc_q = quantize_bias(b_fc, in_scale, wfc_scale)

    with open(weights_path, "w") as f_out:
        guard = f"CNN_{cfg_name.upper()}_WEIGHTS_H"
        f_out.write(f"#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n\n")
        for i, (W, B) in enumerate(zip(all_weights, all_biases)):
            f_out.write(array_to_c(f"conv{i}_weights", W, "int8_t") + "\n\n")
            f_out.write(array_to_c(f"conv{i}_bias", B, "int32_t") + "\n\n")
        f_out.write(array_to_c("fc_weights", Wfc_q, "int8_t") + "\n\n")
        f_out.write(array_to_c("fc_bias", bfc_q, "int32_t") + "\n\n")
        f_out.write("#endif\n")

    with open(params_path, "w") as f_out:
        guard = f"CNN_{cfg_name.upper()}_PARAMS_H"
        f_out.write(f"#ifndef {guard}\n#define {guard}\n\n")
        in_c0, in_h0, in_w0 = cfg["input_shape"]
        f_out.write(f"#define IN_CHANNELS {in_c0}\n")
        f_out.write(f"#define IN_HEIGHT   {in_h0}\n")
        f_out.write(f"#define IN_WIDTH    {in_w0}\n")
        f_out.write(f"#define POOL_SIZE   {pool_size}\n")
        f_out.write(f"#define FC_IN_SIZE  {fc_in}\n")
        f_out.write(f"#define FC_OUT_SIZE {fc_out}\n")
        f_out.write(f"#define NUM_CONV_LAYERS {len(all_weights)}\n\n")

        mults = [int(round(s * (1 << 20))) for s in all_w_scales]
        f_out.write(f"static const int32_t CONV_MULTIPLIERS[] = {{")
        f_out.write(", ".join(str(m) for m in mults))
        f_out.write("};\n")
        f_out.write(f"static const int CONV_SHIFTS[] = {{")
        f_out.write(", ".join("-20" for _ in mults))
        f_out.write("};\n")
        f_out.write(f"#define FC_MULTIPLIER {int(round(wfc_scale * (1 << 20)))}\n")
        f_out.write(f"#define FC_SHIFT -20\n\n")

        f_out.write(f"#define NUM_LAYER_INFO {len(layer_info)}\n")
        f_out.write("typedef struct { int type; int in_c, out_c, k, s, p, in_h, in_w, out_h, out_w; } LayerInfo;\n")
        f_out.write("static const LayerInfo LAYER_INFO[] = {\n")
        for li in layer_info:
            t = 0 if li["type"] == "conv" else 1
            f_out.write(f"    {{{t}, {li['in_c']}, {li['out_c']}, {li['k']}, {li['s']}, {li['p']}, {li['in_h']}, {li['in_w']}, {li['out_h']}, {li['out_w']}}},\n")
        f_out.write("};\n\n")
        f_out.write("#endif\n")

    print(f"Generated {weights_path}")
    print(f"Generated {params_path}")


for cfg_name, cfg in CNN_CONFIGS.items():
    gen_model(cfg_name, cfg)
