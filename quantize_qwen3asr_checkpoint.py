"""Quantize a single-file Qwen3-ASR checkpoint to ComfyUI int8_tensorwise.

Usage:
    python quantize_qwen3asr_checkpoint.py <in.safetensors> <out.safetensors> [--convrot]

Every 2D Linear/Embedding weight is stored as symmetric INT8 plus scale(s),
using comfy-kit's TensorWiseINT8Layout (the same kernels Comfy uses to run
it). Biases, norms, convs and buffers stay bf16. With --convrot, weights get
a Hadamard rotation + per-row scales (lower error on outlier-heavy layers);
layers whose width isn't divisible by 256 or 64 fall back to plain per-tensor
INT8.

Not bit-exact: rounding error is at most half an LSB. Acceptance is transcript
parity with the bf16 checkpoint.
"""

import json
import os
import sys

import torch

QUANT_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "out_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    "fc1.weight", "fc2.weight", "conv_out.weight",
    "linear_1.weight", "linear_2.weight", "embed_tokens.weight",
)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    use_convrot = "--convrot" in sys.argv[3:]

    from safetensors import safe_open
    from safetensors.torch import save_file
    try:
        from comfy.quant_ops import TensorWiseINT8Layout
    except ImportError:  # standalone use without ComfyUI on sys.path
        from comfy_kitchen.tensor import TensorWiseINT8Layout

    out = {}
    n_quant = 0
    n_rot = 0
    worst_rel = 0.0
    with safe_open(in_path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        if (metadata.get("qwen3asr.format_version") != "1"
                or "language_model.embed_tokens.weight" not in f.keys()):
            raise SystemExit("Not a single-file Qwen3-ASR checkpoint: {}".format(in_path))
        for key in f.keys():
            t = f.get_tensor(key)
            if key.endswith(QUANT_SUFFIXES) and t.dim() == 2:
                w = t.float()
                peak = float(w.abs().max().item())
                conf = {"format": "int8_tensorwise"}
                if peak == 0.0:
                    qdata = torch.zeros_like(w, dtype=torch.int8)
                    scale = torch.tensor(1.0, dtype=torch.float32)
                else:
                    gs = 0
                    if use_convrot:
                        # Kitchen's Hadamard needs a power-of-4 group dividing the width.
                        gs = next((g for g in (256, 64) if w.shape[1] % g == 0), 0)
                    if gs:
                        qdata, params = TensorWiseINT8Layout.quantize(
                            w, stochastic_rounding=0, per_channel=True,
                            convrot=True, convrot_groupsize=gs)
                        conf["convrot"] = True
                        conf["convrot_groupsize"] = gs
                        n_rot += 1
                    else:
                        qdata, params = TensorWiseINT8Layout.quantize(w, stochastic_rounding=0)
                    qdata = qdata.to(torch.int8)
                    scale = params.scale.float()
                    deq = TensorWiseINT8Layout.dequantize(qdata, params).float()
                    err = float(((deq - w).abs().max() / peak).item())
                    worst_rel = max(worst_rel, err)
                stem = key[:-len("weight")]
                out[key] = qdata.contiguous()
                out[stem + "weight_scale"] = scale.contiguous()
                out[stem + "comfy_quant"] = torch.tensor(
                    list(json.dumps(conf).encode("utf-8")),
                    dtype=torch.uint8)
                n_quant += 1
            else:
                out[key] = t.contiguous() if not t.is_contiguous() else t

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_file(out, out_path, metadata=metadata)
    size_gb = os.path.getsize(out_path) / 1e9
    print("Wrote {} ({:.2f} GB, {} tensors, {} quantized ({} rotated), worst rel err {:.4f})".format(
        out_path, size_gb, len(out), n_quant, n_rot, worst_rel))


if __name__ == "__main__":
    main()
