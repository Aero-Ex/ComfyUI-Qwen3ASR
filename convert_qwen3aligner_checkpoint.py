"""Merge an HF-layout Qwen3-ForcedAligner folder into a single-file ComfyUI checkpoint.

Usage:
    python convert_qwen3aligner_checkpoint.py <model_folder> <out.safetensors>

Reads <model_folder>/model.safetensors + config.json + tokenizer.json and writes
a single safetensors file with ComfyUI-style metadata:

    qwen3aligner.format_version = "1"
    qwen3aligner.config         = contents of config.json
    qwen3aligner.tokenizer      = contents of tokenizer.json

State-dict keys are stored stripped of the leading "model." prefix, matching the
native module layout (audio_tower.*, language_model.*, multi_modal_projector.*).
The timestamp classification head ("score.weight") is kept as-is.
"""

import json
import os
import sys

FORMAT_VERSION = "1"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    model_folder, out_path = sys.argv[1], sys.argv[2]
    sd_path = os.path.join(model_folder, "model.safetensors")
    cfg_path = os.path.join(model_folder, "config.json")
    tok_path = os.path.join(model_folder, "tokenizer.json")
    for p in (sd_path, cfg_path, tok_path):
        if not os.path.isfile(p):
            raise SystemExit("Missing file: {}".format(p))

    with open(cfg_path, "r", encoding="utf-8") as f:
        config_raw = f.read()
    config = json.loads(config_raw)
    arch = config.get("architectures", [])
    if "Qwen3ASRForTokenClassification" not in arch:
        raise SystemExit(
            "Not a Qwen3-ForcedAligner checkpoint (architectures={}). "
            "Transcription checkpoints use convert_qwen3asr_checkpoint.py.".format(arch)
        )
    with open(tok_path, "r", encoding="utf-8") as f:
        tokenizer_raw = f.read()

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    with safe_open(sd_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key == "score.weight":
                tensors[key] = f.get_tensor(key)
            elif key.startswith("model."):
                tensors[key[len("model."):]] = f.get_tensor(key)
            else:
                raise SystemExit("Unexpected top-level key: {}".format(key))

    if "language_model.embed_tokens.weight" not in tensors or "audio_tower.conv2d1.weight" not in tensors:
        raise SystemExit("Not a Qwen3-ForcedAligner checkpoint: missing expected keys.")
    if "score.weight" not in tensors:
        raise SystemExit("Not a Qwen3-ForcedAligner checkpoint: missing score.weight.")

    metadata = {
        "qwen3aligner.format_version": FORMAT_VERSION,
        "qwen3aligner.config": config_raw,
        "qwen3aligner.tokenizer": tokenizer_raw,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, out_path, metadata=metadata)
    size_gb = os.path.getsize(out_path) / 1e9
    print("Wrote {} ({:.2f} GB, {} tensors)".format(out_path, size_gb, len(tensors)))


if __name__ == "__main__":
    main()
