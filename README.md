# ComfyUI-Qwen3ASR

Native, dependency-free Qwen3-ASR speech-to-text for ComfyUI.

## Install

Manual:

```
cd ComfyUI/custom_nodes
git clone https://github.com/Aero-Ex/ComfyUI-Qwen3ASR
```

Restart ComfyUI. No installation needed.

## Checkpoints

Pre-converted single-file checkpoints: https://huggingface.co/Aero-Ex/ComfyUI-Qwen3-ASR/tree/main

Download the `qwen3asr` folder contents into `ComfyUI/models/qwen3asr/` (0.6B ASR, 1.7 ASR + optional forced aligner for word timestamps).

## Usage

`Load Qwen3-ASR Model` -> `Qwen3-ASR Transcribe`. Leave `language` empty for auto-detect, or force with `English`/`en`.
