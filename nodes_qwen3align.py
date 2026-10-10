import json

import folder_paths
import comfy.audio
import comfy.utils
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import torch

from .qwen3asr.features import extract_input_features, audio_token_length
from .qwen3asr.align_model import Qwen3AlignerModel
from .qwen3asr.align_text import encode_timestamp, parse_timestamp
from .qwen3asr.tokenizer import QwenBpeTokenizer


class Qwen3AlignerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "qwen3aligner_name": (folder_paths.get_filename_list("qwen3asr"), {
                "tooltip": "Single-file Qwen3-ForcedAligner checkpoint (see convert_qwen3aligner_checkpoint.py)."}),
        }}

    RETURN_TYPES = ("QWEN3ALIGNER",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "Qwen3ASR"

    def execute(self, qwen3aligner_name):
        model_path = folder_paths.get_full_path_or_raise("qwen3asr", qwen3aligner_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)
        metadata = metadata or {}

        if (metadata.get("qwen3aligner.format_version") != "1"
                or "language_model.embed_tokens.weight" not in sd
                or "audio_tower.conv2d1.weight" not in sd
                or "score.weight" not in sd):
            raise RuntimeError(
                "ERROR: not a single-file Qwen3-ForcedAligner checkpoint. "
                "Merge the model folder first: "
                "python convert_qwen3aligner_checkpoint.py <model_folder> <out.safetensors>"
            )

        config = json.loads(metadata["qwen3aligner.config"])
        num_classes = sd["score.weight"].shape[0]

        dtype = comfy.model_management.unet_dtype(model_params=-1, supported_dtypes=[torch.bfloat16, torch.float32])
        manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, comfy.model_management.get_torch_device(), supported_dtypes=[torch.bfloat16, torch.float32])
        quant = comfy.utils.detect_layer_quantization(sd, "")
        if quant is not None:
            operations = comfy.ops.mixed_precision_ops(quant, dtype)
        else:
            operations = comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=comfy.model_management.get_torch_device())
        model = Qwen3AlignerModel(config, num_classes, device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=operations)
        model.model.load_state_dict(sd, strict=True)
        model.aligner_tokenizer = QwenBpeTokenizer(json.loads(metadata["qwen3aligner.tokenizer"]))
        model.aligner_config = config
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        return (patcher,)


class Qwen3ForcedAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("QWEN3ALIGNER", {"tooltip": "Loaded Qwen3-ForcedAligner model."}),
            "audio": ("AUDIO", {"tooltip": "Audio to align (resampled to 16kHz mono)."}),
            "transcription": ("STRING", {"multiline": True, "tooltip": "Transcript matching the audio."}),
        }, "optional": {
            "language": ("STRING", {"default": "English", "tooltip": "Language of the transcript (e.g. English/en)."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("words_json", "word_srt")
    FUNCTION = "execute"
    CATEGORY = "Qwen3ASR"

    @staticmethod
    def _format_timestamp(sec):
        sec = max(0.0, float(sec))
        ms = int(round((sec - int(sec)) * 1000.0))
        total = int(sec)
        if ms == 1000:
            total += 1
            ms = 0
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return "{:02d}:{:02d}:{:02d},{:03d}".format(h, m, s, ms)

    def align_mono(self, wrapper, tokenizer, mono, text, language, load_device, target_dtype):
        max_sec = wrapper.num_classes * wrapper.timestamp_segment_time / 1000.0
        if mono.shape[0] / 16000.0 > max_sec:
            raise RuntimeError(
                "Audio is {:.1f}s, aligner supports {:.1f}s per pass. "
                "Split the audio (and matching transcript) first.".format(
                    mono.shape[0] / 16000.0, max_sec))
        word_list, body = encode_timestamp(text, language or "English")
        if not word_list:
            return []
        feats, mask = extract_input_features(mono)
        n_tok = int(audio_token_length(mask.sum(-1)).item())
        full_text = "<|audio_start|>" + "<|audio_pad|>" * n_tok + "<|audio_end|>" + body
        ids = torch.tensor([tokenizer._encode_str(full_text)], dtype=torch.long)
        feats = feats.to(load_device, dtype=target_dtype)
        mask = mask.to(load_device)
        ids = ids.to(load_device)
        with torch.inference_mode():
            audio_embeds = wrapper.model.get_audio_features(feats, mask)
            logits = wrapper.align_logits(ids, audio_embeds.reshape(-1, audio_embeds.shape[-1]))
        pred = logits[0].argmax(dim=-1)
        ts_id = wrapper.timestamp_token_id
        ts_idx = (ids[0] == ts_id).nonzero(as_tuple=True)[0]
        if ts_idx.shape[0] != 2 * len(word_list):
            raise RuntimeError(
                "Timestamp slots {} != 2 * words {}. Tokenizer mismatch?".format(ts_idx.shape[0], len(word_list)))
        stamp_ms = (pred[ts_idx].to(torch.float32) * wrapper.timestamp_segment_time).cpu().numpy()
        items = parse_timestamp(word_list, stamp_ms)
        del feats, mask, ids, audio_embeds, logits, pred
        return [{"text": it["text"], "start": round(it["start_time"] / 1000.0, 3), "end": round(it["end_time"] / 1000.0, 3)} for it in items]

    def execute(self, model, audio, transcription, language="English"):
        wrapper = model.model if isinstance(model, comfy.model_patcher.ModelPatcher) else model
        load_device = comfy.model_management.get_torch_device()
        if isinstance(model, comfy.model_patcher.ModelPatcher):
            comfy.model_management.load_models_gpu([model])
        tokenizer = getattr(wrapper, "aligner_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Qwen3 aligner model has no tokenizer.")
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sr != 16000:
            waveform = comfy.audio.resample(waveform, sr, 16000)
        texts = [transcription] * waveform.shape[0] if isinstance(transcription, str) else list(transcription)
        try:
            target_dtype = getattr(wrapper, "dtype", None) or torch.bfloat16
            out = []
            for b in range(waveform.shape[0]):
                comfy.model_management.throw_exception_if_processing_interrupted()
                mono = waveform[b].reshape(-1).to(torch.float32).cpu()
                text = texts[b] if b < len(texts) else texts[-1]
                out.append(self.align_mono(wrapper, tokenizer, mono, text, language, load_device, target_dtype))
        finally:
            comfy.model_management.soft_empty_cache()
        strs = []
        srts = []
        for items in out:
            strs.append(json.dumps(items, ensure_ascii=False))
            lines = []
            for i, it in enumerate(items):
                lines.append(str(i + 1))
                lines.append("{} --> {}".format(self._format_timestamp(it["start"]), self._format_timestamp(it["end"])))
                lines.append(it["text"])
                lines.append("")
            srts.append("\n".join(lines).strip() + ("\n" if lines else ""))
        if len(out) == 1:
            return (strs[0], srts[0])
        return (strs, srts)


NODE_CLASS_MAPPINGS = {
    "Qwen3AlignerLoader": Qwen3AlignerLoader,
    "Qwen3ForcedAlign": Qwen3ForcedAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3AlignerLoader": "Load Qwen3-ForcedAligner Model",
    "Qwen3ForcedAlign": "Qwen3 Forced Align",
}
