import json

import folder_paths
import comfy.audio
import comfy.utils
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import torch

from .qwen3asr.features import extract_input_features, audio_token_length
from .qwen3asr.align_text import is_cjk_char, is_kana_char, join_cues_text, format_timestamp, format_srt_cues
from .qwen3asr.model import Qwen3ASRModel
from .qwen3asr.prompt import build_prompt_text, resolve_language
from .qwen3asr.tokenizer import QwenBpeTokenizer, parse_output
from .nodes_qwen3align import Qwen3ForcedAlign


class Qwen3ASRLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "qwen3asr_name": (folder_paths.get_filename_list("qwen3asr"), {
                "tooltip": "Single-file Qwen3-ASR checkpoint (see Model Links in README)."}),
        }}

    RETURN_TYPES = ("QWEN3ASR",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "Qwen3ASR"

    def execute(self, qwen3asr_name):
        model_path = folder_paths.get_full_path_or_raise("qwen3asr", qwen3asr_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)
        metadata = metadata or {}

        if (metadata.get("qwen3asr.format_version") != "1"
                or "language_model.embed_tokens.weight" not in sd
                or "audio_tower.conv2d1.weight" not in sd):
            raise RuntimeError(
                "ERROR: not a single-file Qwen3-ASR checkpoint. "
                "Download a pre-converted one: "
                "https://huggingface.co/Aero-Ex/ComfyUI-Qwen3ASR"
            )

        config = json.loads(metadata["qwen3asr.config"])

        dtype = comfy.model_management.unet_dtype(model_params=-1, supported_dtypes=[torch.bfloat16, torch.float32])
        manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, comfy.model_management.get_torch_device(), supported_dtypes=[torch.bfloat16, torch.float32])
        quant = comfy.utils.detect_layer_quantization(sd, "")
        if quant is not None:
            operations = comfy.ops.mixed_precision_ops(quant, dtype)
        else:
            operations = comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=comfy.model_management.get_torch_device())
        model = Qwen3ASRModel(config, device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=operations)
        model.model.load_state_dict(sd, strict=True)
        model.asr_tokenizer = QwenBpeTokenizer(json.loads(metadata["qwen3asr.tokenizer"]))
        model.asr_config = config
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        size = comfy.model_management.module_size(model.model)
        patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device, size=size)
        return (patcher,)


class Qwen3ASRTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("QWEN3ASR", {"tooltip": "Loaded Qwen3-ASR model."}),
            "audio": ("AUDIO", {"tooltip": "Audio to transcribe (resampled to 16kHz mono)."}),
            "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
            "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
        }, "optional": {
            "language": ("STRING", {"default": "", "tooltip": "Force language (name or code, e.g. English/en). Empty for auto."}),
            "system_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional context/hotwords bias."}),
            "chunk_seconds": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 1200.0, "step": 5.0, "tooltip": "Split long audio into chunks of this length at a low-energy boundary. 0 disables chunking."}),
            "aligner": ("QWEN3ALIGNER", {"tooltip": "Optional forced aligner for word-level timestamps."}),
            "srt_max_words": ("INT", {"default": 8, "min": 1, "max": 30, "step": 1, "tooltip": "Max words per srt cue (aligner connected only)."}),
            "srt_max_chars": ("INT", {"default": 84, "min": 10, "max": 200, "step": 1, "tooltip": "Max characters per srt cue (aligner connected only)."}),
            "srt_max_gap": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Start a new srt cue when the gap between words exceeds this (seconds, aligner connected only)."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcription", "language", "srt", "word_srt")
    FUNCTION = "execute"
    CATEGORY = "Qwen3ASR"

    _format_timestamp = staticmethod(format_timestamp)

    @staticmethod
    def _split_sentences(text):
        import re
        text = (text or "").strip()
        if not text:
            return []
        parts = re.split(r'([.!?。\n\r]+|\s*\n+\s*)', text)
        sents = []
        buf = ""
        for p in parts:
            buf += p
            if re.search(r'[.!?。\n\r]$', buf):
                s = buf.strip()
                if s:
                    sents.append(s)
                buf = ""
        if buf.strip():
            sents.append(buf.strip())
        return sents if sents else [text]

    def _split_chunks(self, wav, sr, max_sec, search_expand_sec=5.0, min_window_ms=100.0):
        total_len = wav.shape[0]
        if total_len / float(sr) <= max_sec:
            min_len = int(0.5 * sr)
            if total_len < min_len:
                pad = min_len - total_len
                wav = torch.nn.functional.pad(wav, (0, pad))
            return [(wav, 0.0)]
        max_len = int(max_sec * sr)
        expand = int(search_expand_sec * sr)
        win = max(4, int((min_window_ms / 1000.0) * sr))
        chunks = []
        start = 0
        offset_sec = 0.0
        while (total_len - start) > max_len:
            cut = start + max_len
            left = max(start, cut - expand)
            right = min(total_len, cut + expand)
            if right - left <= win:
                boundary = cut
            else:
                seg_abs = wav[left:right].abs()
                cs = torch.cumsum(seg_abs, dim=0)
                padded = torch.cat([seg_abs.new_zeros(1), cs])
                window_sums = padded[win:] - padded[:-win]
                min_pos = int(torch.argmin(window_sums).item())
                local = seg_abs[min_pos:min_pos + win]
                inner = int(torch.argmin(local).item())
                boundary = left + min_pos + inner
            boundary = int(max(boundary, start + 1))
            boundary = int(min(boundary, total_len))
            chunk = wav[start:boundary]
            chunks.append((chunk, offset_sec))
            offset_sec += (boundary - start) / float(sr)
            start = boundary
        tail = wav[start:total_len]
        chunks.append((tail, offset_sec))
        min_len = int(0.5 * sr)
        padded_chunks = []
        for c, off in chunks:
            if c.shape[0] < min_len:
                pad = min_len - c.shape[0]
                c = torch.nn.functional.pad(c, (0, pad))
            padded_chunks.append((c, off))
        return padded_chunks

    @staticmethod
    def _cue_text(words):
        out = ""
        for w in words:
            out = join_cues_text(out, w)
        return out

    @staticmethod
    def _group_word_cues(wlines, max_words=8, max_gap=0.6, max_chars=84, min_dur=0.2):
        words = [(st, max(en, st + 0.05), tx) for st, en, tx in wlines if tx]
        cues = []
        cur = []
        cur_len = 0
        for st, en, tx in words:
            if cur and (len(cur) >= max_words or cur_len + 1 + len(tx) > max_chars
                    or st - cur[-1][1] > max_gap):
                cues.append((cur[0][0], cur[-1][1], Qwen3ASRTranscribe._cue_text([w[2] for w in cur])))
                cur = []
                cur_len = 0
            cur.append((st, en, tx))
            cur_len += len(tx) + (1 if cur_len else 0)
        if cur:
            cues.append((cur[0][0], cur[-1][1], Qwen3ASRTranscribe._cue_text([w[2] for w in cur])))
        out = []
        i = 0
        while i < len(cues):
            st, en, tx = cues[i]
            if en <= st:
                en = st + min_dur
            want = st + min_dur
            if i + 1 < len(cues):
                nxt = cues[i + 1][0]
                cap = nxt - 0.04
                if want > cap:
                    if cap <= st + 0.2:
                        nst, nen, ntx = cues[i + 1]
                        cues[i + 1] = (st, max(nen, en), join_cues_text(tx, ntx))
                        i += 1
                        continue
                    en = max(en, cap)
                else:
                    en = max(en, want)
            else:
                en = max(en, want)
            out.append((st, en, tx))
            i += 1
        return out

    def _transcribe_parts(self, wrapper, tokenizer, lang, system_prompt, parts, max_new_tokens, temperature, load_device, target_dtype, pbar, base):
        texts = []
        langs = []
        chunk_langs = []
        eff_system = system_prompt or ""
        for ci, (cwav, coff) in enumerate(parts):
            comfy.model_management.throw_exception_if_processing_interrupted()
            feats, mask = extract_input_features(cwav)
            n_tok = int(audio_token_length(mask.sum(-1)).item())
            prefill = "language {}<asr_text>".format(lang) if lang else ""
            prompt_text = build_prompt_text(eff_system, n_tok, prefill)
            ids = torch.tensor([tokenizer._encode_str(prompt_text)], dtype=torch.long)
            feats = feats.to(load_device, dtype=target_dtype)
            mask = mask.to(load_device)
            ids = ids.to(load_device)
            step0 = base + ci * max_new_tokens

            def _cb(s, b=step0):
                pbar.update_absolute(b + s)
                comfy.model_management.throw_exception_if_processing_interrupted()

            audio_embeds = wrapper.model.get_audio_features(feats, mask)
            out_ids = wrapper.generate(ids, audio_embeds.reshape(-1, audio_embeds.shape[-1]), max_new_tokens=max_new_tokens, temperature=temperature, callback=_cb)
            raw = tokenizer.decode_ids(out_ids, skip_special_tokens=True)
            if prefill and "<asr_text>" not in raw:
                raw = prefill + raw
            parsed = parse_output(raw, user_language=lang)
            texts.append(parsed["transcription"])
            chunk_langs.append(parsed["language"])
            if parsed["language"]:
                langs.append(parsed["language"])
            del feats, mask, ids, audio_embeds, out_ids
        merged = "".join(texts)
        merged_lang = ""
        prev = None
        for x in langs:
            x = (x or "").strip()
            if not x or x == prev:
                continue
            merged_lang = x if not merged_lang else merged_lang + "," + x
            prev = x
        chunk_cues = []
        for (cwav, coff), text in zip(parts, texts):
            body = (text or "").strip()
            if not body:
                continue
            start = float(coff)
            chunk_dur = float(cwav.shape[0]) / 16000.0
            sents = self._split_sentences(body)
            total_chars = max(1, sum(len(s) for s in sents))
            cur_st = start
            for s in sents:
                frac = len(s) / total_chars
                sent_dur = max(0.4, chunk_dur * frac)
                cur_en = min(start + chunk_dur, cur_st + sent_dur)
                if cur_en <= cur_st:
                    cur_en = cur_st + 0.4
                chunk_cues.append((cur_st, cur_en, s))
                cur_st = cur_en
        srt = format_srt_cues(chunk_cues)
        return (merged, merged_lang or (lang or ""), srt, texts, chunk_langs)

    def execute(self, model, audio, max_new_tokens=512, temperature=0.0, language="", system_prompt="", chunk_seconds=60.0, aligner=None, srt_max_words=8, srt_max_chars=84, srt_max_gap=0.6):
        wrapper = model.model if isinstance(model, comfy.model_patcher.ModelPatcher) else model
        load_device = comfy.model_management.get_torch_device()
        to_load = [model]
        awrapper = None
        atokenizer = None
        if aligner is not None:
            awrapper = aligner.model if isinstance(aligner, comfy.model_patcher.ModelPatcher) else aligner
            atokenizer = getattr(awrapper, "aligner_tokenizer", None)
            if atokenizer is None:
                raise RuntimeError("Qwen3 aligner model has no tokenizer.")
            to_load.append(aligner)
        if isinstance(model, comfy.model_patcher.ModelPatcher):
            comfy.model_management.load_models_gpu(to_load)
        tokenizer = getattr(wrapper, "asr_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Qwen3ASR model has no tokenizer.")
        lang = resolve_language(language or None)
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sr != 16000:
            waveform = comfy.audio.resample(waveform, sr, 16000)
        peak = float(waveform.abs().max().item())
        if peak > 1.0:
            waveform = waveform / peak
        monos = [waveform[b].reshape(-1).to(torch.float32).cpu() for b in range(waveform.shape[0])]
        try:
            align_node = Qwen3ForcedAlign() if awrapper is not None else None
            atarget = (getattr(awrapper, "dtype", None) or torch.bfloat16) if awrapper is not None else None
            target_dtype = getattr(wrapper, "dtype", None) or torch.bfloat16
            parts_all = [self._split_chunks(m, 16000, chunk_seconds) if chunk_seconds and chunk_seconds > 0 else [(m, 0.0)] for m in monos]
            pbar = comfy.utils.ProgressBar(max_new_tokens * sum(len(p) for p in parts_all))
            out = []
            word_srts = []
            base = 0
            for parts in parts_all:
                merged, merged_lang, srt, texts, chunk_langs = self._transcribe_parts(wrapper, tokenizer, lang, system_prompt, parts, max_new_tokens, temperature, load_device, target_dtype, pbar, base)
                base += len(parts) * max_new_tokens
                if align_node is not None:
                    wlines = []
                    for (cwav, coff), text, clang in zip(parts, texts, chunk_langs):
                        if not (text or "").strip():
                            continue
                        items = align_node.align_mono(awrapper, atokenizer, cwav, text, clang or lang or merged_lang or "English", load_device, atarget)
                        for it in items:
                            wlines.append((coff + it["start"], coff + it["end"], it["text"]))
                    word_srts.append(format_srt_cues(wlines))
                    cues = self._group_word_cues(wlines, max_words=int(srt_max_words), max_gap=float(srt_max_gap), max_chars=int(srt_max_chars))
                    if cues:
                        srt = format_srt_cues(cues)
                else:
                    word_srts.append("")
                out.append((merged, merged_lang, srt))
        finally:
            comfy.model_management.soft_empty_cache()
        if len(out) == 1:
            return (out[0][0], out[0][1], out[0][2], word_srts[0])
        return ([o[0] for o in out], [o[1] for o in out], [o[2] for o in out], word_srts)


NODE_CLASS_MAPPINGS = {
    "Qwen3ASRLoader": Qwen3ASRLoader,
    "Qwen3ASRTranscribe": Qwen3ASRTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASRLoader": "Load Qwen3-ASR Model",
    "Qwen3ASRTranscribe": "Qwen3-ASR Transcribe",
}
