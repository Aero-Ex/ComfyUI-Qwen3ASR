import json
import re

import folder_paths
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import torch
from tokenizers import Tokenizer

from .qwen3asr.align_text import encode_timestamp, format_srt_cues, join_cues_text, parse_timestamp
from .qwen3asr.features import SAMPLE_RATE, audio_token_length, extract_input_features, resample_16k
from .qwen3asr.model import Qwen3ASR, Qwen3ForcedAligner
from .qwen3asr.prompt import build_prompt_text, parse_output, resolve_language

MAX_ALIGN_CHUNK_SECONDS = 180  # upstream chunk length when timestamps are requested


def load_qwen3asr(path, prefix, model_class, required_keys):
    sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    metadata = metadata or {}
    if metadata.get(prefix + ".format_version") != "1" or any(k not in sd for k in required_keys):
        raise RuntimeError("Not a single-file {} checkpoint. Download a pre-converted one: "
                           "https://huggingface.co/Aero-Ex/ComfyUI-Qwen3-ASR".format(model_class.__name__))
    config = json.loads(metadata[prefix + ".config"])

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    # Same dtype policy as core's model loading: bf16 where supported, fp16 on older GPUs, honoring --force-fp16 and similar flags.
    supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    # INT8 checkpoints carry per-layer .comfy_quant keys, consumed by the mixed-precision ops.
    quant = comfy.utils.detect_layer_quantization(sd, "")
    weight_dtype = None if quant is not None else comfy.utils.weight_dtype(sd)
    dtype = comfy.model_management.unet_dtype(model_params=comfy.utils.calculate_parameters(sd), supported_dtypes=supported_dtypes, weight_dtype=weight_dtype)
    manual_cast_dtype = comfy.model_management.unet_manual_cast(None if quant is not None else dtype, load_device, supported_dtypes)
    if quant is not None:
        operations = comfy.ops.mixed_precision_ops(quant, manual_cast_dtype or dtype)
    else:
        operations = comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=load_device)
        if weight_dtype != dtype:
            # Lazily loaded layers keep the checkpoint tensors, so convert them to the model dtype once here.
            sd = {k: v.to(dtype) if v.dtype == weight_dtype else v for k, v in sd.items()}

    model = model_class(config, device=offload_device, dtype=dtype, operations=operations)
    model.load_state_dict(sd)
    model.dtype = manual_cast_dtype or dtype
    # Plain ModelPatcher: under dynamic VRAM the per-chunk decode loop ran ~1.7x slower (its CUDA graphs can't span several generate calls).
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    model.tokenizer = Tokenizer.from_str(metadata[prefix + ".tokenizer"])
    return patcher


def max_align_seconds(aligner):
    return aligner.score.out_features * aligner.timestamp_segment_time / 1000.0


def align_mono(patcher, wav, text, language):
    """Word timestamps (seconds) for a transcript of one mono 16 kHz chunk."""
    aligner = patcher.model
    if wav.shape[0] / SAMPLE_RATE > max_align_seconds(aligner):
        raise RuntimeError("Audio is {:.1f}s, the aligner supports {:.1f}s per pass. Split the audio (and matching transcript) first.".format(
            wav.shape[0] / SAMPLE_RATE, max_align_seconds(aligner)))
    words, body = encode_timestamp(text, language)
    if not words:
        return []
    features, num_frames = extract_input_features(wav)
    prompt = "<|audio_start|>" + "<|audio_pad|>" * audio_token_length(num_frames) + "<|audio_end|>" + body
    ids = torch.tensor([aligner.tokenizer.encode(prompt).ids], device=patcher.load_device)
    logits = aligner(ids, features.to(patcher.load_device, aligner.dtype), num_frames)
    stamps = logits[0, ids[0] == aligner.timestamp_token_id].argmax(dim=-1) * aligner.timestamp_segment_time
    items = parse_timestamp(words, stamps.tolist())
    return [{"text": it["text"], "start": round(it["start_time"] / 1000.0, 3), "end": round(it["end_time"] / 1000.0, 3)} for it in items]


def split_chunks(wav, max_sec, search_expand_sec=5.0, min_window_ms=100.0):
    """Upstream chunking: cut near every max_sec at the quietest point within +-search_expand_sec."""
    total_len = wav.shape[0]
    chunks = []
    start = 0
    if max_sec > 0:
        max_len = int(max_sec * SAMPLE_RATE)
        expand = min(int(search_expand_sec * SAMPLE_RATE), max_len // 2)  # keep chunks at least half of max_sec
        win = max(4, int(min_window_ms / 1000.0 * SAMPLE_RATE))
        while total_len - start > max_len:
            cut = start + max_len
            left = max(start, cut - expand)
            right = min(total_len, cut + expand)
            boundary = cut
            if right - left > win:
                seg_abs = wav[left:right].abs()
                cs = torch.nn.functional.pad(torch.cumsum(seg_abs, dim=0), (1, 0))
                min_pos = int(torch.argmin(cs[win:] - cs[:-win]))
                boundary = left + min_pos + int(torch.argmin(seg_abs[min_pos:min_pos + win]))
            boundary = min(max(boundary, start + 1), total_len)
            chunks.append((wav[start:boundary], start / SAMPLE_RATE))
            start = boundary
    chunks.append((wav[start:], start / SAMPLE_RATE))
    min_len = SAMPLE_RATE // 2
    return [(torch.nn.functional.pad(c, (0, max(0, min_len - c.shape[0]))), offset) for c, offset in chunks]


def split_sentences(text):
    parts = re.split(r'([.!?。\n\r]+|\s*\n+\s*)', text)
    sents = []
    buf = ""
    for p in parts:
        buf += p
        if re.search(r'[.!?。\n\r]$', buf):
            if buf.strip():
                sents.append(buf.strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    return sents


def sentence_cues(parts, texts):
    """Approximate cues without an aligner: spread each chunk's sentences over its duration by length."""
    cues = []
    for (cwav, start), text in zip(parts, texts):
        sents = split_sentences(text)
        chunk_dur = cwav.shape[0] / SAMPLE_RATE
        total_chars = max(1, sum(len(s) for s in sents))
        cur_st = start
        for s in sents:
            cur_en = min(start + chunk_dur, cur_st + max(0.4, chunk_dur * len(s) / total_chars))
            if cur_en <= cur_st:
                cur_en = cur_st + 0.4
            cues.append((cur_st, cur_en, s))
            cur_st = cur_en
    return cues


def join_texts(texts):
    out = ""
    for text in texts:
        out = join_cues_text(out, text)
    return out


def group_word_cues(words, max_words=8, max_gap=0.6, max_chars=84, min_dur=0.2):
    words = [(st, max(en, st + 0.05), tx) for st, en, tx in words if tx]
    cues = []
    cur = []
    cur_len = 0
    for st, en, tx in words:
        if cur and (len(cur) >= max_words or cur_len + 1 + len(tx) > max_chars or st - cur[-1][1] > max_gap):
            cues.append(cur)
            cur = []
            cur_len = 0
        cur.append((st, en, tx))
        cur_len += len(tx) + (1 if cur_len else 0)
    if cur:
        cues.append(cur)
    cues = [(c[0][0], c[-1][1], join_texts(w[2] for w in c)) for c in cues]
    out = []
    i = 0
    while i < len(cues):
        st, en, tx = cues[i]
        if en <= st:
            en = st + min_dur
        want = st + min_dur
        if i + 1 < len(cues):
            cap = cues[i + 1][0] - 0.04
            if want > cap:
                if cap <= st + 0.2:
                    _, nen, ntx = cues[i + 1]
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
        path = folder_paths.get_full_path_or_raise("qwen3asr", qwen3asr_name)
        return (load_qwen3asr(path, "qwen3asr", Qwen3ASR, ("language_model.embed_tokens.weight", "audio_tower.conv2d1.weight")),)


class Qwen3ASRTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("QWEN3ASR", {"tooltip": "Loaded Qwen3-ASR model."}),
            "audio": ("AUDIO", {"tooltip": "Audio to transcribe (resampled to 16kHz mono)."}),
            "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1, "tooltip": "Maximum tokens generated per chunk."}),
            "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "0 is greedy decoding, as upstream."}),
        }, "optional": {
            "language": ("STRING", {"default": "", "tooltip": "Force language (name or code, e.g. English/en). Empty for auto."}),
            "system_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional context/hotwords bias."}),
            "chunk_seconds": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 1200.0, "step": 5.0, "tooltip": "Split long audio into chunks of this length at a low-energy boundary. 0 disables chunking. Capped at 180 when an aligner is connected."}),
            "aligner": ("QWEN3ALIGNER", {"tooltip": "Optional forced aligner for word-level timestamps."}),
            "srt_max_words": ("INT", {"default": 8, "min": 1, "max": 30, "step": 1, "tooltip": "Max words per srt cue (aligner connected only)."}),
            "srt_max_chars": ("INT", {"default": 84, "min": 10, "max": 200, "step": 1, "tooltip": "Max characters per srt cue (aligner connected only)."}),
            "srt_max_gap": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Start a new srt cue when the gap between words exceeds this (seconds, aligner connected only)."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcription", "language", "srt", "word_srt")
    OUTPUT_IS_LIST = (True, True, True, True)
    OUTPUT_TOOLTIPS = ("One entry per audio in the batch.",) * 4
    FUNCTION = "execute"
    CATEGORY = "Qwen3ASR"

    def execute(self, model, audio, max_new_tokens=512, temperature=0.0, language="", system_prompt="", chunk_seconds=60.0, aligner=None, srt_max_words=8, srt_max_chars=84, srt_max_gap=0.6):
        comfy.model_management.load_models_gpu([model] if aligner is None else [model, aligner])
        asr = model.model
        lang = resolve_language(language)
        prefill = "language {}<asr_text>".format(lang) if lang else ""
        if aligner is not None:
            chunk_seconds = min(chunk_seconds or MAX_ALIGN_CHUNK_SECONDS, MAX_ALIGN_CHUNK_SECONDS, max_align_seconds(aligner.model))

        out = ([], [], [], [])
        for wav in resample_16k(audio["waveform"], audio["sample_rate"]):
            parts = split_chunks(wav, chunk_seconds)
            texts, langs = [], []
            for cwav, _ in parts:
                comfy.model_management.throw_exception_if_processing_interrupted()
                features, num_frames = extract_input_features(cwav)
                ids = asr.tokenizer.encode(build_prompt_text(system_prompt, audio_token_length(num_frames), prefill)).ids
                ids = torch.tensor([ids], device=model.load_device)
                embeds = asr.embed(ids, features.to(model.load_device, asr.dtype), num_frames)
                out_ids = asr.generate(embeds, max_new_tokens, temperature)
                chunk_lang, text = parse_output(asr.tokenizer.decode(out_ids, skip_special_tokens=True), lang)
                texts.append(text)
                langs.append(chunk_lang)

            merged_lang = []
            for x in langs:
                if x and (not merged_lang or merged_lang[-1] != x):
                    merged_lang.append(x)
            word_cues = []
            if aligner is not None:
                for (cwav, offset), text, chunk_lang in zip(parts, texts, langs):
                    if text:
                        for item in align_mono(aligner, cwav, text, chunk_lang or "English"):
                            word_cues.append((offset + item["start"], offset + item["end"], item["text"]))
                cues = group_word_cues(word_cues, max_words=srt_max_words, max_gap=srt_max_gap, max_chars=srt_max_chars)
            else:
                cues = sentence_cues(parts, texts)
            for i, value in enumerate((join_texts(texts), ",".join(merged_lang), format_srt_cues(cues), format_srt_cues(word_cues))):
                out[i].append(value)
        return out


class Qwen3AlignerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "qwen3aligner_name": (folder_paths.get_filename_list("qwen3asr"), {
                "tooltip": "Single-file Qwen3-ForcedAligner checkpoint (see Model Links in README)."}),
        }}

    RETURN_TYPES = ("QWEN3ALIGNER",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "Qwen3ASR"

    def execute(self, qwen3aligner_name):
        path = folder_paths.get_full_path_or_raise("qwen3asr", qwen3aligner_name)
        return (load_qwen3asr(path, "qwen3aligner", Qwen3ForcedAligner, ("language_model.embed_tokens.weight", "audio_tower.conv2d1.weight", "score.weight")),)


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
    OUTPUT_IS_LIST = (True, True)
    OUTPUT_TOOLTIPS = ("One entry per audio in the batch.",) * 2
    FUNCTION = "execute"
    CATEGORY = "Qwen3ASR"

    def execute(self, model, audio, transcription, language="English"):
        comfy.model_management.load_models_gpu([model])
        language = resolve_language(language) or language
        words_json, word_srt = [], []
        for wav in resample_16k(audio["waveform"], audio["sample_rate"]):
            comfy.model_management.throw_exception_if_processing_interrupted()
            items = align_mono(model, wav, transcription, language)
            words_json.append(json.dumps(items, ensure_ascii=False))
            word_srt.append(format_srt_cues([(it["start"], it["end"], it["text"]) for it in items]))
        return (words_json, word_srt)


NODE_CLASS_MAPPINGS = {
    "Qwen3ASRLoader": Qwen3ASRLoader,
    "Qwen3ASRTranscribe": Qwen3ASRTranscribe,
    "Qwen3AlignerLoader": Qwen3AlignerLoader,
    "Qwen3ForcedAlign": Qwen3ForcedAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASRLoader": "Load Qwen3-ASR Model",
    "Qwen3ASRTranscribe": "Qwen3-ASR Transcribe",
    "Qwen3AlignerLoader": "Load Qwen3-ForcedAligner Model",
    "Qwen3ForcedAlign": "Qwen3 Forced Align",
}
