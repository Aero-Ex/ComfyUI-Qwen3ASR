LANGUAGE_CODE_TO_NAME = {
    "ar": "Arabic", "yue": "Cantonese", "zh": "Chinese", "cs": "Czech",
    "da": "Danish", "nl": "Dutch", "en": "English", "fil": "Filipino",
    "fi": "Finnish", "fr": "French", "de": "German", "el": "Greek",
    "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "mk": "Macedonian", "ms": "Malay",
    "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "es": "Spanish", "sv": "Swedish", "th": "Thai",
    "tr": "Turkish", "vi": "Vietnamese",
}

NAME_TO_CODE = {v.lower(): k for k, v in LANGUAGE_CODE_TO_NAME.items()}
CODE_SET = set(LANGUAGE_CODE_TO_NAME)


def resolve_language(language):
    if language is None:
        return None
    s = str(language).strip()
    if not s or s.lower() in ("auto", "none"):
        return None
    if s in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[s]
    low = s.lower()
    if low in NAME_TO_CODE:
        return LANGUAGE_CODE_TO_NAME[NAME_TO_CODE[low]]
    for code, name in LANGUAGE_CODE_TO_NAME.items():
        if name.lower() == low:
            return name
    return None


def build_prompt_text(system_text="", num_audio_tokens=0, prefill=""):
    audio_seq = "<|audio_start|>" + "<|audio_pad|>" * num_audio_tokens + "<|audio_end|>"
    prompt = "<|im_start|>system\n" + system_text + "<|im_end|>\n"
    prompt += "<|im_start|>user\n" + audio_seq + "<|im_end|>\n"
    prompt += "<|im_start|>assistant\n" + prefill
    return prompt
