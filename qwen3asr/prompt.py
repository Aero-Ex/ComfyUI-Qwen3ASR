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

LANGUAGES = {**LANGUAGE_CODE_TO_NAME, **{name.lower(): name for name in LANGUAGE_CODE_TO_NAME.values()}}


def resolve_language(language):
    """Language name or code -> canonical name, None for auto-detect."""
    return LANGUAGES.get(language.strip().lower())


def build_prompt_text(system_text="", num_audio_tokens=0, prefill=""):
    audio_seq = "<|audio_start|>" + "<|audio_pad|>" * num_audio_tokens + "<|audio_end|>"
    prompt = "<|im_start|>system\n" + system_text + "<|im_end|>\n"
    prompt += "<|im_start|>user\n" + audio_seq + "<|im_end|>\n"
    prompt += "<|im_start|>assistant\n" + prefill
    return prompt


def _fix_char_repeats(s, thresh=20):
    res = []
    i, n = 0, len(s)
    while i < n:
        count = 1
        while i + count < n and s[i + count] == s[i]:
            count += 1
        if count > thresh:
            res.append(s[i])
            i += count
        else:
            res.append(s[i:i + count])
            i += count
    return "".join(res)


def _fix_pattern_repeats(s, thresh=20, max_len=20):
    n = len(s)
    min_repeat_chars = thresh * 2
    if n < min_repeat_chars:
        return s
    i = 0
    result = []
    found = False
    while i <= n - min_repeat_chars:
        hit = False
        for k in range(1, max_len + 1):
            if i + k * thresh > n:
                break
            pattern = s[i:i + k]
            valid = True
            for rep in range(1, thresh):
                if s[i + rep * k:i + rep * k + k] != pattern:
                    valid = False
                    break
            if valid:
                end_index = i + thresh * k
                while end_index + k <= n and s[end_index:end_index + k] == pattern:
                    end_index += k
                result.append(pattern)
                result.append(_fix_pattern_repeats(s[end_index:], thresh, max_len))
                i = n
                hit = True
                found = True
                break
        if hit:
            break
        result.append(s[i])
        i += 1
    if not found:
        result.append(s[i:])
    return "".join(result)


def fix_repetitions(text, threshold=20):
    return _fix_pattern_repeats(_fix_char_repeats(text, threshold), threshold)


def parse_output(text, language=None):
    """Split raw model output "language X<asr_text>text" into (language, transcription), as upstream does."""
    text = fix_repetitions(text.strip())
    if language:
        return language, text
    if "<asr_text>" not in text:
        return None, text
    prefix, text = text.split("<asr_text>", 1)
    for line in prefix.splitlines():
        line = line.strip()
        if line.lower().startswith("language "):
            line = line[len("language "):].strip()
            if line and line.lower() != "none":
                return line[:1].upper() + line[1:].lower(), text.strip()
            break
    return None, text.strip()
