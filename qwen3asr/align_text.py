import unicodedata


TIMESTAMP_TOKEN = "<timestamp>"


def is_kept_char(ch):
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    if cat.startswith("L") or cat.startswith("N"):
        return True
    return False


def clean_token(token):
    return "".join(ch for ch in token if is_kept_char(ch))


def is_cjk_char(ch):
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def tokenize_chinese_mixed(text):
    tokens = []
    current_latin = []

    def flush_latin():
        if current_latin:
            token = "".join(current_latin)
            cleaned = clean_token(token)
            if cleaned:
                tokens.append(cleaned)
            del current_latin[:]

    for ch in text:
        if is_cjk_char(ch):
            flush_latin()
            tokens.append(ch)
        else:
            if is_kept_char(ch):
                current_latin.append(ch)
            else:
                flush_latin()

    flush_latin()

    return tokens


def split_segment_with_chinese(seg):
    tokens = []
    buf = []

    def flush_buf():
        if buf:
            tokens.append("".join(buf))
            del buf[:]

    for ch in seg:
        if is_cjk_char(ch):
            flush_buf()
            tokens.append(ch)
        else:
            buf.append(ch)

    flush_buf()
    return tokens


def tokenize_space_lang(text):
    tokens = []
    for seg in text.split():
        cleaned = clean_token(seg)
        if cleaned:
            tokens.extend(split_segment_with_chinese(cleaned))
    return tokens


def is_kana_char(ch):
    code = ord(ch)
    return (
        0x3040 <= code <= 0x309F  # Hiragana
        or 0x30A0 <= code <= 0x30FF  # Katakana
        or 0xFF66 <= code <= 0xFF9D  # Halfwidth Katakana
    )


def is_hangul_char(ch):
    code = ord(ch)
    return (
        0xAC00 <= code <= 0xD7A3  # Hangul Syllables
        or 0x1100 <= code <= 0x11FF  # Hangul Jamo
        or 0x3130 <= code <= 0x318F  # Compatibility Jamo
    )


def tokenize_char_units(text):
    # Character-level units for Japanese/Korean: each CJK/kana/hangul char is
    # one unit, latin/digit runs stay grouped. The aligner natively supports
    # character granularity; upstream word splits need nagisa/soynlp.
    tokens = []
    buf = []

    def flush_buf():
        if buf:
            token = "".join(buf)
            cleaned = clean_token(token)
            if cleaned:
                tokens.append(cleaned)
            del buf[:]

    for ch in text:
        if is_cjk_char(ch) or is_kana_char(ch) or is_hangul_char(ch):
            flush_buf()
            if is_kept_char(ch):
                tokens.append(ch)
        elif is_kept_char(ch):
            buf.append(ch)
        else:
            flush_buf()

    flush_buf()
    return tokens


# Upstream uses nagisa (Japanese) and soynlp (Korean) for word-level splits.
# Both are extra dependencies, so this port uses character-level units for
# those two languages and the space/CJK path (identical to upstream) elsewhere.
def tokenize_words(text, language):
    lang = (language or "").strip().lower()
    if lang in ("japanese", "ja", "jpn", "korean", "ko", "kor"):
        return tokenize_char_units(text)
    return tokenize_space_lang(text)


def encode_timestamp(text, language):
    word_list = tokenize_words(text, language)
    body = (TIMESTAMP_TOKEN + TIMESTAMP_TOKEN).join(word_list) + TIMESTAMP_TOKEN + TIMESTAMP_TOKEN
    return word_list, body


def fix_timestamp(data):
    data = list(data.tolist() if hasattr(data, "tolist") else data)
    n = len(data)

    dp = [1] * n
    parent = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    max_length = max(dp)
    max_idx = dp.index(max_length)

    lis_indices = []
    idx = max_idx
    while idx != -1:
        lis_indices.append(idx)
        idx = parent[idx]
    lis_indices.reverse()

    is_normal = [False] * n
    for idx in lis_indices:
        is_normal[idx] = True

    result = data.copy()
    i = 0

    while i < n:
        if not is_normal[i]:
            j = i
            while j < n and not is_normal[j]:
                j += 1

            anomaly_count = j - i

            if anomaly_count <= 2:
                left_val = None
                for k in range(i - 1, -1, -1):
                    if is_normal[k]:
                        left_val = result[k]
                        break

                right_val = None
                for k in range(j, n):
                    if is_normal[k]:
                        right_val = result[k]
                        break

                for k in range(i, j):
                    if left_val is None:
                        result[k] = right_val
                    elif right_val is None:
                        result[k] = left_val
                    else:
                        result[k] = left_val if (k - (i - 1)) <= ((j) - k) else right_val

            else:
                left_val = None
                for k in range(i - 1, -1, -1):
                    if is_normal[k]:
                        left_val = result[k]
                        break

                right_val = None
                for k in range(j, n):
                    if is_normal[k]:
                        right_val = result[k]
                        break

                if left_val is not None and right_val is not None:
                    step = (right_val - left_val) / (anomaly_count + 1)
                    for k in range(i, j):
                        result[k] = left_val + step * (k - i + 1)
                elif left_val is not None:
                    for k in range(i, j):
                        result[k] = left_val
                elif right_val is not None:
                    for k in range(i, j):
                        result[k] = right_val

            i = j
        else:
            i += 1

    return [int(res) for res in result]


def parse_timestamp(word_list, timestamp):
    timestamp_output = []
    timestamp_fixed = fix_timestamp(timestamp)
    for i, word in enumerate(word_list):
        start_time = timestamp_fixed[i * 2]
        end_time = timestamp_fixed[i * 2 + 1]
        timestamp_output.append({
            "text": word,
            "start_time": start_time,
            "end_time": end_time
        })
    return timestamp_output


def join_cues_text(a, b):
    a = (a or "").strip()
    b = (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    if (is_cjk_char(a[-1]) or is_kana_char(a[-1])) and (is_cjk_char(b[0]) or is_kana_char(b[0])):
        return a + b
    return a + " " + b


def format_timestamp(sec):
    sec = max(0.0, float(sec))
    total_ms = int(round(sec * 1000.0))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_srt_cues(cues):
    lines = []
    for idx, (st, en, tx) in enumerate(cues, 1):
        tx = (tx or "").strip()
        if not tx:
            continue
        lines.append(str(idx))
        lines.append(f"{format_timestamp(st)} --> {format_timestamp(en)}")
        lines.append(tx)
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if lines else "")
