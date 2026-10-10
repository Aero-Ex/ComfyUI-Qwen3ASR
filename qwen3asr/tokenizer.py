import json
import re
import unicodedata

import torch


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def _is_letter(ch):
    return unicodedata.category(ch).startswith("L")


def _is_number(ch):
    return unicodedata.category(ch).startswith("N")


_CONTRACTIONS = ("s", "t", "re", "ve", "m", "ll", "d")


def _match_at(text, i):
    n = len(text)
    ch = text[i]
    if ch == "'":
        for contraction in _CONTRACTIONS:
            end = i + 1 + len(contraction)
            if text[i + 1:end].lower() == contraction:
                return text[i:end], end
    if ch not in "\r\n" and not _is_letter(ch) and not _is_number(ch):
        if i + 1 < n and _is_letter(text[i + 1]):
            j = i + 2
            while j < n and _is_letter(text[j]):
                j += 1
            return text[i:j], j
    if _is_letter(ch):
        j = i + 1
        while j < n and _is_letter(text[j]):
            j += 1
        return text[i:j], j
    if _is_number(ch):
        return ch, i + 1
    j = i + (1 if ch == " " else 0)
    k = j
    while k < n and text[k] not in "\r\n" and not _is_letter(text[k]) and not _is_number(text[k]) and not text[k].isspace():
        k += 1
    if k > j:
        while k < n and text[k] in "\r\n":
            k += 1
        return text[i:k], k
    if ch.isspace():
        j = i
        while j < n and text[j].isspace():
            j += 1
        last_nl = -1
        for p in range(i, j):
            if text[p] in "\r\n":
                last_nl = p
        if last_nl >= 0:
            return text[i:last_nl + 1], last_nl + 1
        if j == n:
            return text[i:j], j
        if j - 1 > i:
            return text[i:j - 1], j - 1
        return text[i:j], j
    return ch, i + 1


def _split_gpt2(text):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        token, i = _match_at(text, i)
        tokens.append(token)
    return tokens


class QwenBpeTokenizer:
    def __init__(self, source, device=None):
        if isinstance(source, dict):
            data = source
        else:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        model = data["model"]
        self.vocab = model["vocab"]
        self.merges = model["merges"]
        ranks = {}
        for i, m in enumerate(self.merges):
            pair = tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1))
            ranks[pair] = i
        self.bpe_ranks = ranks
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.added_tokens = {}
        self.id_to_added = {}
        self.skip_ids = set()
        for tok in data.get("added_tokens", []):
            self.added_tokens[tok["content"]] = tok["id"]
            self.id_to_added[tok["id"]] = tok["content"]
            if tok.get("special"):
                self.skip_ids.add(tok["id"])
        self.id_to_vocab = {v: k for k, v in self.vocab.items()}
        self._added_re = None
        if self.added_tokens:
            self._added_re = re.compile("|".join(re.escape(t) for t in sorted(self.added_tokens, key=len, reverse=True)))
        self.cache = {}
        self.device = device

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        result = " ".join(word)
        self.cache[token] = result
        return result

    def _encode_chunk(self, text):
        ids = []
        for token in _split_gpt2(text):
            token_bytes = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            for bpe_token in self._bpe(token_bytes).split(" "):
                ids.append(self.vocab[bpe_token])
        return ids

    def _encode_str(self, text):
        text = unicodedata.normalize("NFC", text)
        ids = []
        if self._added_re is not None:
            pos = 0
            for m in self._added_re.finditer(text):
                if m.start() > pos:
                    ids.extend(self._encode_chunk(text[pos:m.start()]))
                ids.append(self.added_tokens[m.group(0)])
                pos = m.end()
            if pos < len(text):
                ids.extend(self._encode_chunk(text[pos:]))
        else:
            ids = self._encode_chunk(text)
        return ids

    def encode(self, text):
        return torch.tensor([self._encode_str(text)], dtype=torch.long, device=self.device)

    def decode_ids(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        text_parts = []
        byte_run = ""
        def flush_bytes():
            nonlocal byte_run
            if not byte_run:
                return
            try:
                text_parts.append(bytearray([self.byte_decoder[c] for c in byte_run]).decode("utf-8", errors="replace"))
            except Exception:
                text_parts.append("")
            byte_run = ""
        for i in ids:
            if i in self.id_to_added:
                flush_bytes()
                if skip_special_tokens and i in self.skip_ids:
                    continue
                text_parts.append(self.id_to_added[i])
                continue
            piece = self.id_to_vocab.get(i, "")
            byte_run += piece
        flush_bytes()
        return "".join(text_parts)


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


def parse_output(text):
    if text is None or not str(text).strip():
        return {"language": None, "transcription": ""}
    text = str(text).strip()
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[-1]
    text = fix_repetitions(text)
    marker = "<asr_text>"
    if marker not in text:
        return {"language": None, "transcription": text.strip()}
    prefix, transcription = text.split(marker, 1)
    prefix = prefix.strip()
    if prefix.lower() == "language none":
        return {"language": None, "transcription": transcription.strip()}
    language = None
    for line in prefix.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("language "):
            val = line[len("language "):].strip()
            if val:
                language = val
        else:
            language = line
        break
    return {"language": language or None, "transcription": transcription.strip()}
