import numpy as np
import torch


def hertz_to_mel(freq, mel_scale="slaney"):
    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    if mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels = mels.copy() if np.shares_memory(mels, freq) else mels
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels


def mel_to_hertz(mels, mel_scale="slaney"):
    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    if mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq = freq.copy()
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    return freq


def _triangular_filter_bank(fft_freqs, filter_freqs):
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def mel_filter_bank(num_frequency_bins=201, num_mel_filters=128, min_frequency=0.0,
                    max_frequency=8000.0, sampling_rate=16000):
    mel_min = hertz_to_mel(min_frequency, mel_scale="slaney")
    mel_max = hertz_to_mel(max_frequency, mel_scale="slaney")
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale="slaney")
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
    mel_filters = _triangular_filter_bank(fft_freqs, filter_freqs)
    enorm = 2.0 / (filter_freqs[2:num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters *= np.expand_dims(enorm, 0)
    return mel_filters


_MEL_CACHE = {}


def get_mel_filters(device="cpu"):
    key = str(device)
    if key not in _MEL_CACHE:
        mel = mel_filter_bank(1 + 400 // 2, 128, 0.0, 8000.0, 16000)
        _MEL_CACHE[key] = torch.from_numpy(mel).to(device, torch.float32)
    else:
        cached = _MEL_CACHE[key]
        if str(cached.device) != str(device):
            mel = mel_filter_bank(1 + 400 // 2, 128, 0.0, 8000.0, 16000)
            _MEL_CACHE[key] = torch.from_numpy(mel).to(device, torch.float32)
    return _MEL_CACHE[key]


def log_mel_features(waveform, device="cpu", n_fft=400, hop_length=160):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.to(device, torch.float32)
    window = torch.hann_window(n_fft, device=device)
    stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_filters = get_mel_filters(device)
    mel_spec = mel_filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    if waveform.dim() == 2:
        max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_val - 8.0)
    else:
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def extract_input_features(waveform_16k, n_window=50, min_length=8000, sampling_rate=16000):
    if waveform_16k.dim() > 1:
        waveform_16k = waveform_16k.reshape(-1)
    waveform_16k = waveform_16k.to(torch.float32).cpu()
    if waveform_16k.shape[0] < min_length:
        pad = min_length - waveform_16k.shape[0]
        waveform_16k = torch.cat([waveform_16k, torch.zeros(pad)])
    raw_len = waveform_16k.shape[0]
    raw_mask = torch.ones(raw_len, dtype=torch.long)
    mel = log_mel_features(waveform_16k.unsqueeze(0), device="cpu").squeeze(0)
    mel_frames = mel.shape[-1]
    frame_mask = raw_mask[::hop_length_default()].clone()
    if raw_mask.shape[0] % hop_length_default() != 0:
        frame_mask = frame_mask[:-1]
    frame_mask = frame_mask[:mel_frames]
    if frame_mask.shape[0] < mel_frames:
        frame_mask = torch.cat([frame_mask, torch.zeros(mel_frames - frame_mask.shape[0], dtype=torch.long)])
    multiple = n_window * 2
    if multiple and multiple > 1:
        remainder = mel.shape[-1] % multiple
        pad = (multiple - remainder) if remainder else 0
        if pad:
            mel = torch.cat([mel, torch.zeros(mel.shape[0], pad)], dim=-1)
            frame_mask = torch.cat([frame_mask, torch.zeros(pad, dtype=torch.long)])
    return mel.unsqueeze(0), frame_mask.unsqueeze(0)


def hop_length_default():
    return 160


def audio_token_length(feature_lens, n_window=50):
    chunk_len = n_window * 2
    remainder = feature_lens % chunk_len
    feat_lengths = (remainder - 1) // 2 + 1
    per_chunk = (feat_lengths - 1) // 2 + 1
    return (per_chunk - 1) // 2 + 1 + (feature_lens // chunk_len) * 13
