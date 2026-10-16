import torch

import comfy.audio

SAMPLE_RATE = 16000
CHUNK_FRAMES = 100  # mel frames per audio tower chunk (2 * n_window)


def resample_16k(waveform, sample_rate):
    """[batch, channels, samples] -> list of mono 16 kHz float32 waveforms on CPU."""
    waveform = waveform.mean(dim=1).float().cpu()
    # Close to soxr_hq, which upstream resamples with; the default filter noticeably changes transcripts.
    waveform = comfy.audio.resample(waveform, sample_rate, SAMPLE_RATE, lowpass_filter_width=64, rolloff=0.955)
    return [w / max(w.abs().max().item(), 1.0) for w in waveform]


def log_mel(waveform, n_mels):
    """Whisper log-mel spectrogram of mono 16 kHz audio [..., samples] -> [..., n_mels, frames]."""
    mel = comfy.audio.MelSpectrogram(SAMPLE_RATE, 400, 160, n_mels, f_max=8000.0).to(waveform.device)(waveform)[..., :-1]
    log_spec = mel.clamp(min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
    return (log_spec + 4.0) / 4.0


def extract_input_features(waveform):
    """Mono 16 kHz waveform -> ([1, 128, frames padded to CHUNK_FRAMES], valid frame count)."""
    if waveform.shape[0] < 8000:
        waveform = torch.nn.functional.pad(waveform, (0, 8000 - waveform.shape[0]))
    mel = log_mel(waveform, 128)
    num_frames = mel.shape[-1]
    mel = torch.nn.functional.pad(mel, (0, -num_frames % CHUNK_FRAMES))
    return mel.unsqueeze(0), num_frames


def conv_output_length(num_frames):
    """Tokens left after the three stride-2 convolutions of one chunk."""
    for _ in range(3):
        num_frames = (num_frames - 1) // 2 + 1 if num_frames > 0 else 0
    return num_frames


def audio_token_length(num_frames):
    return conv_output_length(num_frames % CHUNK_FRAMES) + (num_frames // CHUNK_FRAMES) * conv_output_length(CHUNK_FRAMES)
