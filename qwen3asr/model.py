import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_prefetch
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.text_encoders.llama import BaseGenerate, BaseQwen3, Llama2_, Qwen3_06BConfig

from .features import CHUNK_FRAMES, conv_output_length


def sinusoids(length, channels, device):
    inv_timescales = torch.exp(-math.log(10000) / (channels // 2 - 1) * torch.arange(channels // 2, device=device, dtype=torch.float32))
    scaled_time = torch.arange(length, device=device, dtype=torch.float32)[:, None] * inv_timescales[None, :]
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class AudioAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        out = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), self.num_heads, skip_reshape=True, preferred_attention=self.comfy_attention)
        return self.out_proj(out)


class AudioEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = AudioAttention(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.self_attn_layer_norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.fc1 = operations.Linear(d_model, ffn_dim, device=device, dtype=dtype)
        self.fc2 = operations.Linear(ffn_dim, d_model, device=device, dtype=dtype)
        self.final_layer_norm = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class AudioTower(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        d_model = config["d_model"]
        hidden = config["downsample_hidden_size"]
        self.window_chunks = config["n_window_infer"] // CHUNK_FRAMES
        self.conv2d1 = operations.Conv2d(1, hidden, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv2d2 = operations.Conv2d(hidden, hidden, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv2d3 = operations.Conv2d(hidden, hidden, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv_out = operations.Linear(hidden * conv_output_length(config["num_mel_bins"]), d_model, bias=False, device=device, dtype=dtype)
        self.layers = nn.ModuleList([AudioEncoderLayer(d_model, config["encoder_attention_heads"], config["encoder_ffn_dim"], device=device, dtype=dtype, operations=operations)
                                     for _ in range(config["encoder_layers"])])
        self.ln_post = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, features, num_frames):
        """features: [1, mel, frames padded to CHUNK_FRAMES] -> [audio tokens, d_model]."""
        chunks = features[0].unflatten(-1, (-1, CHUNK_FRAMES)).transpose(0, 1).unsqueeze(1)
        x = []
        for i in range(0, chunks.shape[0], 32):
            h = F.gelu(self.conv2d1(chunks[i:i + 32]))
            h = F.gelu(self.conv2d2(h))
            x.append(F.gelu(self.conv2d3(h)))
        x = torch.cat(x).permute(0, 3, 1, 2).flatten(2)
        x = self.conv_out(x)
        x = x + sinusoids(x.shape[1], x.shape[2], x.device).to(x.dtype)
        window = x.shape[1] * self.window_chunks  # attention runs within windows of window_chunks chunks
        last_len = conv_output_length(num_frames - CHUNK_FRAMES * (x.shape[0] - 1))
        x = torch.cat([x[:-1].flatten(0, 1), x[-1, :last_len]])
        full = x.shape[0] // window * window
        groups = [x[:full].unflatten(0, (-1, window))] if full else []
        if full < x.shape[0]:
            groups.append(x[full:].unsqueeze(0))

        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {"prefetch_dynamic_vbars": True})
        for layer in self.layers:
            def core():
                nonlocal groups
                groups = [layer(g) for g in groups]
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, layer, x.dtype, core=core, malloc_scope="block")
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None, malloc_scope="block")
        return self.ln_post(torch.cat([g.flatten(0, 1) for g in groups]))


class MultiModalProjector(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.linear_1 = operations.Linear(config["d_model"], config["d_model"], device=device, dtype=dtype)
        self.linear_2 = operations.Linear(config["d_model"], config["output_dim"], device=device, dtype=dtype)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(x)))


class Qwen3ASR(nn.Module):
    """Qwen3-ASR: windowed audio encoder whose projected features replace <|audio_pad|> tokens in a Qwen3 decoder."""

    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        text = config["text_config"]
        llm_config = Qwen3_06BConfig(vocab_size=text["vocab_size"], hidden_size=text["hidden_size"], intermediate_size=text["intermediate_size"],
                                     num_hidden_layers=text["num_hidden_layers"], num_attention_heads=text["num_attention_heads"],
                                     num_key_value_heads=text["num_key_value_heads"], rms_norm_eps=text["rms_norm_eps"],
                                     rope_theta=text["rope_parameters"]["rope_theta"])
        llm_config.head_dim = text["head_dim"]
        self.audio_token_id = config["audio_token_id"]
        self.audio_tower = AudioTower(config["audio_config"], device=device, dtype=dtype, operations=operations)
        self.multi_modal_projector = MultiModalProjector(config["audio_config"], device=device, dtype=dtype, operations=operations)
        self.language_model = Llama2_(llm_config, device=device, dtype=dtype, ops=operations)

    def embed(self, input_ids, features, num_frames):
        audio = self.multi_modal_projector(self.audio_tower(features, num_frames))
        embeds = self.language_model.embed_tokens(input_ids, out_dtype=audio.dtype)
        embeds[input_ids == self.audio_token_id] = audio
        return embeds

    def generate(self, embeds, max_new_tokens, temperature):
        out = Generator(self.language_model).generate(embeds, do_sample=temperature > 0, max_length=max_new_tokens, temperature=temperature, top_k=0, top_p=1.0,
                                                      seed=0, execution_dtype=self.dtype)
        if out and out[-1] in self.language_model.config.stop_tokens:
            out.pop()
        return out


class Qwen3ForcedAligner(Qwen3ASR):
    """Qwen3-ForcedAligner: the Qwen3-ASR stack with a per-token timestamp classifier."""

    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__(config, device=device, dtype=dtype, operations=operations)
        self.timestamp_token_id = config["timestamp_token_id"]
        self.timestamp_segment_time = config["timestamp_segment_time"]
        self.score = operations.Linear(config["text_config"]["hidden_size"], len(config["id2label"]), bias=False, device=device, dtype=dtype)

    def forward(self, input_ids, features, num_frames):
        x, _ = self.language_model(None, embeds=self.embed(input_ids, features, num_frames))
        return self.score(x)


class Generator(BaseQwen3, BaseGenerate):
    """Comfy's KV-cached decoding loop over a Llama2_ with tied embeddings."""

    def __init__(self, model):
        self.model = model
