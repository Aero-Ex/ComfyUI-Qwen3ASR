import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return comfy.ops.cast_to_input(self.weight, x).to(dtype) * x.to(dtype)


def precompute_freqs_cis(position_ids, inv_freq, device=None):
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    if device is not None:
        inv_freq_expanded = inv_freq_expanded.to(device)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos.unsqueeze(1), sin.unsqueeze(1)


def apply_rope(xq, xk, freqs_cis):
    cos, sin = freqs_cis
    cos = cos.to(dtype=xq.dtype, device=xq.device)
    sin = sin.to(dtype=xq.dtype, device=xq.device)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_ndim, k_ndim = xq.ndim, xk.ndim
    if q_ndim == 3:
        xq = xq.unsqueeze(0)
    if k_ndim == 3:
        xk = xk.unsqueeze(0)
    xq = (xq * cos) + (rotate_half(xq) * sin)
    xk = (xk * cos) + (rotate_half(xk) * sin)
    if q_ndim == 3:
        xq = xq.squeeze(0)
    if k_ndim == 3:
        xk = xk.squeeze(0)
    return xq, xk


class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps, device=None, dtype=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.inner_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.q_proj = operations.Linear(hidden_size, self.inner_size, bias=False, device=device, dtype=dtype)
        self.k_proj = operations.Linear(hidden_size, self.kv_size, bias=False, device=device, dtype=dtype)
        self.v_proj = operations.Linear(hidden_size, self.kv_size, bias=False, device=device, dtype=dtype)
        self.o_proj = operations.Linear(self.inner_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None, past_k=None, past_v=None):
        bsz, q_len, _ = hidden_states.shape
        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)
        xq = xq.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        if freqs_cis is not None:
            xq, xk = apply_rope(xq, xk, freqs_cis)
        if past_k is not None and past_v is not None:
            xk = torch.cat([past_k, xk], dim=2)
            xv = torch.cat([past_v, xv], dim=2)
        new_k, new_v = xk, xv
        if self.num_heads != self.num_kv_heads:
            xk, xv = comfy.ops.repeat_kv_for_gqa(xk, xv, self.num_heads, 1)
        q, k, v = AttentionTensorContainer(xq), AttentionTensorContainer(xk), AttentionTensorContainer(xv)
        output = optimized_attention(q, k, v, self.num_heads, mask=attention_mask, skip_reshape=True, preferred_attention=self.comfy_attention)
        return self.o_proj(output), new_k, new_v


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, device=None, dtype=None, operations=None):
        super().__init__()
        self.gate_proj = operations.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = Qwen3Attention(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps, device=device, dtype=dtype, operations=operations)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, device=device, dtype=dtype, operations=operations)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, freqs_cis=None, past_k=None, past_v=None):
        residual = x
        x = self.input_layernorm(x)
        x, new_k, new_v = self.self_attn(x, attention_mask=attention_mask, freqs_cis=freqs_cis, past_k=past_k, past_v=past_v)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x, new_k, new_v


class Qwen3TextModel(nn.Module):
    def __init__(self, text_config, device=None, dtype=None, operations=None):
        super().__init__()
        self.hidden_size = text_config.get("hidden_size", 1024)
        num_layers = text_config.get("num_hidden_layers", 28)
        num_heads = text_config.get("num_attention_heads", 16)
        num_kv_heads = text_config.get("num_key_value_heads", 8)
        head_dim = text_config.get("head_dim", 128)
        intermediate_size = text_config.get("intermediate_size", 3072)
        rms_norm_eps = text_config.get("rms_norm_eps", 1e-6)
        vocab_size = text_config.get("vocab_size", 151936)
        self.embed_tokens = operations.Embedding(vocab_size, self.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            Qwen3Layer(self.hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(self.hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        rope_theta = text_config.get("rope_parameters", {}).get("rope_theta", 1000000)
        rope_inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("rope_inv_freq", rope_inv_freq.to(dtype), persistent=False)

    def forward_embeds(self, inputs_embeds, attention_mask=None, past_kvs=None, start_pos=0):
        bsz, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        position_ids = torch.arange(start_pos, start_pos + seq_len, device=device).unsqueeze(0)
        freqs_cis = precompute_freqs_cis(position_ids, self.rope_inv_freq, inputs_embeds.device)
        x = inputs_embeds
        new_kvs = []
        for i, layer in enumerate(self.layers):
            pk, pv = (None, None)
            if past_kvs is not None:
                pk, pv = past_kvs[i]
            x, nk, nv = layer(x, attention_mask=attention_mask, freqs_cis=freqs_cis, past_k=pk, past_v=pv)
            new_kvs.append((nk, nv))
        return self.norm(x), new_kvs


class AudioAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = operations.Linear(d_model, d_model, bias=True, device=device, dtype=dtype)
        self.k_proj = operations.Linear(d_model, d_model, bias=True, device=device, dtype=dtype)
        self.v_proj = operations.Linear(d_model, d_model, bias=True, device=device, dtype=dtype)
        self.out_proj = operations.Linear(d_model, d_model, bias=True, device=device, dtype=dtype)

    def forward(self, hidden_states):
        seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(seq_len, self.num_heads, -1).transpose(0, 1).unsqueeze(0)
        k = self.k_proj(hidden_states).reshape(seq_len, self.num_heads, -1).transpose(0, 1).unsqueeze(0)
        v = self.v_proj(hidden_states).reshape(seq_len, self.num_heads, -1).transpose(0, 1).unsqueeze(0)
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        out = optimized_attention(q, k, v, self.num_heads, mask=None, skip_reshape=True, preferred_attention=self.comfy_attention)
        return self.out_proj(out.reshape(seq_len, -1))


class AudioEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = AudioAttention(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.self_attn_layer_norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.fc1 = operations.Linear(d_model, ffn_dim, device=device, dtype=dtype)
        self.fc2 = operations.Linear(ffn_dim, d_model, device=device, dtype=dtype)
        self.final_layer_norm = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.self_attn(self.self_attn_layer_norm(hidden_states))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


def _feat_extract_output_lengths(input_lengths, n_window=50):
    chunk_len = n_window * 2
    leave = input_lengths % chunk_len
    feat_lengths = (leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // chunk_len) * 13


def _post_cnn_length(lengths):
    for _ in range(3):
        lengths = torch.where(lengths > 0, (lengths - 1) // 2 + 1, torch.zeros_like(lengths))
    return lengths


def get_audio_cu_seqlens(chunk_lengths, feature_lens, n_window_infer, n_window):
    aftercnn_lens = _feat_extract_output_lengths(feature_lens, n_window)
    feature_lens_after_cnn = _feat_extract_output_lengths(chunk_lengths, n_window)
    max_len_after_cnn = int(feature_lens_after_cnn.max().item())
    n_window_ratio = n_window_infer // (n_window * 2)
    window_aftercnn = max_len_after_cnn * n_window_ratio
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens.tolist():
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    return torch.tensor(cu_chunk_lens, device=feature_lens.device).cumsum(-1, dtype=torch.int32)


class AudioTower(nn.Module):
    def __init__(self, audio_config, device=None, dtype=None, operations=None):
        super().__init__()
        d_model = audio_config.get("d_model", 896)
        self.n_window = audio_config.get("n_window", 50)
        self.n_window_infer = audio_config.get("n_window_infer", 800)
        num_mel_bins = audio_config.get("num_mel_bins", 128)
        down_hidden = audio_config.get("downsample_hidden_size", 480)
        num_layers = audio_config.get("encoder_layers", 18)
        num_heads = audio_config.get("encoder_attention_heads", 14)
        ffn_dim = audio_config.get("encoder_ffn_dim", 3584)
        self.d_model = d_model
        self.conv2d1 = operations.Conv2d(1, down_hidden, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv2d2 = operations.Conv2d(down_hidden, down_hidden, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv2d3 = operations.Conv2d(down_hidden, down_hidden, 3, 2, padding=1, device=device, dtype=dtype)
        freq_bins = num_mel_bins
        for _ in range(3):
            freq_bins = (freq_bins + 2 - 3) // 2 + 1
        self.conv_out = operations.Linear(down_hidden * freq_bins, d_model, bias=False, device=device, dtype=dtype)
        length, channels = 13, d_model
        log_timescale = math.log(10000) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
        pos_emb = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        self.register_buffer("positional_embedding", pos_emb.to(dtype), persistent=False)
        self.layers = nn.ModuleList([
            AudioEncoderLayer(d_model, num_heads, ffn_dim, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])
        self.ln_post = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, input_features, input_features_mask):
        batch_size, _, padded_len = input_features.shape
        chunk_len = self.n_window * 2
        num_chunks = padded_len // chunk_len
        feature_lens = input_features_mask.sum(-1).to(torch.long)
        chunk_lengths = input_features_mask.view(batch_size, num_chunks, chunk_len).sum(dim=-1).reshape(-1).to(torch.long)
        cu_seqlens = get_audio_cu_seqlens(chunk_lengths, feature_lens, self.n_window_infer, self.n_window)
        chunked = input_features.view(batch_size, -1, num_chunks, chunk_len).permute(0, 2, 1, 3).reshape(batch_size * num_chunks, 1, -1, chunk_len)
        conv_outs = []
        for i in range(0, chunked.shape[0], 32):
            h = F.gelu(self.conv2d1(chunked[i:i + 32]))
            h = F.gelu(self.conv2d2(h))
            h = F.gelu(self.conv2d3(h))
            conv_outs.append(h)
        conv_out = torch.cat(conv_outs, dim=0)
        total_chunks, _, _, time_steps = conv_out.size()
        conv_out = self.conv_out(conv_out.permute(0, 3, 1, 2).contiguous().view(total_chunks, time_steps, -1))
        conv_out = conv_out + self.positional_embedding[:time_steps].to(device=conv_out.device, dtype=conv_out.dtype)
        chunk_post_lens = _post_cnn_length(input_features_mask.view(batch_size, num_chunks, chunk_len).sum(dim=-1).reshape(-1).to(torch.long))
        valid_mask = torch.arange(time_steps, device=input_features.device).unsqueeze(0) < chunk_post_lens.unsqueeze(1)
        hidden_states = conv_out.reshape(-1, conv_out.shape[-1])[valid_mask.flatten()]
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = torch.split(hidden_states, lengths, dim=0)
        out_chunks = []
        for chunk in splits:
            h = chunk
            for layer in self.layers:
                h = layer(h)
            out_chunks.append(h)
        hidden_states = torch.cat(out_chunks, dim=0)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states


class MultiModalProjector(nn.Module):
    def __init__(self, audio_config, device=None, dtype=None, operations=None):
        super().__init__()
        d_model = audio_config.get("d_model", 896)
        self.linear_1 = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.linear_2 = operations.Linear(d_model, audio_config.get("output_dim", 1024), device=device, dtype=dtype)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(x)))


class Qwen3ASRInner(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.config = config
        self.audio_tower = AudioTower(config["audio_config"], device=device, dtype=dtype, operations=operations)
        self.language_model = Qwen3TextModel(config["text_config"], device=device, dtype=dtype, operations=operations)
        self.multi_modal_projector = MultiModalProjector(config["audio_config"], device=device, dtype=dtype, operations=operations)
        self.audio_token_id = config.get("audio_token_id", 151676)

    def get_audio_features(self, input_features, input_features_mask):
        hidden = self.audio_tower(input_features, input_features_mask)
        return self.multi_modal_projector(hidden)


class Qwen3ASRModel(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.config = config
        self.model = Qwen3ASRInner(config, device=device, dtype=dtype, operations=operations)
        self.device = device
        self.dtype = dtype

    def embed_merged(self, input_ids, audio_embeds):
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        mask = input_ids == self.model.audio_token_id
        n_tok = int(mask.sum().item())
        n_feat = int(audio_embeds.shape[0])
        if n_tok != n_feat:
            raise RuntimeError("Audio tokens {} != audio features {}".format(n_tok, n_feat))
        inputs_embeds[mask] = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds

    @torch.inference_mode()
    def generate(self, input_ids, audio_embeds, max_new_tokens=512, eos_ids=(151643, 151645), temperature=0.0, callback=None):
        device = input_ids.device
        lm = self.model.language_model
        inputs_embeds = self.embed_merged(input_ids, audio_embeds)
        bsz, seq_len, _ = inputs_embeds.shape
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        causal = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)
        hidden, kvs = lm.forward_embeds(inputs_embeds, attention_mask=causal, past_kvs=None, start_pos=0)
        logit_weight = lm.embed_tokens.weight.to(hidden.device)
        logits = F.linear(hidden[:, -1:, :], logit_weight)
        out_ids = []
        next_tok = int(logits[0, 0].argmax().item())
        pos = seq_len
        for step in range(max_new_tokens):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if next_tok in eos_ids:
                break
            out_ids.append(next_tok)
            if callback is not None:
                callback(step + 1)
            tok = torch.tensor([[next_tok]], device=device, dtype=torch.long)
            emb = lm.embed_tokens(tok)
            hidden, kvs = lm.forward_embeds(emb, attention_mask=None, past_kvs=kvs, start_pos=pos)
            pos += 1
            logits = F.linear(hidden[:, -1:, :], logit_weight)
            if temperature > 0:
                probs = torch.softmax(logits[0, 0].float() / temperature, dim=-1)
                next_tok = int(torch.multinomial(probs, 1).item())
            else:
                next_tok = int(logits[0, 0].argmax().item())
        return out_ids
