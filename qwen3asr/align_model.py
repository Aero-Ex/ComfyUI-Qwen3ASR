import torch
import torch.nn as nn

from .model import AudioTower, MultiModalProjector, Qwen3TextModel


class Qwen3AlignerInner(nn.Module):
    def __init__(self, config, num_classes, device=None, dtype=None, operations=None):
        super().__init__()
        self.config = config
        self.audio_tower = AudioTower(config["audio_config"], device=device, dtype=dtype, operations=operations)
        text_config = config["text_config"]
        self.language_model = Qwen3TextModel(text_config, device=device, dtype=dtype, operations=operations)
        self.multi_modal_projector = MultiModalProjector(config["audio_config"], device=device, dtype=dtype, operations=operations)
        self.audio_token_id = config.get("audio_token_id", 151676)
        self.score = operations.Linear(text_config.get("hidden_size", 1024), num_classes, bias=False, device=device, dtype=dtype)

    def get_audio_features(self, input_features, input_features_mask):
        hidden = self.audio_tower(input_features, input_features_mask)
        return self.multi_modal_projector(hidden)


class Qwen3AlignerModel(nn.Module):
    def __init__(self, config, num_classes, device=None, dtype=None, operations=None):
        super().__init__()
        self.config = config
        self.model = Qwen3AlignerInner(config, num_classes, device=device, dtype=dtype, operations=operations)
        self.timestamp_token_id = config.get("timestamp_token_id", 151705)
        self.timestamp_segment_time = float(config.get("timestamp_segment_time", 80))
        self.num_classes = num_classes
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

    def align_logits(self, input_ids, audio_embeds):
        inputs_embeds = self.embed_merged(input_ids, audio_embeds)
        bsz, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        causal = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)
        hidden, _ = self.model.language_model.forward_embeds(inputs_embeds, attention_mask=causal, past_kvs=None, start_pos=0)
        return self.model.score(hidden)
