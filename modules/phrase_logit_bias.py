import torch
import torch.nn as nn


class AnatomyGuidedPhraseLogitBias(nn.Module):
    """Region-gated phrase retrieval that directly biases decoder logits."""

    def __init__(
        self,
        embed_dim=512,
        vocab_size=0,
        num_regions=11,
        phrases_per_region=50,
        num_heads=8,
        bottleneck_dim=128,
        max_bias_scale=1.5,
    ):
        super(AnatomyGuidedPhraseLogitBias, self).__init__()
        self.num_regions = num_regions
        self.phrases_per_region = phrases_per_region
        self.total_phrases = num_regions * phrases_per_region
        self.max_bias_scale = max_bias_scale

        self.phrase_memory = nn.Parameter(torch.empty(1, self.total_phrases, embed_dim))
        nn.init.xavier_uniform_(self.phrase_memory)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, vocab_size + 1, bias=False)
        nn.init.zeros_(self.up_proj.weight)

        # Near-zero initialization keeps the warm-start behavior close to the checkpoint.
        self.bias_logit = nn.Parameter(torch.tensor(-6.0))

    def _build_phrase_gating(self, anatomy_weights, batch_size, device):
        is_weights_valid = (
            anatomy_weights is not None
            and torch.is_tensor(anatomy_weights)
            and anatomy_weights.nelement() > 0
        )
        if not is_weights_valid:
            return torch.ones(batch_size, self.total_phrases, device=device)

        if anatomy_weights.dim() == 4:
            region_activation = anatomy_weights.mean(dim=(1, 2))
        else:
            region_activation = anatomy_weights.mean(dim=1)

        if region_activation.nelement() == 0:
            region_activation = torch.ones(batch_size, self.num_regions, device=device)
        elif region_activation.size(0) != batch_size:
            beam_size = batch_size // region_activation.size(0)
            region_activation = region_activation.repeat_interleave(beam_size, dim=0)

        return region_activation.unsqueeze(-1).repeat(1, 1, self.phrases_per_region).view(
            batch_size, self.total_phrases
        )

    def forward(self, dec_states, anatomy_weights):
        squeeze_time = False
        if dec_states.dim() == 2:
            dec_states = dec_states.unsqueeze(1)
            squeeze_time = True

        batch_size = dec_states.size(0)
        memory_bank = self.phrase_memory.expand(batch_size, -1, -1)
        phrase_gating = self._build_phrase_gating(anatomy_weights, batch_size, dec_states.device)
        gated_memory = memory_bank * phrase_gating.unsqueeze(-1)

        retrieved_features, _ = self.cross_attn(
            query=dec_states,
            key=gated_memory,
            value=gated_memory,
        )
        retrieved_features = self.norm(retrieved_features)
        bias_hidden = torch.tanh(self.down_proj(retrieved_features))
        bias_logits = self.up_proj(bias_hidden)

        bias_scale = torch.sigmoid(self.bias_logit) * self.max_bias_scale
        bias_logits = bias_scale * bias_logits
        if squeeze_time:
            bias_logits = bias_logits.squeeze(1)
        return bias_logits, bias_scale.detach()
