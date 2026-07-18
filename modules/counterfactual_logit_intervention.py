import torch
import torch.nn as nn


class CounterfactualNormalizer(nn.Module):
    """Build a soft normal-reference visual context for causal logit contrast."""

    def __init__(
        self,
        embed_dim,
        num_pathologies=14,
        num_regions=11,
        anomaly_threshold=0.35,
        anomaly_temperature=0.10,
        use_region_aware_normalizer=False,
    ):
        super(CounterfactualNormalizer, self).__init__()
        self.anomaly_threshold = float(anomaly_threshold)
        self.anomaly_temperature = float(anomaly_temperature)
        self.num_regions = int(num_regions)
        self.use_region_aware_normalizer = bool(use_region_aware_normalizer)
        self.normal_prototype = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.region_normal_prototypes = nn.Parameter(torch.zeros(1, self.num_regions, embed_dim))
        self.anomaly_proj = nn.Linear(num_pathologies, 1)
        nn.init.zeros_(self.anomaly_proj.weight)
        nn.init.constant_(self.anomaly_proj.bias, 2.0)

    def _align(self, x, batch_size):
        if x is None:
            return None
        if x.size(0) == batch_size:
            return x
        if x.size(0) > 0 and batch_size % x.size(0) == 0:
            return x.repeat_interleave(batch_size // x.size(0), dim=0)
        return x[:batch_size]

    def _build_normal_context(self, hv_factual, anatomy_weights=None):
        batch_size, token_len, embed_dim = hv_factual.size()
        global_context = self.normal_prototype.expand(batch_size, token_len, embed_dim)
        if (
            not self.use_region_aware_normalizer
            or anatomy_weights is None
            or not torch.is_tensor(anatomy_weights)
            or anatomy_weights.numel() == 0
        ):
            return global_context

        if anatomy_weights.dim() == 4:
            region_activation = anatomy_weights.mean(dim=1)
        else:
            region_activation = anatomy_weights
        if region_activation.size(-1) != self.num_regions:
            return global_context

        patch_len = region_activation.size(1)
        if patch_len <= 0 or token_len < 1 + patch_len:
            return global_context

        region_activation = region_activation / region_activation.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        region_bank = self.region_normal_prototypes.expand(batch_size, -1, -1)
        patch_context = torch.matmul(region_activation, region_bank)

        cls_context = self.normal_prototype.expand(batch_size, 1, embed_dim)
        if token_len > 1 + patch_len:
            extra_context = self.normal_prototype.expand(batch_size, token_len - 1 - patch_len, embed_dim)
            return torch.cat([cls_context, patch_context, extra_context], dim=1)
        return torch.cat([cls_context, patch_context], dim=1)

    def forward(self, hv_factual, pathology_probs=None, anatomy_weights=None):
        batch_size = hv_factual.size(0)
        if pathology_probs is None:
            anomaly_gate = hv_factual.new_zeros(batch_size, 1, 1)
        else:
            pathology_probs = self._align(pathology_probs, batch_size)
            learned_gate = torch.sigmoid(self.anomaly_proj(pathology_probs))
            max_prob = pathology_probs.max(dim=-1, keepdim=True).values
            confidence_gate = torch.sigmoid(
                (max_prob - self.anomaly_threshold) / max(self.anomaly_temperature, 1e-6)
            )
            anomaly_gate = (learned_gate * confidence_gate).unsqueeze(-1)

        normal_context = self._build_normal_context(hv_factual, anatomy_weights=anatomy_weights)
        return hv_factual + anomaly_gate * (normal_context - hv_factual), anomaly_gate.squeeze(-1)


class CounterfactualLogitIntervention(nn.Module):
    """Inject bounded direct causal effect from factual vs. normal-reference logits."""

    def __init__(
        self,
        num_pathologies=14,
        hidden_dim=16,
        max_alpha=0.25,
        anomaly_threshold=0.35,
        anomaly_temperature=0.10,
    ):
        super(CounterfactualLogitIntervention, self).__init__()
        self.max_alpha = float(max_alpha)
        self.anomaly_threshold = float(anomaly_threshold)
        self.anomaly_temperature = float(anomaly_temperature)
        self.alpha_logit = nn.Parameter(torch.tensor(-5.0))
        self.gate_proj = nn.Sequential(
            nn.Linear(num_pathologies, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate_proj[-1].weight)
        nn.init.constant_(self.gate_proj[-1].bias, 2.0)

    def _align(self, x, batch_size):
        if x is None:
            return None
        if x.size(0) == batch_size:
            return x
        if x.size(0) > 0 and batch_size % x.size(0) == 0:
            return x.repeat_interleave(batch_size // x.size(0), dim=0)
        return x[:batch_size]

    def forward(self, logits_factual, logits_counter, pathology_probs=None):
        if pathology_probs is None:
            return logits_factual, logits_factual.new_zeros(1)

        batch_size = logits_factual.size(0)
        pathology_probs = self._align(pathology_probs, batch_size)
        max_prob = pathology_probs.max(dim=-1, keepdim=True).values
        confidence_gate = torch.sigmoid(
            (max_prob - self.anomaly_threshold) / max(self.anomaly_temperature, 1e-6)
        )
        gate = torch.sigmoid(self.gate_proj(pathology_probs)) * confidence_gate
        if logits_factual.dim() == 3:
            gate = gate.unsqueeze(1)

        alpha = torch.sigmoid(self.alpha_logit) * self.max_alpha
        delta = torch.tanh(logits_factual - logits_counter)
        return logits_factual + alpha * gate * delta, alpha.detach()
