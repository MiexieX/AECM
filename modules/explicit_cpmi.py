import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplicitCPMI(nn.Module):
    """Counterfactual phrase routing with explicit token binding."""

    def __init__(
        self,
        embed_dim,
        vocab_size,
        phrase_bank_path,
        route_temperature=0.10,
        topk=8,
        max_bias_scale=1.0,
        cls_blend=0.35,
        no_finding_index=13,
    ):
        super(ExplicitCPMI, self).__init__()

        bank = torch.load(phrase_bank_path, map_location="cpu")
        phrase_token_mask = bank["token_mask"].float()
        if phrase_token_mask.numel() == 0:
            raise ValueError(f"Explicit CPMI phrase bank is empty: {phrase_bank_path}")

        token_ids = bank["token_ids"]
        max_len = max(len(ids) for ids in token_ids)
        padded_ids = torch.zeros(len(token_ids), max_len, dtype=torch.long)
        lengths = torch.zeros(len(token_ids), dtype=torch.long)
        for i, ids in enumerate(token_ids):
            lengths[i] = len(ids)
            padded_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        self.route_temperature = route_temperature
        self.topk = max(1, topk)
        self.max_bias_scale = max_bias_scale
        self.cls_blend = cls_blend
        self.no_finding_index = no_finding_index

        self.register_buffer("phrase_token_mask", phrase_token_mask)
        self.register_buffer("phrase_token_ids", padded_ids)
        self.register_buffer("phrase_lengths", lengths)

        self.num_phrases = phrase_token_mask.size(0)
        if phrase_token_mask.size(1) != vocab_size + 1:
            raise ValueError(
                f"Phrase bank vocab mismatch: bank={phrase_token_mask.size(1)} model={vocab_size + 1}"
            )

        self.phrase_memory = nn.Parameter(torch.empty(self.num_phrases, embed_dim))
        nn.init.normal_(self.phrase_memory, std=0.02)

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.key_norm = nn.LayerNorm(embed_dim)
        self.alpha_logit = nn.Parameter(torch.tensor(-6.0))

    @torch.no_grad()
    def initialize_from_token_embeddings(self, embedding_weight):
        for idx in range(self.num_phrases):
            length = int(self.phrase_lengths[idx].item())
            if length <= 0:
                continue
            token_ids = self.phrase_token_ids[idx, :length]
            phrase_embed = embedding_weight[token_ids].mean(dim=0)
            self.phrase_memory[idx].copy_(phrase_embed)

    def _pool_visual_query(self, hv, anatomy_weights, pathology_probs):
        cls_token = hv[:, 0, :]
        if anatomy_weights is not None and torch.is_tensor(anatomy_weights) and anatomy_weights.numel() > 0:
            if anatomy_weights.dim() == 4:
                region_activation = anatomy_weights.mean(dim=1)
            else:
                region_activation = anatomy_weights
            patch_len = region_activation.size(1)
            patch_tokens = hv[:, 1:1 + patch_len, :]
            patch_scores = region_activation.max(dim=-1).values
            patch_scores = patch_scores / patch_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            patch_summary = (patch_tokens * patch_scores.unsqueeze(-1)).sum(dim=1)
        else:
            patch_summary = hv[:, 1:, :].mean(dim=1)

        if pathology_probs is not None and pathology_probs.size(1) > self.no_finding_index:
            anomaly_strength = 1.0 - pathology_probs[:, self.no_finding_index:self.no_finding_index + 1]
        else:
            anomaly_strength = cls_token.new_full((cls_token.size(0), 1), 0.5)
        blend = self.cls_blend + (1.0 - self.cls_blend) * anomaly_strength
        return (1.0 - blend) * cls_token + blend * patch_summary

    def _sparsify_delta(self, delta_w):
        k = min(self.topk, delta_w.size(-1))
        topk_vals, topk_idx = torch.topk(delta_w, k=k, dim=-1)
        sparse = torch.zeros_like(delta_w)
        sparse.scatter_(1, topk_idx, topk_vals)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return sparse

    def forward(self, hv_factual, hv_counterfactual, anatomy_weights=None, pathology_probs=None):
        query_factual = self.query_proj(self._pool_visual_query(hv_factual, anatomy_weights, pathology_probs))
        query_counter = self.query_proj(self._pool_visual_query(hv_counterfactual, anatomy_weights, pathology_probs))

        phrase_keys = self.key_norm(self.key_proj(self.phrase_memory))
        w_factual = F.softmax(query_factual @ phrase_keys.t() / self.route_temperature, dim=-1)
        w_counter = F.softmax(query_counter @ phrase_keys.t() / self.route_temperature, dim=-1)
        delta_w = F.relu(w_factual - w_counter)
        sparse_delta = self._sparsify_delta(delta_w)

        phrase_bias = sparse_delta @ self.phrase_token_mask
        alpha = torch.sigmoid(self.alpha_logit) * self.max_bias_scale
        return phrase_bias, alpha, sparse_delta
