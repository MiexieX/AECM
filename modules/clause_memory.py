import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomyGuidedClauseMemory(nn.Module):
    """Differentiable clause prototype memory.

    This module upgrades phrase-level latent slots into explicit clause
    prototypes loaded from an offline bank. The bank stores clause texts and
    token ids; actual clause embeddings are initialized from the model's own
    token embedding weights at runtime so the feature space stays aligned with
    the report generator.
    """

    def __init__(
        self,
        embed_dim=512,
        num_regions=11,
        num_pathologies=14,
        clause_bank_path="data/mimic_cxr_clause_prototypes.pt",
        topk=8,
        temperature=0.10,
        cls_blend=0.35,
        dropout=0.1,
    ):
        super(AnatomyGuidedClauseMemory, self).__init__()
        if not os.path.exists(clause_bank_path):
            raise FileNotFoundError(f"Clause prototype bank not found: {clause_bank_path}")

        bank = torch.load(clause_bank_path, map_location="cpu")
        clauses = bank.get("clauses", [])
        token_ids = bank.get("token_ids", [])
        if len(clauses) == 0 or len(token_ids) == 0:
            raise ValueError(f"Clause prototype bank is empty: {clause_bank_path}")

        self.num_clauses = len(clauses)
        self.num_regions = num_regions
        self.num_pathologies = num_pathologies
        self.topk = max(1, min(int(topk), self.num_clauses))
        self.temperature = float(temperature)
        self.cls_blend = float(cls_blend)

        max_len = max(len(ids) for ids in token_ids)
        padded_ids = torch.zeros(self.num_clauses, max_len, dtype=torch.long)
        lengths = torch.zeros(self.num_clauses, dtype=torch.float32)
        for i, ids in enumerate(token_ids):
            cur = torch.tensor(ids, dtype=torch.long)
            padded_ids[i, : cur.numel()] = cur
            lengths[i] = float(cur.numel())

        self.register_buffer("clause_token_ids", padded_ids, persistent=True)
        self.register_buffer("clause_lengths", lengths.clamp_min(1.0), persistent=True)
        self.clause_slots = nn.Parameter(torch.empty(self.num_clauses, embed_dim))
        nn.init.xavier_uniform_(self.clause_slots)

        self.query_proj = nn.Linear(embed_dim + num_pathologies, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.key_norm = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Near-zero warm start keeps Stage-I language behavior intact.
        self.memory_logit = nn.Parameter(torch.tensor(-6.0))
        self.initialized_from_embeddings = False

    def initialize_from_token_embeddings(self, embedding_weight):
        with torch.no_grad():
            clause_embed = embedding_weight[self.clause_token_ids]  # [C, T, D]
            mask = (self.clause_token_ids != 0).float().unsqueeze(-1)
            summed = (clause_embed * mask).sum(dim=1)
            mean_embed = summed / self.clause_lengths.unsqueeze(-1).clamp_min(1.0)
            self.clause_slots.copy_(mean_embed)
        self.initialized_from_embeddings = True

    def _region_activation(self, anatomy_weights, batch_size, device):
        if anatomy_weights is None or not torch.is_tensor(anatomy_weights) or anatomy_weights.nelement() == 0:
            return torch.full((batch_size, self.num_regions), 1.0 / self.num_regions, device=device)

        if anatomy_weights.dim() == 4:
            region_activation = anatomy_weights.mean(dim=(1, 2))
        else:
            region_activation = anatomy_weights.mean(dim=1)

        if region_activation.size(0) != batch_size:
            if region_activation.size(0) > 0 and batch_size % region_activation.size(0) == 0:
                repeat = batch_size // region_activation.size(0)
                region_activation = region_activation.repeat_interleave(repeat, dim=0)
            else:
                region_activation = region_activation[:batch_size]

        region_activation = region_activation.clamp_min(1e-6)
        region_activation = region_activation / region_activation.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return region_activation

    def _visual_summary(self, hv, anatomy_weights):
        batch_size = hv.size(0)
        device = hv.device
        patch_tokens = hv[:, 1:, :]
        cls_token = hv[:, :1, :].squeeze(1)

        region_activation = self._region_activation(anatomy_weights, batch_size, device)
        if anatomy_weights is None or not torch.is_tensor(anatomy_weights) or anatomy_weights.nelement() == 0:
            patch_summary = patch_tokens.mean(dim=1)
        else:
            if anatomy_weights.dim() == 4:
                patch_region = anatomy_weights.mean(dim=1)
            else:
                patch_region = anatomy_weights
            if patch_region.size(0) != batch_size:
                repeat = batch_size // patch_region.size(0)
                patch_region = patch_region.repeat_interleave(repeat, dim=0)
            patch_region = patch_region / patch_region.sum(dim=1, keepdim=True).clamp_min(1e-6)
            region_features = torch.einsum("bpr,bpd->brd", patch_region, patch_tokens)
            patch_summary = torch.bmm(region_activation.unsqueeze(1), region_features).squeeze(1)

        return self.cls_blend * cls_token + (1.0 - self.cls_blend) * patch_summary

    def forward(self, hv, anatomy_weights, pathology_probs):
        batch_size = hv.size(0)
        visual_summary = self._visual_summary(hv, anatomy_weights)
        if pathology_probs is None:
            pathology_probs = torch.zeros(
                batch_size, self.num_pathologies, device=hv.device, dtype=hv.dtype
            )

        query = torch.cat([visual_summary, pathology_probs], dim=-1)
        query = self.query_proj(query)

        keys = self.key_norm(self.key_proj(self.clause_slots))
        scores = torch.matmul(query, keys.transpose(0, 1)) / max(self.temperature, 1e-6)
        attn = F.softmax(scores, dim=-1)

        if self.topk < self.num_clauses:
            top_vals, top_idx = torch.topk(attn, k=self.topk, dim=-1)
            sparse = torch.zeros_like(attn)
            sparse.scatter_(1, top_idx, top_vals)
            attn = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        clause_token = torch.matmul(attn, self.clause_slots).unsqueeze(1)
        clause_token = self.out_norm(clause_token)
        clause_token = self.dropout(clause_token)
        scale = torch.sigmoid(self.memory_logit)
        clause_token = scale * clause_token
        return clause_token, scale.detach()
