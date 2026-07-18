import torch
import torch.nn as nn


class AnatomyPathologyMemoryPrompt(nn.Module):
    """Lightweight anatomy/pathology prompt prefix for decoder cross-attention.

    The module keeps structured memory banks internally, but exports only three
    prompt tokens per image: anatomy-guided, pathology-guided, and
    template-guided. This keeps the decoder context short and reduces the risk
    of prompt tokens stealing too much attention from visual evidence.
    """

    def __init__(
        self,
        embed_dim=512,
        num_regions=11,
        num_pathologies=14,
        num_templates=5,
        dropout=0.1,
        use_confidence_gated_pathology_prompt=True,
        pathology_gate_threshold=0.5,
        pathology_gate_temperature=0.12,
        use_normal_aware_suppression=True,
        normal_suppression_threshold=0.35,
        normal_suppression_temperature=0.10,
    ):
        super(AnatomyPathologyMemoryPrompt, self).__init__()
        self.num_regions = num_regions
        self.num_pathologies = num_pathologies
        self.num_templates = num_templates
        self.use_confidence_gated_pathology_prompt = use_confidence_gated_pathology_prompt
        self.pathology_gate_threshold = pathology_gate_threshold
        self.pathology_gate_temperature = pathology_gate_temperature
        self.use_normal_aware_suppression = use_normal_aware_suppression
        self.normal_suppression_threshold = normal_suppression_threshold
        self.normal_suppression_temperature = normal_suppression_temperature

        self.region_memory = nn.Parameter(torch.empty(1, num_regions, embed_dim))
        self.pathology_memory = nn.Parameter(torch.empty(1, num_pathologies, embed_dim))
        self.template_memory = nn.Parameter(torch.empty(1, num_templates, embed_dim))
        nn.init.xavier_uniform_(self.region_memory)
        nn.init.xavier_uniform_(self.pathology_memory)
        nn.init.xavier_uniform_(self.template_memory)

        self.visual_norm = nn.LayerNorm(embed_dim)
        self.prompt_norm = nn.LayerNorm(embed_dim)
        self.pathology_head = nn.Linear(embed_dim, num_pathologies)
        self.template_head = nn.Linear(embed_dim, num_templates)
        self.dropout = nn.Dropout(dropout)

        # Keep each prompt branch near-zero at initialization so the model can
        # warm start from the language-preserving distilled baseline.
        self.region_prompt_logit = nn.Parameter(torch.tensor(-6.0))
        self.pathology_prompt_logit = nn.Parameter(torch.tensor(-6.0))
        self.template_prompt_logit = nn.Parameter(torch.tensor(-6.0))

    def _soft_gate(self, probs, threshold, temperature):
        temperature = max(float(temperature), 1e-6)
        return torch.sigmoid((probs - threshold) / temperature)

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

    def forward(self, hv, anatomy_weights=None):
        batch_size = hv.size(0)
        device = hv.device

        visual_summary = self.visual_norm(hv.mean(dim=1))
        region_activation = self._region_activation(anatomy_weights, batch_size, device)
        pathology_logits = self.pathology_head(visual_summary)
        pathology_probs = torch.sigmoid(pathology_logits)
        template_logits = self.template_head(visual_summary)
        template_probs = torch.sigmoid(template_logits)

        region_memory = self.region_memory.expand(batch_size, -1, -1)
        pathology_memory = self.pathology_memory.expand(batch_size, -1, -1)
        template_memory = self.template_memory.expand(batch_size, -1, -1)

        region_prompt = torch.bmm(region_activation.unsqueeze(1), region_memory)
        pathology_gate = pathology_probs
        if self.use_confidence_gated_pathology_prompt:
            pathology_gate = self._soft_gate(
                pathology_probs,
                self.pathology_gate_threshold,
                self.pathology_gate_temperature,
            )
        if self.use_normal_aware_suppression:
            abnormal_strength = pathology_probs.max(dim=-1, keepdim=True).values
            normal_suppress = self._soft_gate(
                abnormal_strength,
                self.normal_suppression_threshold,
                self.normal_suppression_temperature,
            )
            pathology_gate = pathology_gate * normal_suppress
        pathology_prompt = torch.bmm(pathology_gate.unsqueeze(1), pathology_memory) / float(self.num_pathologies)
        template_prompt = torch.bmm(template_probs.unsqueeze(1), template_memory)
        template_prompt = template_prompt / template_probs.sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)

        region_prompt = torch.sigmoid(self.region_prompt_logit) * region_prompt
        pathology_prompt = torch.sigmoid(self.pathology_prompt_logit) * pathology_prompt
        template_prompt = torch.sigmoid(self.template_prompt_logit) * template_prompt
        prompts = torch.cat([region_prompt, pathology_prompt, template_prompt], dim=1)
        prompts = self.dropout(self.prompt_norm(prompts))
        return torch.cat([hv, prompts], dim=1), pathology_logits, template_logits
