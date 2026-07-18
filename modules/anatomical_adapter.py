import torch
import torch.nn as nn


class AnatomicalAdapter(nn.Module):
    """Residual anatomical adapter with a learnable chest-region knowledge base."""

    CHEST_REGION_NAMES = (
        "heart",
        "left lung",
        "right lung",
        "mediastinum",
        "hila",
        "pleura",
        "diaphragm",
        "ribs",
        "spine",
        "trachea",
        "aortic arch",
    )

    def __init__(self, embed_dim=512, num_regions=11, num_heads=8):
        super(AnatomicalAdapter, self).__init__()

        if num_regions != len(self.CHEST_REGION_NAMES):
            # Keep the code honest about the 11-region knowledge bank.
            num_regions = len(self.CHEST_REGION_NAMES)

        self.num_regions = num_regions
        self.region_names = self.CHEST_REGION_NAMES[:num_regions]

        # Learnable anatomical knowledge base: [1, R, D]
        self.anatomy_dict = nn.Parameter(torch.empty(1, num_regions, embed_dim))
        nn.init.xavier_uniform_(self.anatomy_dict)

        # Query-key-value alignment from image patches to anatomical prototypes.
        self.query_norm = nn.LayerNorm(embed_dim)
        self.dict_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(embed_dim)

        # Zero-initialized residual gate for hot-start fine-tuning.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, v_features, return_attn=False):
        """
        Args:
            v_features: [B, L, D] visual patch tokens.
            return_attn: if True, also return attention weights for analysis.
        """
        B = v_features.size(0)
        query = self.query_norm(v_features)
        dict_features = self.dict_norm(self.anatomy_dict).expand(B, -1, -1)

        anatomy_aware_features, attn_weights = self.cross_attn(
            query=query,
            key=dict_features,
            value=dict_features,
            need_weights=True,
            average_attn_weights=False,
        )

        v_new = v_features + self.alpha * self.out_norm(anatomy_aware_features)
        if return_attn:
            return v_new, attn_weights
        return v_new
