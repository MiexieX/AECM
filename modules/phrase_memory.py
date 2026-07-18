import torch
import torch.nn as nn

class AnatomyGuidedPhraseMemory(nn.Module):
    def __init__(self, embed_dim=512, num_regions=11, phrases_per_region=50, num_heads=8):
        super(AnatomyGuidedPhraseMemory, self).__init__()
        self.num_regions = num_regions
        self.phrases_per_region = phrases_per_region
        self.total_phrases = num_regions * phrases_per_region
        
        # 显式的 550 个短语记忆槽 (11个区域 * 每个区域50个短语)
        # 这构建了一个结构化的知识库，每个实体对应特定胸部区域
        self.phrase_memory = nn.Parameter(torch.empty(1, self.total_phrases, embed_dim))
        nn.init.xavier_uniform_(self.phrase_memory)
        
        # 交叉注意力：用于根据当前解码状态检索相关的语义实体
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        # 零初始化残差门控：确保微调初期不破坏预训练模型的语言能力
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, dec_states, anatomy_weights):
        """
        dec_states: [B_new, L, D] - 解码器当前状态 (B_new 可能是 Batch * Beam_Size)
        anatomy_weights: 来自 AnatomicalAdapter 的权重，形状可能为 [B, Patches, 11] 或 [B, Heads, Patches, 11]
        """
        B_new = dec_states.size(0)
        memory_bank = self.phrase_memory.expand(B_new, -1, -1) # [B_new, 550, D]

        # === 🌟 核心防御逻辑：确保任何情况下 B_new 都不为 0 且权重有效 🌟 ===
        # 检查权重是否有效
        is_weights_valid = (anatomy_weights is not None) and \
                           (torch.is_tensor(anatomy_weights)) and \
                           (anatomy_weights.nelement() > 0)

        if not is_weights_valid:
            # 如果没有权重（例如 MIMIC 测试中的某些异常情况），直接生成全 1 门控，不进行区域过滤
            phrase_gating = torch.ones(B_new, self.total_phrases).to(dec_states.device)
        else:
            # 1. 预处理权重，将其坍缩为 [Batch_Size, 11]
            if anatomy_weights.dim() == 4:
                region_activation = anatomy_weights.mean(dim=(1, 2)) 
            else:
                region_activation = anatomy_weights.mean(dim=1)

            # 2. 处理 Beam Search 导致的 Batch 维度不一致 (如 16 -> 48)
            # 再次确认 region_activation 本身不为空
            if region_activation.nelement() > 0 and region_activation.size(0) != B_new:
                beam_size = B_new // region_activation.size(0)
                region_activation = region_activation.repeat_interleave(beam_size, dim=0)
            elif region_activation.nelement() == 0:
                # 兜底：如果计算完变成空的了，强制填充 1
                region_activation = torch.ones(B_new, self.num_regions).to(dec_states.device)

            # 3. 生成门控矩阵
            # 使用 B_new 动态推导，确保 view(B_new, ...) 永远合法
            phrase_gating = region_activation.unsqueeze(-1).repeat(1, 1, self.phrases_per_region).view(B_new, self.total_phrases)
        
        # === 检索增强生成 (RAG) ===
        # 4. 施加解剖学门控：根据视觉区域关注度过滤短语
        gated_memory = memory_bank * phrase_gating.unsqueeze(-1)
        
        # 5. 从门控后的记忆库提取细粒度语义
        retrieved_features, _ = self.cross_attn(
            query=dec_states,
            key=gated_memory,
            value=gated_memory
        )
        
        # 6. 残差融合：注入检索到的规范短语特征
        enhanced_states = dec_states + self.beta * self.norm(retrieved_features)
        return enhanced_states