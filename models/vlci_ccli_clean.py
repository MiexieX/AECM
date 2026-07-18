"""
Clean CCLI-only model path for reproducible Stage-II experiments.

This file intentionally keeps only the components needed by the strongest
historical line:

- anatomical adapter
- anatomy/pathology memory prompt
- counterfactual normalizer
- counterfactual logit intervention

It excludes later exploratory branches such as phrase memory, phrase logit
bias, clause memory, and Explicit CPMI. Optional CCD decoding is supported as
a pure inference-time variant and is disabled by default.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.modules4vlp import get_ht_mask, get_hv_mask, get_cross_mask
from modules.modules4transformer import LayerNorm, Encoder, DecoderLayer
from models.baseline import Baseline
from utils import tensor_utils
from modules.modules4vlci import LocalSample, LDM, GlobalSample, VDM
from modules.anatomical_adapter import AnatomicalAdapter
from modules.memory_prompt import AnatomyPathologyMemoryPrompt
from modules.counterfactual_logit_intervention import (
    CounterfactualLogitIntervention,
    CounterfactualNormalizer,
)


class VLCICCLIClean(Baseline):
    def __init__(self, args, tokenizer):
        super(VLCICCLIClean, self).__init__(args, tokenizer)
        self.z = None
        self.fl = None
        self.vocab = torch.arange(0, self.vocab_size + 1).unsqueeze(-1).long().cuda()
        self.text_embed_z = self.text_embed.__class__(
            embed_dim=self.embed_dim,
            vocab_size=self.vocab_size + 1,
            dropout=self.dropout,
        )
        self.z_norm = LayerNorm(self.embed_dim)
        self.fl_norm = LayerNorm(self.embed_dim)

        self.use_anatomical_adapter = self._flag(args.get("use_anatomical_adapter", 1))
        self.use_memory_prompt = self._flag(args.get("use_memory_prompt", 1))
        self.use_counterfactual_intervention = self._flag(args.get("use_counterfactual_intervention", 1))
        self.ccd_gamma = float(args.get("ccd_gamma", 0.0))
        self.ccd_use_confidence_gate = self._flag(args.get("ccd_use_confidence_gate", 1))

        self.anatomical_adapter = AnatomicalAdapter(
            embed_dim=self.embed_dim,
            num_regions=11,
            num_heads=self.num_heads,
        )

        self.encoder = CausalEncoder(
            embed_dim=self.embed_dim,
            num_layer=self.en_num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            mode=args["v_causal"],
        )

        self.decoder = CausalDecoder(
            embed_dim=self.embed_dim,
            num_layer=self.de_num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            mode=args["l_causal"],
        )

        self.memory_prompt = AnatomyPathologyMemoryPrompt(
            embed_dim=self.embed_dim,
            num_regions=11,
            num_pathologies=14,
            dropout=self.dropout,
            use_confidence_gated_pathology_prompt=self._flag(
                args.get("use_confidence_gated_pathology_prompt", 1)
            ),
            pathology_gate_threshold=float(args.get("pathology_gate_threshold", 0.5)),
            pathology_gate_temperature=float(args.get("pathology_gate_temperature", 0.12)),
            use_normal_aware_suppression=self._flag(
                args.get("use_normal_aware_suppression", 1)
            ),
            normal_suppression_threshold=float(args.get("normal_suppression_threshold", 0.35)),
            normal_suppression_temperature=float(args.get("normal_suppression_temperature", 0.10)),
        )

        self.counterfactual_normalizer = CounterfactualNormalizer(
            embed_dim=self.embed_dim,
            num_pathologies=14,
            num_regions=11,
            anomaly_threshold=float(args.get("ccli_anomaly_threshold", 0.35)),
            anomaly_temperature=float(args.get("ccli_anomaly_temperature", 0.10)),
            use_region_aware_normalizer=False,
        )
        self.counterfactual_intervention = CounterfactualLogitIntervention(
            num_pathologies=14,
            hidden_dim=int(args.get("ccli_hidden_dim", 16)),
            max_alpha=float(args.get("ccli_max_alpha", 0.25)),
            anomaly_threshold=float(args.get("ccli_anomaly_threshold", 0.35)),
            anomaly_temperature=float(args.get("ccli_anomaly_temperature", 0.10)),
        )
        self._sample_counter_out = None
        self._sample_pathology_probs = None

    @staticmethod
    def _flag(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).lower() in {"1", "true", "t", "yes", "y"}

    def _sample_confidence_gate(self, pathology_probs):
        if pathology_probs is None:
            return None
        max_prob = pathology_probs.max(dim=-1, keepdim=True).values
        return torch.sigmoid(
            (max_prob - self.counterfactual_intervention.anomaly_threshold)
            / max(self.counterfactual_intervention.anomaly_temperature, 1e-6)
        )

    def _forward(self, hv, targets, mode, B):
        hv = hv.reshape([B, -1, self.embed_dim])
        if self.use_anatomical_adapter:
            hv, self.anatomy_weights = self.anatomical_adapter(hv, return_attn=True)
        else:
            self.anatomy_weights = None

        cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
        hv = torch.cat((cls_tokens, hv), dim=1)

        hv_mask = get_hv_mask(hv)
        hv, mediator = self.encoder(hv, hv_mask, self.vis_embed.pos_embed, proj=self.args['causal_proj'])

        if self.use_memory_prompt:
            hv, pathology_logits, template_logits = self.memory_prompt(hv, self.anatomy_weights)
        else:
            pathology_logits = None
            template_logits = None

        if self.args["l_causal"] != 'n':
            z = self.text_embed_z.embed(self.vocab).reshape(1, -1, hv.size(-1)).detach()
            z = self.z_norm(z)
            self.fl = self.fl_norm(mediator['local'])
            self.z = z.repeat(hv.size(0), 1, 1)

        if mode == 'train':
            cross_mask = get_cross_mask(hv, targets)
            ht_mask, targets = get_ht_mask(targets)
            ht = self.text_embed(targets)
            out = self.decoder(ht, hv, self_mask=ht_mask, cross_mask=cross_mask,
                               z=self.z, fl=self.fl, proj=self.args['causal_proj'])
            factual_ht = self.decoder.ht.detach()

            raw_logits = self.logit(out)
            counter_logits = None
            ccli_alpha = None
            if self.use_counterfactual_intervention and pathology_logits is not None:
                pathology_probs = torch.sigmoid(pathology_logits.detach())
                hv_counter, _ = self.counterfactual_normalizer(hv, pathology_probs)
                out_counter = self.decoder(ht, hv_counter, self_mask=ht_mask, cross_mask=cross_mask,
                                           z=self.z, fl=self.fl, proj=self.args['causal_proj'])
                counter_logits = self.logit(out_counter)
                raw_logits, ccli_alpha = self.counterfactual_intervention(
                    raw_logits,
                    counter_logits,
                    pathology_probs,
                )

            log_probs = F.log_softmax(raw_logits, dim=-1)
            outputs = [
                log_probs,
                mediator,
                {
                    'ht': factual_ht,
                    'hv': hv.detach(),
                    'pathology_logits': pathology_logits,
                    'template_logits': template_logits,
                    'raw_logits': raw_logits,
                    'counter_logits': counter_logits,
                    'ccli_alpha': ccli_alpha,
                }
            ]
        elif mode == 'sample':
            if self.use_counterfactual_intervention and pathology_logits is not None:
                self._sample_pathology_probs = torch.sigmoid(pathology_logits.detach())
                self.beam_search.load_model(self.sample_forward, self.sample_logit)
            else:
                self._sample_pathology_probs = None
                self.beam_search.load_model(self.sample_forward, self.logit)
            outputs, _ = self.beam_search.sample_beam(hv)
            self.beam_search.clean_model()
            self._sample_counter_out = None
            self._sample_pathology_probs = None
        else:
            raise ValueError
        return outputs

    def sample_forward(self, hv, ht, v_mask, t_mask):
        ht = self.text_embed(ht)
        v_mask = get_hv_mask(hv)

        if self.z is not None and self.z.size(0) != ht.size(0):
            self.z = tensor_utils.repeat_tensors(self.args["beam_size"], self.z)
            self.fl = tensor_utils.repeat_tensors(self.args["beam_size"], self.fl)
            if self._sample_pathology_probs is not None:
                self._sample_pathology_probs = tensor_utils.repeat_tensors(
                    self.args["beam_size"], self._sample_pathology_probs
                )

        out = self.decoder(ht, hv, self_mask=t_mask, cross_mask=v_mask,
                           z=self.z, fl=self.fl, proj=self.args['causal_proj'])
        if self.use_counterfactual_intervention and self._sample_pathology_probs is not None:
            hv_counter, _ = self.counterfactual_normalizer(hv, self._sample_pathology_probs)
            self._sample_counter_out = self.decoder(
                ht, hv_counter, self_mask=t_mask, cross_mask=v_mask,
                z=self.z, fl=self.fl, proj=self.args['causal_proj']
            )
        else:
            self._sample_counter_out = None
        return out

    def sample_logit(self, out):
        logits_factual = self.logit(out)
        if self._sample_counter_out is None or self._sample_pathology_probs is None:
            return logits_factual
        logits_counter = self.logit(self._sample_counter_out[:, -1, :])
        if self.ccd_gamma > 0:
            delta = logits_factual - logits_counter
            if self.ccd_use_confidence_gate:
                confidence_gate = self._sample_confidence_gate(self._sample_pathology_probs)
                return logits_factual + self.ccd_gamma * confidence_gate * delta
            return logits_factual + self.ccd_gamma * delta
        logits, _ = self.counterfactual_intervention(
            logits_factual,
            logits_counter,
            self._sample_pathology_probs,
        )
        return logits


class CausalEncoder(Encoder):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout, mode='none'):
        super(CausalEncoder, self).__init__(embed_dim, num_layer, num_heads, ff_dim, dropout)
        self.mode = mode
        self.do = VDM(embed_dim, num_heads, ff_dim, dropout)
        self.global_sample = GlobalSample(embed_dim, num_heads, ff_dim, dropout)
        self.local_sample = LocalSample(embed_dim, num_heads, ff_dim, dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, h, mask=None, pos=None, k=6, proj=False):
        attn = []
        for layer in self.layers:
            h = layer(h, mask)
            attn.append(layer.attn.attn)

        if h.size(1) > 197:
            h = h + torch.cat([pos, pos[:, 1:, :]], dim=1)
        else:
            h = h + pos

        fl = self.local_sample(h, attn, k)
        fg = self.global_sample(h)
        mediator = {"local": fl, "global": fg, 'attn': attn}
        h = self.do(h, fl=fl, fg=fg, mode=self.mode, proj=proj)

        h = self.norm(h)
        return h, mediator


class CausalDecoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout, mode='none'):
        super(CausalDecoder, self).__init__()
        self.mode = mode
        self.norm = LayerNorm(embed_dim)
        self.do = LDM(embed_dim)
        self.ht = None
        self.layers = nn.ModuleList(
            [DecoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layer)]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, output, h, self_mask=None, cross_mask=None, z=None, fl=None, proj=False):
        output = self.do(output, z, fl, self.mode, proj=proj)
        self.ht = output
        for i in range(len(self.layers)):
            output = self.layers[i](output, h, self_mask, cross_mask)

        output = self.norm(output)
        return output
