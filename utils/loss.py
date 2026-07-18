"""
Language Model loss
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_im_loss(pred, imgs, mask, p=16, norm_pix_loss=False):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    target = patchify(imgs, p)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    # p = self.patch_embed.patch_size[0]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x


def compute_lm_loss(output, reports_ids, reports_masks):
    if isinstance(output, list):
        output = output[0]
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


def compute_cls_loss(logits, targets, valid_mask=None):
    """Multi-label BCE loss for optional pseudo-label supervision.

    All-zero labels are valid negative samples, so they must not be ignored by
    default. Pass valid_mask only to skip samples whose pseudo labels are absent.
    """
    targets = targets.to(dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(dim=-1)
    if valid_mask is not None:
        valid_mask = valid_mask.to(dtype=loss.dtype, device=loss.device)
        loss = (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    else:
        loss = loss.mean()
    return loss


def compute_distill_loss(student_logits, teacher_logits, mask, temperature=2.0):
    """Token-level KL distillation for preserving report language behavior.

    student_logits and teacher_logits are raw pre-softmax logits with shape
    [B, L, V]. mask is [B, L] with 1 for valid language positions.
    """
    length = min(student_logits.size(1), teacher_logits.size(1), mask.size(1))
    student_logits = student_logits[:, :length, :]
    teacher_logits = teacher_logits[:, :length, :]
    mask = mask[:, :length].to(dtype=student_logits.dtype, device=student_logits.device)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(dim=-1)
    loss = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    return loss * (temperature ** 2)


def compute_recon_loss(pred, target, mask, mode='text'):
    if mode == 'text':
        loss = compute_lm_loss(pred, target, mask)
    elif mode == 'img':
        loss = compute_im_loss(pred, target, mask)
    else:
        raise ValueError
    return loss
