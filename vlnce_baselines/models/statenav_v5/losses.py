from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def planner_loss(
    rescored_scores: Tensor,
    teacher_cand_label: Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    return F.cross_entropy(
        rescored_scores,
        teacher_cand_label.long(),
        ignore_index=ignore_index,
        reduction=reduction,
    )


def latent_kl_loss(
    post_logits: Tensor,
    prior_logits: Tensor,
    free_bits: float = 0.0,
    reduction: str = "mean",
) -> tuple[Tensor, Tensor]:
    """
    Returns:
        kl_loss: scalar if reduction != "none", else [B]
        kl_per_sample: [B]
    """
    post_log_probs = torch.log_softmax(post_logits, dim=-1)
    prior_log_probs = torch.log_softmax(prior_logits, dim=-1)
    post_probs = post_log_probs.exp()
    group_kl = torch.sum(post_probs * (post_log_probs - prior_log_probs), dim=-1)
    if free_bits > 0:
        group_kl = torch.clamp_min(group_kl, free_bits)
    kl_per_sample = torch.sum(group_kl, dim=-1)

    if reduction == "none":
        return kl_per_sample, kl_per_sample
    if reduction == "sum":
        return kl_per_sample.sum(), kl_per_sample
    return kl_per_sample.mean(), kl_per_sample


def progress_curve_loss(
    pred_curve: Tensor,
    gt_curve: Tensor,
    valid_mask: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Self-supervised future progress-curve regression.

    `pred_curve` and `gt_curve` are shaped [T, H] or [B, H].
    `valid_mask` is shaped [T] / [B] and marks steps whose current progress label is
    considered reliable enough to supervise.
    """
    per_step = F.mse_loss(pred_curve, gt_curve, reduction="none").mean(dim=-1)
    valid_mask = valid_mask.bool()
    if not valid_mask.any():
        return per_step.new_zeros(())
    if reduction == "sum":
        return per_step[valid_mask].sum()
    return per_step[valid_mask].mean()

def attention_transition_loss(
    predicted_attn: Tensor,
    target_attn: Tensor,
    lang_mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    eps = torch.finfo(predicted_attn.dtype).eps
    if lang_mask is not None:
        valid = lang_mask.bool()
        predicted_attn = predicted_attn * valid.to(predicted_attn.dtype)
        target_attn = target_attn * valid.to(target_attn.dtype)
    target_norm = target_attn / target_attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    predicted_norm = predicted_attn / predicted_attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    losses = -(target_norm * torch.log(predicted_norm.clamp_min(eps))).sum(dim=-1)
    if reduction == "sum":
        return losses.sum()
    return losses.mean() if losses.numel() > 0 else predicted_attn.new_zeros(())
