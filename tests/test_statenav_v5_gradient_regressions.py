from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_baselines.models.statenav_v5.attention_head import AttentionHead
from vlnce_baselines.models.statenav_v5.losses import attention_transition_loss
from vlnce_baselines.models.statenav_v5.self_rssm_latent import SelfRSSMLatent
from vlnce_baselines.models.statenav_v5.self_rssm_transition import SelfRSSMTransition
from vlnce_baselines.models.statenav_v5.unified_rollout_trunk import UnifiedRolloutTrunk


def _nonzero_grad(params) -> bool:
    for p in params:
        if p.grad is not None and torch.count_nonzero(p.grad).item() > 0:
            return True
    return False


def test_rollout_prior_net_receives_gradients_in_training():
    torch.manual_seed(0)
    batch_size = 2
    num_candidates = 3
    hidden_dim = 8
    lang_dim = 6
    token_dim = 5
    cand_dim = 4
    latent_groups = 2
    latent_classes = 3
    z_dim = latent_groups * latent_classes
    relation_dim = 7

    transition = SelfRSSMTransition(
        hidden_dim=hidden_dim,
        lang_dim=lang_dim,
        action_dim=cand_dim,
        z_dim=z_dim,
        mlp_hidden_dim=16,
    )
    latent = SelfRSSMLatent(
        hidden_dim=hidden_dim,
        x_dim=token_dim,
        g_dim=token_dim,
        lang_dim=lang_dim,
        latent_groups=latent_groups,
        latent_classes=latent_classes,
        mlp_hidden_dim=16,
    )
    trunk = UnifiedRolloutTrunk(
        hidden_dim=hidden_dim,
        z_dim=z_dim,
        token_dim=token_dim,
        cand_dim=cand_dim,
        relation_dim=relation_dim,
        ffn_hidden_dim=16,
        max_visual_tokens=8,
        max_graph_tokens=8,
    )
    action_encoder = nn.Identity()

    transition.train()
    latent.train()
    trunk.train()

    h_post = torch.randn(batch_size, hidden_dim, requires_grad=True)
    z_post_flat = torch.randn(batch_size, z_dim, requires_grad=True)
    relation_state = torch.randn(batch_size, relation_dim, requires_grad=True)
    cand_feats = torch.randn(batch_size, num_candidates, cand_dim, requires_grad=True)
    lang_tokens = torch.randn(batch_size, 5, lang_dim)
    x_tokens = torch.randn(batch_size, 4, token_dim)
    g_tokens = torch.randn(batch_size, 3, token_dim)
    lang_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    x_mask = torch.ones(batch_size, 4, dtype=torch.bool)
    g_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    candidate_mask = torch.tensor([[False, False, True], [False, True, True]])

    out = trunk(
        transition_model=transition,
        latent_model=latent,
        action_encoder=action_encoder,
        h_post=h_post,
        z_post_flat=z_post_flat,
        relation_state=relation_state,
        cand_feats=cand_feats,
        lang_tokens=lang_tokens,
        x_tokens=x_tokens,
        g_tokens=g_tokens,
        lang_mask=lang_mask,
        x_mask=x_mask,
        g_mask=g_mask,
        candidate_mask=candidate_mask,
        temp=0.7,
        hard=True,
    )
    loss = out["unified_state"].pow(2).mean() + out["candidate_z_next"].pow(2).mean()
    loss.backward()

    assert _nonzero_grad(latent.prior_net.parameters()), "future prior_net should receive rollout gradients"


def test_attention_transition_storage_keeps_gradient_path():
    torch.manual_seed(1)
    batch_size = 2
    num_candidates = 3
    relation_dim = 7
    lang_dim = 5
    lang_len = 6

    head = AttentionHead(relation_dim=relation_dim, lang_dim=lang_dim, hidden_dim=9)
    head.train()

    # step t-1: teacher-forced prediction stored for step t supervision
    attn_post_prev = torch.softmax(torch.randn(batch_size, lang_len), dim=-1)
    unified_prev = torch.randn(batch_size, num_candidates, relation_dim, requires_grad=True)
    lang_tokens = torch.randn(batch_size, lang_len, lang_dim)
    lang_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    predicted_prev = head(
        attn_post=attn_post_prev,
        unified_states=unified_prev,
        lang_tokens=lang_tokens,
        lang_mask=lang_mask,
    )

    teacher_safe = torch.tensor([1, 2], dtype=torch.long)
    teacher_valid = torch.tensor([True, True])
    prev_gt_predicted_attn = [None for _ in range(batch_size)]
    gathered = predicted_prev[torch.arange(batch_size), teacher_safe]
    for i in range(batch_size):
        prev_gt_predicted_attn[i] = gathered[i].clone() if bool(teacher_valid[i].item()) else None

    # step t: target comes from current posterior attention and is detached as a label
    target_attn = torch.softmax(torch.randn(batch_size, lang_len), dim=-1)
    loss = torch.zeros(())
    pairs = 0
    for i in range(batch_size):
        loss = loss + attention_transition_loss(
            predicted_attn=prev_gt_predicted_attn[i].unsqueeze(0),
            target_attn=target_attn[i].detach().unsqueeze(0),
            lang_mask=lang_mask[i].unsqueeze(0),
            reduction="sum",
        )
        pairs += 1
    (loss / pairs).backward()

    assert _nonzero_grad(head.parameters()), "attention head should receive gradients through stored prev prediction"
