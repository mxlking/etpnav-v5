from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SelfBinder(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_z: int = 256,
        d_action: int = 128,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.z_proj = nn.Linear(int(d_z), self.d_model)
        self.action_proj = nn.Linear(int(d_action), self.d_model)
        self.body_scalar_proj = nn.Linear(4, self.d_model)
        self.intent_scalar_proj = nn.Linear(2, self.d_model)
        self.trace_scalar_proj = nn.Linear(2, self.d_model)
        self.intent_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=int(num_heads),
            batch_first=True,
        )
        self.trace_fuse = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.body_encoder = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.intent_encoder = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.trace_encoder = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        pair_dim = self.d_model * 4
        self.energy_bi = nn.Sequential(nn.Linear(pair_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.energy_bt = nn.Sequential(nn.Linear(pair_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.energy_it = nn.Sequential(nn.Linear(pair_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.msg_bi = nn.Sequential(nn.Linear(pair_dim, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model))
        self.msg_bt = nn.Sequential(nn.Linear(pair_dim, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model))
        self.msg_it = nn.Sequential(nn.Linear(pair_dim, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model))
        self.self_fuse = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.self_norm = nn.LayerNorm(self.d_model)
        self.progress_head = nn.Linear(self.d_model, 1)

    @staticmethod
    def _pair_features(lhs: Tensor, rhs: Tensor) -> Tensor:
        return torch.cat([lhs, rhs, lhs * rhs, (lhs - rhs).abs()], dim=-1)

    def _pair_energy_and_message(
        self,
        lhs: Tensor,
        rhs: Tensor,
        energy_head: nn.Module,
        message_head: nn.Module,
    ) -> tuple[Tensor, Tensor]:
        pair = self._pair_features(lhs, rhs)
        energy = torch.relu(energy_head(pair)).squeeze(-1)
        message = message_head(pair)
        return energy, message

    def forward(
        self,
        prev_self: Tensor,
        instr_tokens: Tensor,
        prev_z: Tensor,
        prev_action_emb: Tensor,
        prev_prior_alpha: Tensor,
        revisit_count: Tensor,
        loop_evidence: Tensor,
        obs_feat: Tensor,
        frontier_size: Tensor,
        topo_novelty_prev: Tensor,
        progress_gap_prev: Tensor,
        instr_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        body_scalars = torch.stack(
            [prev_prior_alpha, revisit_count, loop_evidence, progress_gap_prev],
            dim=-1,
        )
        body_slot = self.body_encoder(
            torch.cat(
                [
                    self.action_proj(prev_action_emb),
                    self.body_scalar_proj(body_scalars),
                ],
                dim=-1,
            )
        )

        query = prev_self.unsqueeze(1)
        key_padding_mask = None
        if instr_mask is not None:
            key_padding_mask = instr_mask.logical_not()
        attended_instr, attn_weights = self.intent_attn(
            query=query,
            key=instr_tokens,
            value=instr_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        intent_scalars = torch.stack([frontier_size, progress_gap_prev], dim=-1)
        intent_slot = self.intent_encoder(
            torch.cat(
                [
                    attended_instr.squeeze(1),
                    self.intent_scalar_proj(intent_scalars),
                    prev_self,
                ],
                dim=-1,
            )
        )

        trace_scalars = torch.stack([topo_novelty_prev, loop_evidence], dim=-1)
        trace_core = self.trace_fuse(torch.cat([prev_self, self.z_proj(prev_z)], dim=-1))
        trace_slot = self.trace_encoder(
            torch.cat([trace_core, self.trace_scalar_proj(trace_scalars), body_slot], dim=-1)
        )

        e_bi, m_bi = self._pair_energy_and_message(body_slot, intent_slot, self.energy_bi, self.msg_bi)
        e_bt, m_bt = self._pair_energy_and_message(body_slot, trace_slot, self.energy_bt, self.msg_bt)
        e_it, m_it = self._pair_energy_and_message(intent_slot, trace_slot, self.energy_it, self.msg_it)
        energies = torch.stack([e_bi, e_bt, e_it], dim=-1)
        gates = torch.softmax(-energies, dim=-1)
        z_self = self.self_fuse(
            torch.cat(
                [
                    gates[:, 0:1] * m_bi,
                    gates[:, 1:2] * m_bt,
                    gates[:, 2:3] * m_it,
                ],
                dim=-1,
            )
        )
        z_self = self.self_norm(prev_self + z_self)

        total_energy = energies.sum(dim=-1)
        kappa_self = torch.sigmoid(2.0 - total_energy)
        progress_t = torch.sigmoid(self.progress_head(z_self))

        mean_attn = attn_weights.mean(dim=1).squeeze(1).clamp_min(1e-8)
        attn_entropy = -(mean_attn * mean_attn.log()).sum(dim=-1) / math.log(mean_attn.size(-1) + 1e-8)
        binding_stats = {
            "energy_bi": e_bi,
            "energy_bt": e_bt,
            "energy_it": e_it,
            "attn_entropy": attn_entropy,
        }
        return z_self, kappa_self, progress_t, binding_stats
