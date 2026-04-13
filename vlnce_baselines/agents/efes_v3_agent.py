from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from vlnce_baselines.models.efes_v3 import Corrector, Predictor, SelfState


class EFESV3Agent(nn.Module):
    def __init__(
        self,
        x_dim: int,
        lang_dim: int,
        cand_dim: int,
        d_model: int = 768,
        d_z: int = 256,
        d_h: int = 512,
        d_action: int = 128,
        max_nodes: int = 50,
        sigma_min: float = 0.01,
        gate_bias: float = -2.0,
        max_delta: float = 1.0,
        use_somatic_loop: bool = True,
        use_predictor: bool = True,
        use_corrector: bool = True,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.lang_dim = int(lang_dim)
        self.cand_dim = int(cand_dim)
        self.d_model = int(d_model)
        self.d_z = int(d_z)
        self.d_h = int(d_h)
        self.d_action = int(d_action)
        self.max_nodes = int(max_nodes)
        self.use_predictor = bool(use_predictor)
        self.use_corrector = bool(use_corrector)

        self.self_state = SelfState(
            d_model=self.d_model,
            d_obs=self.x_dim,
            d_z=self.d_z,
            d_action=self.d_action,
            use_somatic_loop=bool(use_somatic_loop),
        )
        self.predictor = Predictor(
            d_model=self.d_model,
            d_obs=self.x_dim,
            d_action=self.d_action,
            d_h=self.d_h,
            d_z=self.d_z,
            feat_dim=self.x_dim,
            sigma_min=float(sigma_min),
        )
        self.corrector = Corrector(
            d_model=self.d_model,
            cand_dim=self.cand_dim,
            max_delta=float(max_delta),
            gate_bias=float(gate_bias),
        )
        self.action_encoder = nn.Linear(self.cand_dim, self.d_action)

    @classmethod
    def from_config(
        cls,
        config: Any,
        x_dim: int,
        lang_dim: int,
        cand_dim: int,
    ) -> "EFESV3Agent":
        efes_cfg = config.EFES_V3
        return cls(
            x_dim=x_dim,
            lang_dim=lang_dim,
            cand_dim=cand_dim,
            d_model=int(efes_cfg.d_model),
            d_z=int(efes_cfg.d_z),
            d_h=int(efes_cfg.d_h),
            d_action=int(efes_cfg.d_action),
            max_nodes=int(efes_cfg.max_nodes),
            sigma_min=float(efes_cfg.sigma_min),
            gate_bias=float(efes_cfg.gate_bias),
            max_delta=float(efes_cfg.max_delta),
            use_somatic_loop=bool(getattr(efes_cfg, "use_somatic_loop", True)),
            use_predictor=bool(getattr(efes_cfg, "use_predictor", True)),
            use_corrector=bool(getattr(efes_cfg, "use_corrector", True)),
        )

    def init_recurrent_state(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        return {
            "prev_self": torch.zeros(batch_size, self.d_model, device=device),
            "prev_rssm_h": torch.zeros(batch_size, self.d_h, device=device),
            "prev_z": torch.zeros(batch_size, self.d_z, device=device),
            "prev_action_emb": torch.zeros(batch_size, self.d_action, device=device),
            "prev_surprise": torch.zeros(batch_size, device=device),
            "prev_progress": torch.zeros(batch_size, 1, device=device),
        }

    def project_action_features(self, action_feats: Tensor) -> Tensor:
        return self.action_encoder(action_feats)

    def apply_static_freeze(self) -> None:
        return None

    @staticmethod
    def _masked_mean(scores: Tensor, valid_mask: Tensor) -> Tensor:
        count = valid_mask.sum(dim=-1).clamp_min(1)
        safe_scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
        return safe_scores.sum(dim=-1) / count.to(scores.dtype)

    def forward_step(
        self,
        x_t: Tensor,
        node_feat: Tensor,
        lang_tokens: Tensor,
        cand_feats: Tensor,
        etp_candidate_scores: Tensor,
        prev_action_emb: Tensor,
        prev_z: Tensor,
        prev_self: Tensor,
        prev_rssm_h: Tensor,
        prev_progress: Tensor,
        prev_surprise: Tensor,
        topo_bank_feat: Tensor,
        topo_bank_mask: Tensor,
        candidate_mask: Tensor,
        lang_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        self_t, progress_t = self.self_state(
            prev_self=prev_self,
            instr_tokens=lang_tokens,
            prev_z=prev_z,
            prev_action_emb=prev_action_emb,
            prev_surprise=prev_surprise,
            prev_progress=prev_progress,
            obs_feat=x_t,
            instr_mask=lang_mask,
        )

        if self.use_predictor:
            pred = self.predictor(
                s_t=self_t,
                obs_feat=x_t,
                h_prev=prev_rssm_h,
                a_prev_emb=prev_action_emb,
                node_feat=node_feat,
                bank_feat=topo_bank_feat,
                bank_mask=topo_bank_mask,
            )
            surprise_t = pred["surprise_t"]
            h_t = pred["h_t"]
            z_post = pred["z_post"]
            loss_micro = pred["loss_micro"]
            loss_macro = pred["loss_macro"]
            kl = pred["kl"]
            nll = pred["nll"]
            compressed_node_feat = pred["compressed_node_feat"]
        else:
            zero = torch.zeros(self_t.size(0), device=self_t.device, dtype=self_t.dtype)
            surprise_t = zero
            h_t = prev_rssm_h
            z_post = prev_z
            loss_micro = zero
            loss_macro = zero
            kl = zero
            nll = zero
            compressed_node_feat = node_feat.detach()

        if self.use_corrector:
            corr = self.corrector(
                s_t=self_t,
                surprise_t=surprise_t,
                etp_logits=etp_candidate_scores,
                cand_embeds=cand_feats,
                cand_mask=candidate_mask,
            )
            corrected_logits = corr["corrected_logits"]
            gate = corr["gate"]
            delta = corr["delta"]
        else:
            corrected_logits = etp_candidate_scores.masked_fill(candidate_mask, -1.0e4)
            gate = torch.zeros(self_t.size(0), device=self_t.device, dtype=self_t.dtype)
            delta = torch.zeros_like(etp_candidate_scores)

        valid_mask = candidate_mask.logical_not()
        return {
            "corrected_logits": corrected_logits,
            "gate": gate,
            "delta": delta,
            "surprise_t": surprise_t,
            "progress_t": progress_t.squeeze(-1),
            "loss_micro": loss_micro,
            "loss_macro": loss_macro,
            "kl": kl,
            "nll": nll,
            "self_t": self_t,
            "rssm_h_t": h_t,
            "z_flat": z_post,
            "compressed_node_feat": compressed_node_feat,
            "score_before_mean": self._masked_mean(etp_candidate_scores, valid_mask),
            "score_after_mean": self._masked_mean(corrected_logits, valid_mask),
        }

    def forward(self, **kwargs: Any) -> Dict[str, Tensor]:
        return self.forward_step(**kwargs)
