from __future__ import annotations

import torch
from torch import Tensor, nn


class SelfMismatchAggregator(nn.Module):
    def __init__(
        self,
        init_mu: tuple[float, float, float] = (0.5, 1.5, 0.75),
        init_sigma: tuple[float, float, float] = (0.5, 1.0, 0.5),
        ema_tau: float = 0.95,
        warmup_steps: int = 20,
    ) -> None:
        super().__init__()
        self.ema_tau = float(ema_tau)
        self.warmup_steps = int(warmup_steps)
        self._stats_frozen = False
        self.register_buffer("prior_mu", torch.tensor(init_mu, dtype=torch.float32))
        self.register_buffer("prior_var", torch.tensor([s * s for s in init_sigma], dtype=torch.float32))
        self.register_buffer("running_mu", torch.tensor(init_mu, dtype=torch.float32))
        self.register_buffer("running_var", torch.tensor([s * s for s in init_sigma], dtype=torch.float32))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def freeze_stats(self) -> None:
        self._stats_frozen = True

    def reset_stats(self) -> None:
        self.running_mu.copy_(self.prior_mu)
        self.running_var.copy_(self.prior_var)
        self.step_count.zero_()
        self._stats_frozen = False

    def _ema_update_channel(self, channel_idx: int, values: Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach()
        batch_mu = values.mean()
        batch_var = values.var(unbiased=False).clamp_min(1e-6)
        self.running_mu[channel_idx].copy_(
            self.running_mu[channel_idx] * self.ema_tau + batch_mu * (1.0 - self.ema_tau)
        )
        self.running_var[channel_idx].copy_(
            self.running_var[channel_idx] * self.ema_tau + batch_var * (1.0 - self.ema_tau)
        )

    def forward(
        self,
        micro_diag: Tensor,
        macro_diag_score: Tensor,
        ground_diag: Tensor,
        macro_valid_mask: Tensor,
        kappa_self: Tensor | None = None,
    ) -> dict[str, Tensor]:
        raw = torch.stack([micro_diag, macro_diag_score, ground_diag], dim=-1)
        if int(self.step_count.item()) < self.warmup_steps:
            mu = self.prior_mu.to(raw.device, raw.dtype)
            var = self.prior_var.to(raw.device, raw.dtype)
        else:
            mu = self.running_mu.to(raw.device, raw.dtype)
            var = self.running_var.to(raw.device, raw.dtype)
        std = var.sqrt().clamp_min(1e-6)
        z = (raw - mu.unsqueeze(0)) / std.unsqueeze(0)
        z_micro = torch.relu(z[:, 0])
        z_macro = torch.where(macro_valid_mask, torch.relu(z[:, 1]), torch.zeros_like(z[:, 1]))
        z_ground = torch.relu(z[:, 2])
        if kappa_self is not None:
            self_penalty = 1.0 - kappa_self.clamp(0.0, 1.0)
        else:
            self_penalty = torch.zeros_like(z_micro)
        mismatch = torch.log1p(z_micro + z_macro + z_ground + self_penalty)

        if self.training and not self._stats_frozen:
            self._ema_update_channel(0, micro_diag)
            if bool(macro_valid_mask.any().item()):
                self._ema_update_channel(1, macro_diag_score[macro_valid_mask])
            self._ema_update_channel(2, ground_diag)
            self.step_count.add_(1)

        return {
            "self_mismatch": mismatch,
            "z_micro": z_micro,
            "z_macro": z_macro,
            "z_ground": z_ground,
            "delta_control": z_micro,
            "delta_boundary": z_macro,
            "delta_progress": z_ground,
        }
