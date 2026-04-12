from __future__ import annotations

import torch
from torch import Tensor, nn


class FreeEnergyMonitorV2(nn.Module):
    def __init__(
        self,
        init_mu: tuple[float, float, float] = (5.0, 100.0, 0.01),
        init_sigma: tuple[float, float, float] = (3.0, 50.0, 0.01),
        ema_tau: float = 0.95,
        warmup_steps: int = 5,
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

    def load_stats(self, stats: dict[str, Tensor]) -> None:
        self.running_mu.copy_(stats["running_mu"])
        self.running_var.copy_(stats["running_var"])
        self.step_count.copy_(stats["step_count"])

    def export_stats(self) -> dict[str, Tensor]:
        return {
            "running_mu": self.running_mu.detach().clone(),
            "running_var": self.running_var.detach().clone(),
            "step_count": self.step_count.detach().clone(),
        }

    def _ema_update_channel(self, channel_idx: int, values: Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach()
        batch_mu = values.mean()
        batch_var = values.var(unbiased=False).clamp_min(1e-6)
        new_mu = self.running_mu[channel_idx] * self.ema_tau + batch_mu * (1.0 - self.ema_tau)
        new_var = self.running_var[channel_idx] * self.ema_tau + batch_var * (1.0 - self.ema_tau)
        self.running_mu[channel_idx].copy_(new_mu.to(self.running_mu.device))
        self.running_var[channel_idx].copy_(new_var.to(self.running_var.device))

    def forward(
        self,
        c_micro_diag: Tensor,
        c_macro_diag: Tensor,
        g_t: Tensor,
        macro_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        raw = torch.stack([c_micro_diag, c_macro_diag, g_t], dim=-1)
        if int(self.step_count.item()) < self.warmup_steps:
            mu = self.prior_mu.to(raw.device, raw.dtype)
            var = self.prior_var.to(raw.device, raw.dtype)
        else:
            mu = self.running_mu.to(raw.device, raw.dtype)
            var = self.running_var.to(raw.device, raw.dtype)

        std = var.sqrt().clamp_min(1e-6)
        z = (raw - mu.unsqueeze(0)) / std.unsqueeze(0)
        z_macro = torch.where(macro_valid_mask, z[:, 1], torch.full_like(z[:, 1], -float("inf")))
        z_stack = torch.stack([z[:, 0], z_macro, z[:, 2]], dim=-1)
        a_t, _ = z_stack.max(dim=-1)

        if self.training and not self._stats_frozen:
            self._ema_update_channel(0, c_micro_diag)
            if bool(macro_valid_mask.any().item()):
                self._ema_update_channel(1, c_macro_diag[macro_valid_mask])
            self._ema_update_channel(2, g_t)
            self.step_count.add_(1)

        return {
            "A_t": a_t,
            "z_micro": z[:, 0],
            "z_macro": z_macro,
            "z_ground": z[:, 2],
        }
