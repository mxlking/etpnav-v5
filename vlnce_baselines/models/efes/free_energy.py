from __future__ import annotations

import torch
from torch import Tensor, nn


class FreeEnergyMonitor(nn.Module):
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
        self.register_buffer("running_mu", torch.tensor(init_mu, dtype=torch.float32))
        self.register_buffer("running_var", torch.tensor([s * s for s in init_sigma], dtype=torch.float32))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        c_micro: Tensor,
        c_macro: Tensor,
        g_t: Tensor,
        macro_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        raw = torch.stack([c_micro, c_macro, g_t], dim=-1)
        if int(self.step_count.item()) < self.warmup_steps:
            mu = self.running_mu.to(raw.device, raw.dtype)
            std = self.running_var.to(raw.device, raw.dtype).sqrt()
        else:
            batch_mu = raw.mean(dim=0)
            running_mu = self.running_mu.to(batch_mu.device, batch_mu.dtype)
            running_var = self.running_var.to(batch_mu.device, batch_mu.dtype)
            delta = batch_mu - running_mu
            self.running_mu.copy_(
                (running_mu * self.ema_tau + batch_mu.detach() * (1.0 - self.ema_tau)).to(self.running_mu.device)
            )
            self.running_var.copy_(
                (running_var * self.ema_tau + delta.detach().square() * (1.0 - self.ema_tau)).to(self.running_var.device)
            )
            mu = self.running_mu.to(raw.device, raw.dtype)
            std = self.running_var.to(raw.device, raw.dtype).sqrt()
        self.step_count.add_(1)

        z = (raw - mu.unsqueeze(0)) / std.clamp_min(1e-6).unsqueeze(0)
        z_macro = torch.where(macro_valid_mask, z[:, 1], torch.full_like(z[:, 1], -float("inf")))
        z_stack = torch.stack([z[:, 0], z_macro, z[:, 2]], dim=-1)
        a_t, _ = z_stack.max(dim=-1)
        return {
            "A_t": a_t,
            "z_micro": z[:, 0],
            "z_macro": z[:, 1],
            "z_ground": z[:, 2],
        }
