from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SelfRSSMLatent(nn.Module):
    """RSSM latent interface with prior and posterior categorical heads."""

    def __init__(
        self,
        hidden_dim: int,
        x_dim: int,
        g_dim: int,
        lang_dim: int,
        latent_groups: int = 16,
        latent_classes: int = 16,
        mlp_hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.latent_groups = latent_groups
        self.latent_classes = latent_classes
        latent_dim = latent_groups * latent_classes

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, latent_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + x_dim + g_dim + lang_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, latent_dim),
        )

    def forward(
        self,
        h_prior: Tensor,
        h_post: Tensor,
        x_t: Tensor,
        g_t: Tensor,
        lang_glimpse_obs: Tensor,
        temp: float = 1.0,
        hard: bool = True,
    ) -> Dict[str, Tensor]:
        batch_size = h_post.size(0)

        prior_logits = self.prior_net(h_prior).view(
            batch_size, self.latent_groups, self.latent_classes
        )
        post_input = torch.cat([h_post, x_t, g_t, lang_glimpse_obs], dim=-1)
        post_logits = self.posterior_net(post_input).view(
            batch_size, self.latent_groups, self.latent_classes
        )

        z_prior_indices = prior_logits.argmax(dim=-1)
        z_prior_onehot = F.one_hot(
            z_prior_indices, num_classes=self.latent_classes
        ).to(prior_logits.dtype)

        if self.training:
            z_post_onehot = F.gumbel_softmax(
                post_logits.view(-1, self.latent_classes),
                tau=temp,
                hard=hard,
                dim=-1,
            ).view(batch_size, self.latent_groups, self.latent_classes)
        else:
            z_indices = post_logits.argmax(dim=-1)
            z_post_onehot = F.one_hot(
                z_indices, num_classes=self.latent_classes
            ).to(post_logits.dtype)

        z_prior_flat = z_prior_onehot.flatten(start_dim=1)
        z_post_flat = z_post_onehot.flatten(start_dim=1)

        return {
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "z_prior_onehot": z_prior_onehot,
            "z_post_onehot": z_post_onehot,
            "z_prior_flat": z_prior_flat,
            "z_post_flat": z_post_flat,
            "z_onehot": z_post_onehot,
            "z_flat": z_post_flat,
        }

