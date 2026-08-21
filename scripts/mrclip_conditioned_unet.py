"""
mrclip_conditioned_unet.py
===========================
A thin wrapper around DiffusionModelUNetMaisi that replaces the discrete
integer class-label conditioning (nn.Embedding) with a continuous 512-dim
MR-CLIP text embedding projected into the UNet's time-embedding space.

Key design decisions
--------------------
* The base UNet weights are loaded as-is (strict=False) — no surgery on
  the MONAI source.
* A small trainable projection MLP (512 → time_embed_dim) is added.
* A learnable null embedding is used as the unconditional signal for CFG.
* At inference, pass a (B, 512) float tensor as `clip_embedding`.
  For CFG, the caller concatenates [cond, uncond] along batch dim as usual.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)


class MRCLIPConditionedUNet(nn.Module):
    """
    Wraps DiffusionModelUNetMaisi, replacing discrete class-label conditioning
    with a continuous MR-CLIP text embedding.

    Args:
        unet:            Pretrained DiffusionModelUNetMaisi instance.
        clip_embed_dim:  Dimension of incoming MR-CLIP embeddings (default 512).
        dropout_p:       Dropout applied to the projected embedding during
                         training (helps robustness, similar to class-dropout).
    """

    def __init__(
        self,
        unet: DiffusionModelUNetMaisi,
        clip_embed_dim: int = 512,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.unet = unet

        # The UNet's time_embed_dim = num_channels[0] * 4
        time_embed_dim = unet.block_out_channels[0] * 4  # e.g. 64 * 4 = 256

        # Projection MLP: 512 → time_embed_dim  (same structure as MaRaI's embedding layers)
        self.clip_projection = nn.Sequential(
            nn.Linear(clip_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Learnable null embedding used as unconditional token for CFG
        # (a zero initialised parameter that the model learns to interpret as "no contrast info")
        self.null_embedding = nn.Parameter(torch.zeros(clip_embed_dim))

        self.dropout_p = dropout_p

        # Expose key attributes from inner UNet so callers don't need to unwrap
        self.include_top_region_index_input = unet.include_top_region_index_input
        self.include_bottom_region_index_input = unet.include_bottom_region_index_input
        self.include_spacing_input = unet.include_spacing_input
        # Signal to training loop and inner UNet that we handle conditioning ourselves
        self.num_class_embeds = None
        self.unet.num_class_embeds = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_clip(self, clip_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project (B, 512) CLIP embedding → (B, time_embed_dim).
        Applies training dropout: randomly replaces embeddings with the
        learned null embedding to enable CFG at inference.
        """
        if self.training and self.dropout_p > 0:
            mask = (torch.rand(clip_embedding.shape[0], device=clip_embedding.device)
                    < self.dropout_p).unsqueeze(1)  # (B, 1)
            null = self.null_embedding.to(clip_embedding.dtype).unsqueeze(0)  # (1, 512)
            clip_embedding = torch.where(mask, null.expand_as(clip_embedding), clip_embedding)
        return self.clip_projection(clip_embedding)  # (B, time_embed_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a batch of null (unconditional) embeddings for CFG."""
        return self.null_embedding.unsqueeze(0).expand(batch_size, -1).to(device)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        clip_embedding: torch.Tensor,
        spacing_tensor: torch.Tensor | None = None,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        down_block_additional_residuals: tuple | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        clip_proj = self._project_clip(clip_embedding.to(x.dtype))  # (B, time_embed_dim)

        def _ts_emb_hook(module, input, output):
            # Add clip projection directly to time_embed output
            return output + clip_proj

        hook = self.unet.time_embed.register_forward_hook(_ts_emb_hook)

        try:
            out = self.unet(
                x=x,
                timesteps=timesteps,
                context=context,
                class_labels=None,
                spacing_tensor=spacing_tensor,
                top_region_index_tensor=top_region_index_tensor,
                bottom_region_index_tensor=bottom_region_index_tensor,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
            )
        finally:
            hook.remove()

        return out

    # ------------------------------------------------------------------
    # Convenience: freeze / unfreeze UNet backbone
    # ------------------------------------------------------------------

    def freeze_unet(self) -> None:
        """Phase 1: freeze all UNet params, only train clip_projection + null_embedding."""
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.clip_projection.parameters():
            p.requires_grad_(True)
        self.null_embedding.requires_grad_(True)

    def unfreeze_unet(self, lr_scale: float = 1.0) -> None:
        """Phase 2: unfreeze UNet. Pass lr_scale hint to optimizer builder."""
        for p in self.unet.parameters():
            p.requires_grad_(True)

    def trainable_parameter_groups(self, unet_lr: float, proj_lr: float) -> list:
        """
        Return parameter groups for a two-learning-rate setup:
          - clip_projection + null_embedding at proj_lr (higher LR, new params)
          - rest of UNet at unet_lr (lower LR, pretrained params)
        """
        proj_params = list(self.clip_projection.parameters()) + [self.null_embedding]
        unet_params = [p for p in self.unet.parameters() if p.requires_grad]
        groups = [
            {"params": proj_params, "lr": proj_lr, "name": "clip_projection"},
            {"params": unet_params, "lr": unet_lr, "name": "unet_backbone"},
        ]
        return groups
