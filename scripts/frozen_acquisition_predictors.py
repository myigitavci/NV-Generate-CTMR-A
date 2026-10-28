"""
scripts/frozen_acquisition_predictors.py
========================================
Pillar 1: Frozen Acquisition Predictors (Zero-Shot MR-CLIP Multimodal Oracle)

Evaluates whether synthesized 3D MRI volumes faithfully correspond to the conditioned
acquisition parameters in the shared 512-dim MR-CLIP hypersphere:
- Categorical metadata (Vendor, Field Strength, Modality):
  Evaluated via canonical Zero-Shot CLIP classification:
      c* = argmax_c <E_img(I), E_text(T_c)>
- Continuous metadata (TE, TR, TI, FA):
  Evaluated via continuous Zero-Shot CLIP Softmax Expected Value:
      w_k = softmax(<E_img(I), E_text(T_k)> / tau)
      P_pred = sum_k w_k * P_k
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("frozen_predictors")


def encode_volume_mrclip(vol_np: np.ndarray, clip_model: Any, device: str) -> np.ndarray:
    """
    Encode a 3D MRI volume using the frozen 3D MR-CLIP visual encoder:
    - Normalizes intensities to [0, 1]
    - Resizes to (128, 128, 128)
    - Automatically handles 1-channel and 3-channel visual conv1 kernels
    - Normalizes output embedding to unit Euclidean norm on S^511
    """
    if vol_np.ndim == 3:
        vol_th = torch.from_numpy(vol_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    elif vol_np.ndim == 4:
        vol_th = torch.from_numpy(vol_np.astype(np.float32)).unsqueeze(0)
    else:
        vol_th = torch.from_numpy(vol_np.astype(np.float32))

    v_min, v_max = float(np.percentile(vol_np, 1.0)), float(np.percentile(vol_np, 99.0))
    if v_max > v_min:
        vol_th = (vol_th - v_min) / (v_max - v_min)
        vol_th = torch.clamp(vol_th, 0.0, 1.0)

    if vol_th.shape[2:] != (128, 128, 128):
        vol_th = F.interpolate(vol_th, size=(128, 128, 128), mode="trilinear", align_corners=False)

    vol_th = vol_th.to(device)

    # Inspect visual conv1 in_channels
    in_ch = 1
    if hasattr(clip_model, "visual"):
        for m in clip_model.visual.modules():
            if isinstance(m, (torch.nn.Conv3d, torch.nn.modules.conv.Conv3d)):
                in_ch = m.in_channels
                break

    if in_ch == 3 and vol_th.shape[1] == 1:
        vol_th = vol_th.repeat(1, 3, 1, 1, 1)

    with torch.no_grad():
        if hasattr(clip_model, "encode_image"):
            vis_feat = clip_model.encode_image(vol_th)
        else:
            vis_feat = clip_model.visual(vol_th)
        vis_feat = vis_feat / (vis_feat.norm(dim=-1, keepdim=True) + 1e-8)
        emb_np = vis_feat.cpu().numpy().squeeze(0)

    return emb_np


class MRCLIPZeroShotPredictor:
    """
    Multimodal Zero-Shot Acquisition Predictor based on the frozen MR-CLIP model.
    """

    def __init__(self, clip_model: Any, tokenizer: Any, device: str = "cuda"):
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.device = device
        self.text_cache: dict[str, np.ndarray] = {}

    def get_text_embedding(self, prompt: str) -> np.ndarray:
        """Encode prompt text and cache normalized 512-dim embedding."""
        if prompt not in self.text_cache:
            tokens = self.tokenizer(prompt).to(self.device)
            with torch.no_grad():
                text_feat = self.clip_model.encode_text(tokens)
                text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-8)
            self.text_cache[prompt] = text_feat.cpu().numpy().squeeze(0)
        return self.text_cache[prompt]

    def predict_categorical(
        self,
        vol_feat: np.ndarray,
        candidate_prompts: dict[str, str],
    ) -> tuple[str, float]:
        """
        Zero-Shot CLIP classification across candidate prompts.
        candidate_prompts: {class_label: text_prompt_string}
        Returns: (predicted_class_label, max_cosine_similarity)
        """
        best_class = list(candidate_prompts.keys())[0]
        best_sim = -float("inf")

        for cls_name, prompt in candidate_prompts.items():
            t_feat = self.get_text_embedding(prompt)
            sim = float(np.dot(vol_feat, t_feat))
            if sim > best_sim:
                best_sim = sim
                best_class = cls_name

        return best_class, best_sim

    def predict_continuous(
        self,
        vol_feat: np.ndarray,
        prompt_fn: Callable[[float], str],
        values: list[float],
        temperature: float = 0.05,
    ) -> float:
        """
        Continuous Zero-Shot parameter estimation via Softmax Expected Value
        over candidate prompt embeddings.
        """
        sims = []
        for val in values:
            pr = prompt_fn(val)
            t_feat = self.get_text_embedding(pr)
            sims.append(float(np.dot(vol_feat, t_feat)))

        sims_arr = np.array(sims)
        # Numerical stability: subtract max before exp
        exp_sims = np.exp((sims_arr - np.max(sims_arr)) / temperature)
        weights = exp_sims / np.sum(exp_sims)
        expected_val = float(np.sum(weights * np.array(values)))
        return expected_val

    def predict_volume_continuous(
        self,
        vol_np: np.ndarray,
        prompt_fn: Callable[[float], str],
        values: list[float],
        temperature: float = 0.05,
    ) -> float:
        vol_feat = encode_volume_mrclip(vol_np, self.clip_model, self.device)
        return self.predict_continuous(vol_feat, prompt_fn, values, temperature)

    def predict_volume_categorical(
        self,
        vol_np: np.ndarray,
        candidate_prompts: dict[str, str],
    ) -> tuple[str, float]:
        vol_feat = encode_volume_mrclip(vol_np, self.clip_model, self.device)
        return self.predict_categorical(vol_feat, candidate_prompts)
