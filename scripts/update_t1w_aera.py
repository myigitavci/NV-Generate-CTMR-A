#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    ensure_axial_prompt,
    generate_sample,
    encode_prompt,
    normalize_for_display,
)
from scripts.utils import define_instance
from scripts.generate_10seeds_separate_figures import (
    ALL_SEEDS,
    extract_mid_axial_slice,
    plot_single_modality_10seeds,
)

# Siemens Aera real clinical T1W from Case 989
T1W_INFO = {
    "key": "T1W",
    "title": "T1-Weighted (T1w)",
    "badge_color": "#8c564b",
    "bg_badge": "#fdf2f0",
    "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5), Acquisition (Description, Sequence, Variant): (t1_se_tra_320, SE, SP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.014, 0.589, NONE, 90)",
    "lines": [
        "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5)",
        "Acquisition (Description, Sequence, Variant): (t1_se_tra_320, SE, SP)",
        "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.014, 0.589, NONE, 90)",
    ],
}

def main():
    device = torch.device("cuda:1")
    args = load_config(
        "configs/environment_maisi_diff_model_rflow-mr-brain.json",
        "configs/config_maisi_diff_model_rflow-mr-brain.json",
        "configs/config_network_rflow.json",
    )
    target_dim = tuple(args.diffusion_unet_inference.get("dim", [256, 256, 128]))

    print("Loading autoencoder...")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ae_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
    if "unet_state_dict" in ae_ckpt:
        ae_ckpt = ae_ckpt["unet_state_dict"]
    autoencoder.load_state_dict(ae_ckpt)
    autoencoder.eval()

    print("Loading UNet...")
    diff_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(diff_unet).to(device)
    unet_ckpt = torch.load("models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt", map_location=device, weights_only=False)
    state = unet_ckpt.get("model_state_dict", unet_ckpt.get("unet_state_dict", unet_ckpt.get("state_dict", unet_ckpt)))
    cleaned_state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    if any(k.startswith("clip_projection.") or k.startswith("unet.") for k in cleaned_state.keys()):
        model.load_state_dict(cleaned_state, strict=False)
    else:
        model.unet.load_state_dict(cleaned_state, strict=False)
    model.eval()
    scale_factor = float(unet_ckpt.get("scale_factor", 1.0422))

    print("Loading MR-CLIP...")
    clip_checkpoint = "/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt"
    open_clip_src = "/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src"
    clip_model, tokenizer = load_mrclip(open_clip_src, clip_checkpoint, 98, str(device))
    clip_model.eval()

    prompt = ensure_axial_prompt(T1W_INFO["prompt"])
    emb = encode_prompt(prompt, clip_model, tokenizer, device)

    cache_dir = Path("output/eval_memorization/diversity_cache")
    out_dir = Path("output/eval_memorization")
    slices = []

    for s in ALL_SEEDS:
        print(f"Generating T1W Seed {s}...")
        vol_np, _ = generate_sample(
            args, device, autoencoder, model, scale_factor, emb,
            target_dim, cfg_scale=4.0, seed=s
        )
        vol_norm = (vol_np / 1000.0).astype(np.float32)
        np.save(cache_dir / f"vol_T1W_real_seed{s}.npy", vol_norm)
        slc = extract_mid_axial_slice(vol_norm)
        np.save(cache_dir / f"slice_T1W_real_seed{s}.npy", slc)
        slices.append(slc)

    # Re-plot T1w figure with 10 seeds
    plot_single_modality_10seeds("t1w", T1W_INFO, slices, ALL_SEEDS, out_dir)
    print("T1W successfully updated!")

if __name__ == "__main__":
    main()
