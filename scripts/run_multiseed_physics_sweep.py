#!/usr/bin/env python3
import argparse
import json
import logging
import os
import pickle
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.ndimage import binary_erosion

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    ensure_axial_prompt,
    replace_imaging_param,
    encode_prompt,
)
from scripts.utils import define_instance
from monai.networks.schedulers import RFlowScheduler
from scripts.utils_infer import ReconModel
from scripts.utils import dynamic_infer
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

def extract_tissue_intensities(vol_np: np.ndarray, seg_np: np.ndarray | None = None) -> dict[str, float]:
    """Extract tissue median and mean intensities using robust percentile clustering / SynthSeg."""
    if seg_np is not None:
        raw_csf = np.isin(seg_np, [4, 43, 14, 15])
        core_ventricles = np.isin(seg_np, [4, 43])
        eroded_csf = binary_erosion(core_ventricles, iterations=1)
        csf_mask = eroded_csf if eroded_csf.sum() > 30 else raw_csf

        gm_mask = binary_erosion(np.isin(seg_np, [3, 42]), iterations=1)
        if gm_mask.sum() < 50:
            gm_mask = np.isin(seg_np, [3, 42])

        wm_mask = binary_erosion(np.isin(seg_np, [2, 41]), iterations=1)
        if wm_mask.sum() < 50:
            wm_mask = np.isin(seg_np, [2, 41])

        if csf_mask.sum() > 20 and gm_mask.sum() > 50 and wm_mask.sum() > 50:
            return {
                "csf": float(np.median(vol_np[csf_mask])),
                "gm": float(np.median(vol_np[gm_mask])),
                "wm": float(np.median(vol_np[wm_mask])),
                "brain_mean": float(np.mean(vol_np[csf_mask | gm_mask | wm_mask])),
            }

    v = vol_np.flatten()
    bg_thresh = max(10.0, float(np.percentile(v[v > 0], 10.0)) if (v > 0).any() else 10.0)
    brain_voxels = v[v > bg_thresh]
    return {
        "csf": float(np.percentile(brain_voxels, 15.0)),
        "gm": float(np.percentile(brain_voxels, 50.0)),
        "wm": float(np.percentile(brain_voxels, 85.0)),
        "brain_mean": float(np.mean(brain_voxels)),
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-seed 5-Brain Averaged Physics Sweep Evaluation")
    parser.add_argument("-e", "--env-config", default="configs/environment_maisi_diff_model_rflow-mr-brain.json")
    parser.add_argument("-c", "--model-config", default="configs/config_maisi_diff_model_rflow-mr-brain.json")
    parser.add_argument("-t", "--model-def", default="configs/config_network_rflow.json")
    parser.add_argument("--mrclip-ckpt", default="models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt")
    parser.add_argument("--clip-checkpoint", default="/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt")
    parser.add_argument("--open-clip-src", default="/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src")
    parser.add_argument("--output-file", default="output/eval_physics/physics_sweep_multiseed_5brains.pkl")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = parser.parse_args()

    out_file = Path(args_cli.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    seeds = [1001 + i * 100 for i in range(args_cli.num_seeds)]
    print(f"[INFO] Using {args_cli.num_seeds} seeds for ensemble averaging: {seeds}", flush=True)

    env_dict = json.load(open(args_cli.env_config))
    model_dict = json.load(open(args_cli.model_config))
    net_dict = json.load(open(args_cli.model_def))
    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)

    device = torch.device(args_cli.device)
    print(f"[INFO] Initializing models on {device}...", flush=True)

    # Autoencoder
    ae_path = env_dict.get("trained_autoencoder_path", "models/autoencoder_v1.pt")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt_ae = torch.load(ae_path, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ckpt_ae["state_dict"] if "state_dict" in ckpt_ae else ckpt_ae)
    autoencoder.eval()
    scale_factor = float(ckpt_ae.get("scale_factor", 1.0422))

    # UNet
    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(unet=base_unet).to(device)
    ckpt_diff = torch.load(args_cli.mrclip_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_diff["model_state_dict"] if "model_state_dict" in ckpt_diff else ckpt_diff)
    model.eval()

    # MR-CLIP
    clip_model, tokenizer = load_mrclip(
        checkpoint_path=args_cli.clip_checkpoint,
        open_clip_src=args_cli.open_clip_src,
        context_length=98,
        device=device,
    )
    clip_model.eval()

    target_dim = tuple(model_dict["diffusion_unet_inference"]["dim"])
    divisor = 2 ** (len(args.diffusion_unet_def["num_channels"]) - 2)
    latent_shape = (
        1,
        args.latent_channels,
        target_dim[0] // divisor,
        target_dim[1] // divisor,
        target_dim[2] // divisor,
    )

    spacing_raw = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2
    spacing_tensor = torch.from_numpy(spacing_raw[None, :]).half().to(device)

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    null_emb = model.get_null_embedding(1, device).to(torch.float16)

    @torch.no_grad()
    def generate_brain(prompt_str: str, seed: int) -> np.ndarray:
        set_determinism(seed=seed)
        noise_scheduler = define_instance(args, "noise_scheduler")
        num_steps = args.diffusion_unet_inference["num_inference_steps"]
        dummy_noise = torch.empty(latent_shape, device=device)
        if isinstance(noise_scheduler, RFlowScheduler):
            noise_scheduler.set_timesteps(
                num_inference_steps=num_steps,
                input_img_size_numel=torch.prod(torch.tensor(dummy_noise.shape[2:])),
            )
        else:
            noise_scheduler.set_timesteps(num_inference_steps=num_steps)

        cond_emb = encode_prompt(prompt_str, clip_model, tokenizer, device).to(device)
        noise = torch.randn(latent_shape, device=device)
        image = noise.clone()

        all_t = noise_scheduler.timesteps
        all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

        with torch.amp.autocast("cuda", enabled=True):
            for idx in range(len(all_t)):
                t = all_t[idx]
                next_t = all_next_t[idx]
                x2 = torch.cat([image, image], dim=0)
                t2 = torch.tensor([t, t], device=device)
                emb2 = torch.cat([cond_emb, null_emb], dim=0)
                sp2 = torch.cat([spacing_tensor, spacing_tensor], dim=0)

                out2 = model(x=x2, timesteps=t2, clip_embedding=emb2, spacing_tensor=sp2)
                model_cond, model_uncond = out2.chunk(2)
                model_output = model_uncond + 4.0 * (model_cond - model_uncond)

                if isinstance(noise_scheduler, RFlowScheduler):
                    image, _ = noise_scheduler.step(model_output, t, image, next_t)
                else:
                    image, _ = noise_scheduler.step(model_output, t, image)

            inferer = SlidingWindowInferer(
                roi_size=[80, 80, 80],
                sw_batch_size=1,
                progress=False,
                mode="gaussian",
                overlap=0.4,
                sw_device=device,
                device=device,
            )
            image = image.float()
            synthetic = dynamic_infer(inferer, recon_model, image)
            data = synthetic.squeeze().cpu().detach().numpy()
            data = np.clip(data * 1000.0, 0, 1000).astype(np.float32)
        return data

    base_t2_prompt = CANONICAL_MIDI_TEMPLATES["T2W"]
    base_t1_prompt = CANONICAL_MIDI_TEMPLATES["T1W"]

    te_vals = [10, 20, 30, 40, 60, 80, 100, 140]
    tr_vals = [600, 800, 1400, 2200, 3200, 4500, 6000, 8000]

    # Run TE Sweep
    print("\n=== Running 5-Seed TE Sweep ===", flush=True)
    te_results = []
    for te_ms in te_vals:
        prompt = ensure_axial_prompt(replace_imaging_param(base_t2_prompt, te=te_ms/1000.0, tr=4.5, ti=None))
        csf_list, gm_list, wm_list = [], [], []
        for s in seeds:
            vol = generate_brain(prompt, seed=s)
            tissues = extract_tissue_intensities(vol)
            csf_list.append(tissues["csf"])
            gm_list.append(tissues["gm"])
            wm_list.append(tissues["wm"])
        rec = {
            "param_ms": te_ms,
            "csf": float(np.mean(csf_list)),
            "csf_std": float(np.std(csf_list)),
            "gm": float(np.mean(gm_list)),
            "gm_std": float(np.std(gm_list)),
            "wm": float(np.mean(wm_list)),
            "wm_std": float(np.std(wm_list)),
        }
        te_results.append(rec)
        print(f"TE={te_ms}ms -> CSF={rec['csf']:.1f}±{rec['csf_std']:.1f}, GM={rec['gm']:.1f}±{rec['gm_std']:.1f}, WM={rec['wm']:.1f}±{rec['wm_std']:.1f}", flush=True)

    # Run TR Sweep
    print("\n=== Running 5-Seed TR Sweep ===", flush=True)
    tr_results = []
    for tr_ms in tr_vals:
        prompt = ensure_axial_prompt(replace_imaging_param(base_t1_prompt, te=0.012, tr=tr_ms/1000.0, ti=None))
        csf_list, gm_list, wm_list = [], [], []
        for s in seeds:
            vol = generate_brain(prompt, seed=s)
            tissues = extract_tissue_intensities(vol)
            csf_list.append(tissues["csf"])
            gm_list.append(tissues["gm"])
            wm_list.append(tissues["wm"])
        rec = {
            "param_ms": tr_ms,
            "csf": float(np.mean(csf_list)),
            "csf_std": float(np.std(csf_list)),
            "gm": float(np.mean(gm_list)),
            "gm_std": float(np.std(gm_list)),
            "wm": float(np.mean(wm_list)),
            "wm_std": float(np.std(wm_list)),
        }
        tr_results.append(rec)
        print(f"TR={tr_ms}ms -> CSF={rec['csf']:.1f}±{rec['csf_std']:.1f}, GM={rec['gm']:.1f}±{rec['gm_std']:.1f}, WM={rec['wm']:.1f}±{rec['wm_std']:.1f}", flush=True)

    output_data = {
        "te_records": te_results,
        "tr_records": tr_results,
        "seeds": seeds,
        "num_seeds": args_cli.num_seeds,
    }
    with open(out_file, "wb") as f:
        pickle.dump(output_data, f)
    print(f"\n[INFO] Successfully saved multi-seed results to: {out_file}", flush=True)

if __name__ == "__main__":
    main()
