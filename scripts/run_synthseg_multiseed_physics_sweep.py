#!/usr/bin/env python3
import argparse
import json
import logging
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, zoom

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

def run_synthseg(vol_np: np.ndarray) -> np.ndarray | None:
    """Run FreeSurfer SynthSeg 2.0 on 3D volume."""
    synthseg_bin = "/usr/local/freesurfer/8.1.0/bin/mri_synthseg"
    if not Path(synthseg_bin).exists():
        synthseg_bin = "mri_synthseg"

    env = os.environ.copy()
    env["FREESURFER_HOME"] = "/usr/local/freesurfer/8.1.0"

    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = Path(tmp_dir) / "input.nii.gz"
        out_path = Path(tmp_dir) / "synthseg_out.nii.gz"

        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        nii = nib.Nifti1Image(vol_np.astype(np.float32), affine)
        nib.save(nii, str(in_path))

        cmd = [synthseg_bin, "--i", str(in_path), "--o", str(out_path), "--fast", "--keepgeom", "--threads", "4"]
        try:
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
            if res.returncode == 0 and out_path.exists():
                seg_nii = nib.load(str(out_path))
                seg_np = np.asanyarray(seg_nii.dataobj, dtype=np.int32)
                if seg_np.shape != vol_np.shape:
                    factors = [v / s for v, s in zip(vol_np.shape, seg_np.shape)]
                    seg_np = zoom(seg_np, factors, order=0).astype(np.int32)
                return seg_np
        except Exception as e:
            print(f"[WARN] SynthSeg execution failed: {e}", flush=True)

    return None

def extract_tissue_masks(seg_np: np.ndarray) -> dict[str, np.ndarray]:
    """Create binary tissue masks from FreeSurfer SynthSeg parcellation."""
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

    return {"csf": csf_mask, "gm": gm_mask, "wm": wm_mask}

def measure_tissues(vol_np: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Measure median intensity in each tissue mask."""
    return {
        "csf": float(np.median(vol_np[masks["csf"]])),
        "gm": float(np.median(vol_np[masks["gm"]])),
        "wm": float(np.median(vol_np[masks["wm"]])),
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-seed 5-Brain SynthSeg Evaluation")
    parser.add_argument("-e", "--env-config", default="configs/environment_maisi_diff_model_rflow-mr-brain.json")
    parser.add_argument("-c", "--model-config", default="configs/config_maisi_diff_model_rflow-mr-brain.json")
    parser.add_argument("-t", "--model-def", default="configs/config_network_rflow.json")
    parser.add_argument("--mrclip-ckpt", default="models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt")
    parser.add_argument("--clip-checkpoint", default="/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt")
    parser.add_argument("--open-clip-src", default="/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src")
    parser.add_argument("--output-file", default="output/eval_physics/physics_sweep_synthseg_multiseed.pkl")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = parser.parse_args()

    out_file = Path(args_cli.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    seeds = [1001 + i * 100 for i in range(args_cli.num_seeds)]
    print(f"[INFO] Using {args_cli.num_seeds} seeds for SynthSeg ensemble: {seeds}", flush=True)

    env_dict = json.load(open(args_cli.env_config))
    model_dict = json.load(open(args_cli.model_config))
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

    anchor_step = 8

    @torch.no_grad()
    def generate_locked(prompt_str: str, seed: int, anchor_latent: torch.Tensor | None = None) -> tuple[np.ndarray, torch.Tensor | None]:
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

        if anchor_latent is None:
            noise = torch.randn(latent_shape, device=device)
            image = noise.clone()
            start_idx = 0
            saved_anchor = None
        else:
            image = anchor_latent.clone()
            start_idx = anchor_step
            saved_anchor = None

        all_t = noise_scheduler.timesteps
        all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

        with torch.amp.autocast("cuda", enabled=True):
            for idx in range(start_idx, len(all_t)):
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

                if anchor_latent is None and idx == anchor_step - 1:
                    saved_anchor = image.detach().clone()

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
        return data, saved_anchor

    base_t2_prompt = CANONICAL_MIDI_TEMPLATES["T2W"]
    base_t1_prompt = CANONICAL_MIDI_TEMPLATES["T1W"]

    te_vals = [10, 20, 30, 40, 60, 80, 100, 140]
    tr_vals = [600, 800, 1400, 2200, 3200, 4500, 6000, 8000]

    # -----------------------------------------------------------------------
    # TE Sweep with SynthSeg
    # -----------------------------------------------------------------------
    print("\n=== Running Multi-Seed TE Sweep with SynthSeg 2.0 ===", flush=True)
    te_per_seed = {s: [] for s in seeds}

    for s in seeds:
        print(f"\n[Seed {s}] Generating T2 reference & running SynthSeg...", flush=True)
        ref_prompt = ensure_axial_prompt(replace_imaging_param(base_t2_prompt, te=0.090, tr=4.5, ti=None))
        ref_vol, anchor_latent = generate_locked(ref_prompt, seed=s)
        seg_np = run_synthseg(ref_vol)
        if seg_np is None:
            print(f"[WARN] SynthSeg failed for seed {s}, skipping...", flush=True)
            continue
        masks = extract_tissue_masks(seg_np)
        print(f"[Seed {s}] SynthSeg Masks: CSF={masks['csf'].sum()} voxels, GM={masks['gm'].sum()} voxels, WM={masks['wm'].sum()} voxels", flush=True)

        for te_ms in te_vals:
            prompt = ensure_axial_prompt(replace_imaging_param(base_t2_prompt, te=te_ms/1000.0, tr=4.5, ti=None))
            vol, _ = generate_locked(prompt, seed=s, anchor_latent=anchor_latent)
            tissues = measure_tissues(vol, masks)
            te_per_seed[s].append({"param_ms": te_ms, **tissues})
            print(f"  Seed {s} | TE={te_ms}ms -> CSF={tissues['csf']:.1f}, GM={tissues['gm']:.1f}, WM={tissues['wm']:.1f}", flush=True)

    # Average TE across seeds
    te_records = []
    for i, te_ms in enumerate(te_vals):
        csf_vals = [te_per_seed[s][i]["csf"] for s in seeds if len(te_per_seed[s]) > i]
        gm_vals = [te_per_seed[s][i]["gm"] for s in seeds if len(te_per_seed[s]) > i]
        wm_vals = [te_per_seed[s][i]["wm"] for s in seeds if len(te_per_seed[s]) > i]
        te_records.append({
            "param_ms": te_ms,
            "csf": float(np.mean(csf_vals)),
            "csf_std": float(np.std(csf_vals)),
            "gm": float(np.mean(gm_vals)),
            "gm_std": float(np.std(gm_vals)),
            "wm": float(np.mean(wm_vals)),
            "wm_std": float(np.std(wm_vals)),
        })
        print(f"TE Summary {te_ms}ms -> CSF={np.mean(csf_vals):.1f}±{np.std(csf_vals):.1f}, GM={np.mean(gm_vals):.1f}±{np.std(gm_vals):.1f}, WM={np.mean(wm_vals):.1f}±{np.std(wm_vals):.1f}", flush=True)

    # -----------------------------------------------------------------------
    # TR Sweep with SynthSeg
    # -----------------------------------------------------------------------
    print("\n=== Running Multi-Seed TR Sweep with SynthSeg 2.0 ===", flush=True)
    tr_per_seed = {s: [] for s in seeds}

    for s in seeds:
        print(f"\n[Seed {s}] Generating T1 reference & running SynthSeg...", flush=True)
        ref_prompt = ensure_axial_prompt(replace_imaging_param(base_t1_prompt, te=0.012, tr=2.2, ti=None))
        ref_vol, anchor_latent = generate_locked(ref_prompt, seed=s)
        seg_np = run_synthseg(ref_vol)
        if seg_np is None:
            print(f"[WARN] SynthSeg failed for seed {s}, skipping...", flush=True)
            continue
        masks = extract_tissue_masks(seg_np)
        print(f"[Seed {s}] SynthSeg Masks: CSF={masks['csf'].sum()} voxels, GM={masks['gm'].sum()} voxels, WM={masks['wm'].sum()} voxels", flush=True)

        for tr_ms in tr_vals:
            prompt = ensure_axial_prompt(replace_imaging_param(base_t1_prompt, te=0.012, tr=tr_ms/1000.0, ti=None))
            vol, _ = generate_locked(prompt, seed=s, anchor_latent=anchor_latent)
            tissues = measure_tissues(vol, masks)
            tr_per_seed[s].append({"param_ms": tr_ms, **tissues})
            print(f"  Seed {s} | TR={tr_ms}ms -> CSF={tissues['csf']:.1f}, GM={tissues['gm']:.1f}, WM={tissues['wm']:.1f}", flush=True)

    # Average TR across seeds
    tr_records = []
    for i, tr_ms in enumerate(tr_vals):
        csf_vals = [tr_per_seed[s][i]["csf"] for s in seeds if len(tr_per_seed[s]) > i]
        gm_vals = [tr_per_seed[s][i]["gm"] for s in seeds if len(tr_per_seed[s]) > i]
        wm_vals = [tr_per_seed[s][i]["wm"] for s in seeds if len(tr_per_seed[s]) > i]
        tr_records.append({
            "param_ms": tr_ms,
            "csf": float(np.mean(csf_vals)),
            "csf_std": float(np.std(csf_vals)),
            "gm": float(np.mean(gm_vals)),
            "gm_std": float(np.std(gm_vals)),
            "wm": float(np.mean(wm_vals)),
            "wm_std": float(np.std(wm_vals)),
        })
        print(f"TR Summary {tr_ms}ms -> CSF={np.mean(csf_vals):.1f}±{np.std(csf_vals):.1f}, GM={np.mean(gm_vals):.1f}±{np.std(gm_vals):.1f}, WM={np.mean(wm_vals):.1f}±{np.std(wm_vals):.1f}", flush=True)

    output_data = {
        "te_records": te_records,
        "tr_records": tr_records,
        "seeds": seeds,
        "num_seeds": args_cli.num_seeds,
    }
    with open(out_file, "wb") as f:
        pickle.dump(output_data, f)
    print(f"\n[INFO] Successfully saved SynthSeg multi-seed results to: {out_file}", flush=True)

if __name__ == "__main__":
    main()
