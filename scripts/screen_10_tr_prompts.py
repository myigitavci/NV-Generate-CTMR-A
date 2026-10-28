#!/usr/bin/env python3
"""
scripts/screen_10_tr_prompts.py
===============================
Screens 10 diverse real and de-confounded prompts for TR parameter sweep starting from 200 ms.
Evaluates:
- Qualitative slice appearance (ventricle clarity, brain geometry, contrast)
- Quantitative tissue recovery curves (WM, GM, CSF)
- Spearman rank correlation of signal recovery (rho)
- Finds the ideal prompt that aligns with MRI physics expectations (T1 contrast at short TR, recovery at long TR).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch

from monai.inferers import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism

from scripts.diff_model_setting import load_config, setup_logging
from scripts.embed_with_mrclip import load_mrclip
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.run_mrclip_validation_experiments import encode_prompt, ensure_axial_prompt, replace_imaging_param
from scripts.utils import define_instance, dynamic_infer
from scripts.utils_infer import ReconModel
from scripts.run_real_prompt_publication_physics import crop_brain_tight
from scripts.eval_physics_fidelity import run_synthseg, extract_tissue_intensities


# 10 Diverse Candidate Prompts for TR Sweep
CANDIDATE_PROMPTS = [
    {
        "id": "P01_Avanto_empty",
        "label": "Avanto (NONE, NONE, NONE)",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90)",
    },
    {
        "id": "P02_Symphony_empty",
        "label": "Symphony (NONE, NONE, NONE)",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Symphony, 1.5), Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90)",
    },
    {
        "id": "P03_Avanto_SE_only",
        "label": "Avanto (NONE, SE, NONE)",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (NONE, SE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90)",
    },
    {
        "id": "P04_Symphony_SE_only",
        "label": "Symphony (NONE, SE, NONE)",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Symphony, 1.5), Acquisition (Description, Sequence, Variant): (NONE, SE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90)",
    },
    {
        "id": "P05_Symphony_t1_se",
        "label": "Symphony (t1_se_tra_3mm, SE, SP)",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Symphony, 1.5), Acquisition (Description, Sequence, Variant): (t1_se_tra_3mm, SE, SP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.008, {TR}, NONE, 90)",
    },
    {
        "id": "P06_Avanto_t1_fl2d",
        "label": "Avanto (t1_fl2d_tra, GR, NONE)",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (t1_fl2d_tra, GR, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.005, {TR}, NONE, 70)",
    },
    {
        "id": "P07_Aera_t1_tse",
        "label": "Aera (t1_tse_tra, SE, SK_SP_OSP)",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5), Acquisition (Description, Sequence, Variant): (t1_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 150)",
    },
    {
        "id": "P08_Aera_empty",
        "label": "Aera (NONE, NONE, NONE)",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5), Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90)",
    },
    {
        "id": "P09_GE_Signa_empty",
        "label": "Signa HDxt (NONE, NONE, NONE)",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90)",
    },
    {
        "id": "P10_Avanto_t1_se",
        "label": "Avanto (t1_se_tra, SE, NONE)",
        "prompt": "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (t1_se_tra, SE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90)",
    },
]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    out_dir = Path("output/eval_physics")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "screen_10_tr_prompts.log"

    logger = setup_logging("screen_10_tr_prompts")
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d][%(levelname)5s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    logger.info("Initializing 10-Prompt TR Sweep Physics Screening (starting from 200 ms)...")

    device = torch.device("cuda:0")
    env_config = "configs/environment_maisi_diff_model_rflow-mr-brain.json"
    model_config = "configs/config_maisi_diff_model_rflow-mr-brain.json"
    model_def = "configs/config_network_rflow.json"
    mrclip_ckpt = "models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt"
    clip_checkpoint = "/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt"
    open_clip_src = "/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src"

    env_dict = json.load(open(env_config))
    model_dict = json.load(open(model_config))
    args = load_config(env_config, model_config, model_def)

    ae_path = env_dict.get("trained_autoencoder_path", "models/autoencoder_v1.pt")
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt_ae = torch.load(ae_path, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ckpt_ae["state_dict"] if "state_dict" in ckpt_ae else ckpt_ae)
    autoencoder.eval()
    scale_factor = float(ckpt_ae.get("scale_factor", 1.0422))

    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(unet=base_unet).to(device)
    ckpt_diff = torch.load(mrclip_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_diff["model_state_dict"] if "model_state_dict" in ckpt_diff else ckpt_diff)
    model.eval()

    clip_model, tokenizer = load_mrclip(
        checkpoint_path=clip_checkpoint,
        open_clip_src=open_clip_src,
        context_length=98,
        device=device,
    )
    clip_model.eval()

    target_dim = tuple(model_dict["diffusion_unet_inference"]["dim"])
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    null_emb = model.get_null_embedding(1, device).to(torch.float16 if args else torch.float32)

    divisor = 2 ** (len(args.diffusion_unet_def["num_channels"]) - 2)
    latent_shape = (1, args.latent_channels, target_dim[0] // divisor, target_dim[1] // divisor, target_dim[2] // divisor)
    spacing_raw = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2
    spacing_tensor = torch.from_numpy(spacing_raw[None, :]).half().to(device)

    def gen(prompt_str, anchor=None, start_step=0):
        set_determinism(seed=6256)
        noise_scheduler = RFlowScheduler(num_train_timesteps=1000)
        noise_scheduler.set_timesteps(num_inference_steps=30)
        all_t = noise_scheduler.timesteps
        all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

        if anchor is None:
            image = torch.randn(latent_shape, device=device)
            saved = [image.detach().clone()]
            idx_start = 0
        else:
            image = anchor.clone()
            saved = None
            idx_start = start_step

        cond_emb = encode_prompt(ensure_axial_prompt(prompt_str), clip_model, tokenizer, device).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                for idx in range(idx_start, len(all_t)):
                    t = all_t[idx]
                    next_t = all_next_t[idx]
                    x2 = torch.cat([image, image], dim=0)
                    t2 = torch.tensor([t, t], device=device)
                    emb2 = torch.cat([cond_emb, null_emb], dim=0)
                    sp2 = torch.cat([spacing_tensor, spacing_tensor], dim=0)
                    out2 = model(x=x2, timesteps=t2, clip_embedding=emb2, spacing_tensor=sp2)
                    m_cond, m_uncond = out2.chunk(2)
                    m_out = m_uncond + 4.0 * (m_cond - m_uncond)
                    image, _ = noise_scheduler.step(m_out, t, image, next_t)
                    if saved is not None:
                        saved.append(image.detach().clone())

                inferer = SlidingWindowInferer(roi_size=[80, 80, 80], sw_batch_size=1, progress=False, mode="gaussian", overlap=0.4, sw_device=device, device=device)
                syn = dynamic_infer(inferer, recon_model, image.float())
                vol = syn.squeeze().cpu().detach().numpy()
                vol = np.clip((vol - 0) / (1 - 0) * 1000, 0, 1000).astype(np.float32)
        return vol, saved

    # 1. Generate High-Definition Anatomical Reference Volume (Siemens Avanto T2W)
    ref_cache_path = out_dir / "ref_brain_and_seg.pkl"
    if ref_cache_path.exists():
        logger.info(f"Loading cached anatomical reference brain and SynthSeg from {ref_cache_path}...")
        with open(ref_cache_path, "rb") as f:
            ref_data = pickle.load(f)
        ref_vol = ref_data["ref_vol"]
        ref_latents = ref_data["ref_latents"]
        seg = ref_data["seg"]
        z_vent = ref_data["z_vent"]
    else:
        ref_prompt = "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 4.5, NONE, 150)"
        logger.info("Synthesizing anatomical reference brain for spatial & ventricular alignment...")
        ref_vol, ref_latents = gen(ref_prompt)
        logger.info("Running SynthSeg on reference brain...")
        seg = run_synthseg(ref_vol, logger=logger)
        vent_mask = np.isin(seg, [4, 43, 14, 15]) if seg is not None else None
        if vent_mask is not None:
            vent_counts = [int(np.sum(vent_mask[:, :, z])) for z in range(ref_vol.shape[2])]
            z_vent = int(np.argmax(vent_counts))
        else:
            z_vent = 70
        with open(ref_cache_path, "wb") as f:
            pickle.dump({"ref_vol": ref_vol, "ref_latents": ref_latents, "seg": seg, "z_vent": z_vent}, f)
    logger.info(f"Locked ventricular plane at z={z_vent}")

    # Sweep TR values starting from 200 ms:
    tr_sweep_ms = [200, 400, 800, 1500, 3000, 6000]
    anchor_latent = ref_latents[6]

    results_all = {}
    for p_info in CANDIDATE_PROMPTS:
        pid = p_info["id"]
        plabel = p_info["label"]
        ptemplate = p_info["prompt"]
        logger.info(f"\nEvaluating Candidate [{pid}] - {plabel}...")

        slices = []
        records = []

        for tr_ms in tr_sweep_ms:
            tr_sec = tr_ms / 1000.0
            p = replace_imaging_param(ptemplate.format(TR=tr_sec), tr=tr_sec)
            vol, _ = gen(p, anchor=anchor_latent, start_step=6)
            slc = vol[:, :, z_vent]
            p1, p99 = np.percentile(slc, 1), np.percentile(slc, 99)
            norm = np.clip((slc - p1) / max(p99 - p1, 1e-4), 0, 1)
            slices.append((tr_ms, crop_brain_tight(np.rot90(norm))))

            tissues = extract_tissue_intensities(vol, seg_np=seg)
            records.append({"param_ms": tr_ms, **tissues})
            logger.info(f"  TR={tr_ms:4d}ms -> WM={tissues['wm']:.1f}, GM={tissues['gm']:.1f}, CSF={tissues['csf']:.1f}")

        tr_arr = np.array([r["param_ms"] for r in records])
        mean_arr = np.array([r["brain_mean"] for r in records])
        rho, _ = spearmanr(tr_arr, mean_arr)
        
        # Check T1 contrast at short TR (200ms): WM should ideally be brighter than CSF
        t1_contrast_ratio_200 = records[0]["wm"] / max(records[0]["csf"], 1e-3)
        # Check dynamic range: signal increase from 200ms to 6000ms
        dyn_range = records[-1]["brain_mean"] - records[0]["brain_mean"]

        results_all[pid] = {
            "label": plabel,
            "slices": slices,
            "records": records,
            "spearman_rho": float(rho),
            "wm_csf_ratio_200": float(t1_contrast_ratio_200),
            "dynamic_range": float(dyn_range),
            "prompt_template": ptemplate,
        }
        logger.info(f"  Summary: rho={rho:.3f}, WM/CSF(200ms)={t1_contrast_ratio_200:.2f}, DynRange={dyn_range:.1f}")

    # Save summary cache
    out_dir = Path("output/eval_physics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "screen_10_tr_prompts_results.pkl", "wb") as f:
        pickle.dump(results_all, f)

    # Render Comprehensive Screening Grid (10 rows, 1 curve plot + 6 slice images per row)
    fig = plt.figure(figsize=(22, 3.2 * len(CANDIDATE_PROMPTS)), dpi=180, facecolor="#ffffff")
    gs = fig.add_gridspec(len(CANDIDATE_PROMPTS), 7, width_ratios=[1.8, 1, 1, 1, 1, 1, 1])

    for row_idx, p_info in enumerate(CANDIDATE_PROMPTS):
        pid = p_info["id"]
        res = results_all[pid]
        recs = res["records"]
        tr_vals = [r["param_ms"] for r in recs]

        # 1. Curve plot
        ax_curve = fig.add_subplot(gs[row_idx, 0])
        ax_curve.plot(tr_vals, [r["csf"] for r in recs], "o-", color="#0284c7", lw=2, label="CSF")
        ax_curve.plot(tr_vals, [r["gm"] for r in recs], "s-", color="#16a34a", lw=2, label="Gray Matter")
        ax_curve.plot(tr_vals, [r["wm"] for r in recs], "^-", color="#ea580c", lw=2, label="White Matter")
        ax_curve.set_title(f"[{pid[:3]}] {res['label']}\nSpearman $\\rho$={res['spearman_rho']:.2f} | WM/CSF(200ms)={res['wm_csf_ratio_200']:.2f}", fontsize=10, fontweight="bold")
        ax_curve.set_xlabel("TR [ms]", fontsize=9)
        ax_curve.set_ylabel("Intensity", fontsize=9)
        ax_curve.set_ylim(200, 950)
        ax_curve.grid(True, linestyle="--", alpha=0.4)
        if row_idx == 0:
            ax_curve.legend(loc="lower right", fontsize=8)

        # 2. Six Slices
        for col_idx, (tr_ms, slc_img) in enumerate(res["slices"]):
            ax_img = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax_img.imshow(slc_img, cmap="gray")
            ax_img.set_title(f"{tr_ms} ms", fontsize=10)
            ax_img.axis("off")

    plt.tight_layout()
    out_png = out_dir / "screen_10_tr_prompts_overview.png"
    fig.savefig(str(out_png), bbox_inches="tight")
    logger.info(f"\nCompleted screening of 10 prompts! Full visualization saved to: {out_png}")

    # Rank candidates by physical alignment score:
    # Criteria: Positive Spearman rho (signal recovery) + high WM/CSF ratio at 200ms (T1 contrast)
    ranked = sorted(
        results_all.items(),
        key=lambda x: (x[1]["spearman_rho"] * 2.0 + x[1]["wm_csf_ratio_200"]),
        reverse=True,
    )
    print("\n" + "=" * 80)
    print("RANKED TR PROMPT CANDIDATES (Alignment with MRI Physics):")
    print("=" * 80)
    for rank, (pid, r) in enumerate(ranked, 1):
        print(f"Rank {rank}: [{pid}] {r['label']}")
        print(f"   -> Spearman Rho: {r['spearman_rho']:.3f}")
        print(f"   -> WM/CSF Ratio at 200ms: {r['wm_csf_ratio_200']:.2f}")
        print(f"   -> Dynamic Range: {r['dynamic_range']:.1f} a.u.")
        print("-" * 80)

    best_id, best_info = ranked[0]
    print(f"\nTop Candidate: {best_id} ({best_info['label']})")


if __name__ == "__main__":
    main()
