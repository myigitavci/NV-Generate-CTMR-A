#!/usr/bin/env python3
"""
scripts/screen_canonical_and_10_tr_prompts.py
=============================================
Screens the Canonical T1W prompt + 10 diverse candidate prompts for the TR parameter sweep
starting from 200 ms.

Key design:
- Reference brain is synthesized from the Canonical T1W prompt at TR=2.2s, locking anatomical
  geometry with natural T1 contrast (dark ventricles).
- Anchor latent is extracted at step 8.
- Evaluates TR sweep across [200, 500, 1000, 2000, 4000, 8000] ms.
- Evaluates:
    * Contrast order at 200 ms (expected: WM > GM > CSF)
    * Spearman rank correlation rho (expected: positive recovery)
    * CSF dynamic range (expected: substantial brightening from 200ms to 8000ms)
    * Visual ventricle clarity and anatomical symmetry
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
from scripts.run_mrclip_validation_experiments import (
    CANONICAL_MIDI_TEMPLATES,
    encode_prompt,
    ensure_axial_prompt,
    replace_imaging_param,
)
from scripts.utils import define_instance, dynamic_infer
from scripts.utils_infer import ReconModel
from scripts.run_real_prompt_publication_physics import crop_brain_tight
from scripts.eval_physics_fidelity import run_synthseg, extract_tissue_intensities


PROMPTS_TO_SCREEN = [
    {
        "id": "C00_Canonical_Vida_TRA_SE_SK",
        "label": "Canonical Vida (TRA, SE, SK)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Siemens, MAGNETOM_Vida, 3.0), "
            "Acquisition (Description, Sequence, Variant): (TRA, SE, SK), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90.0)"
        ),
    },
    {
        "id": "C01_Canonical_Vida_empty",
        "label": "Canonical Vida (NONE, NONE, NONE)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Siemens, MAGNETOM_Vida, 3.0), "
            "Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90.0)"
        ),
    },
    {
        "id": "C02_Symphony_t1_se",
        "label": "Symphony (t1_se_tra_3mm, SE, SP)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Siemens, Symphony, 1.5), "
            "Acquisition (Description, Sequence, Variant): (t1_se_tra_3mm, SE, SP), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.008, {TR}, NONE, 90.0)"
        ),
    },
    {
        "id": "C03_Symphony_empty",
        "label": "Symphony (NONE, NONE, NONE)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Siemens, Symphony, 1.5), "
            "Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.008, {TR}, NONE, 90.0)"
        ),
    },
    {
        "id": "C04_Avanto_t1_se",
        "label": "Avanto (t1_se_tra, SE, NONE)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), "
            "Acquisition (Description, Sequence, Variant): (t1_se_tra, SE, NONE), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90.0)"
        ),
    },
    {
        "id": "C05_Avanto_empty",
        "label": "Avanto (NONE, NONE, NONE)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), "
            "Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90.0)"
        ),
    },
    {
        "id": "C06_Aera_t1_tse",
        "label": "Aera (t1_tse_tra, SE, SK_SP_OSP)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Siemens, Aera, 1.5), "
            "Acquisition (Description, Sequence, Variant): (t1_tse_tra, SE, SK_SP_OSP), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 150.0)"
        ),
    },
    {
        "id": "C07_GE_Signa_Ax_T1",
        "label": "Signa HDxt (Ax T1, SE, NONE)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), "
            "Acquisition (Description, Sequence, Variant): (Ax T1, SE, NONE), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.02, {TR}, NONE, 90.0)"
        ),
    },
    {
        "id": "C08_GE_Signa_empty",
        "label": "Signa HDxt (NONE, NONE, NONE)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), "
            "Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.02, {TR}, NONE, 90.0)"
        ),
    },
    {
        "id": "C09_Philips_Achieva_T1",
        "label": "Achieva (T1_SE_TRA, SE, NONE)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Philips, Achieva_dStream, 3.0), "
            "Acquisition (Description, Sequence, Variant): (T1_SE_TRA, SE, NONE), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.015, {TR}, NONE, 90.0)"
        ),
    },
    {
        "id": "C10_Siemens_TrioTim_t1",
        "label": "TrioTim (t1_se_tra, SE, NONE)",
        "prompt": (
            "A brain MRI, plane axial, "
            "Scanner (Manufacturer, Model, Field Strength): (Siemens, TrioTim, 3.0), "
            "Acquisition (Description, Sequence, Variant): (t1_se_tra, SE, NONE), "
            "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, {TR}, NONE, 90.0)"
        ),
    },
]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    out_dir = Path("output/eval_physics")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "screen_canonical_and_10_tr_prompts.log"

    logger = setup_logging("screen_canonical_10_tr")
    fh = logging.FileHandler(str(log_file), mode="w")
    fh.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d][%(levelname)5s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    logger.info("Initializing Screening of Canonical T1W Prompt + 10 Prompts (Starting TR from 200 ms)...")

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
    null_emb = model.get_null_embedding(1, device).to(torch.float16)

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

    # 1. Synthesize Anatomical Reference Volume using the CANONICAL T1W Prompt at TR=2.2s
    canonical_t1_ref_prompt = (
        "A brain MRI, plane axial, "
        "Scanner (Manufacturer, Model, Field Strength): (Siemens, MAGNETOM_Vida, 3.0), "
        "Acquisition (Description, Sequence, Variant): (TRA, SE, SK), "
        "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, 2.2, NONE, 90.0)"
    )
    t1_cache_file = out_dir / "canonical_t1_ref_brain_and_seg.pkl"
    if t1_cache_file.exists():
        logger.info(f"Loading cached Canonical T1W reference brain & SynthSeg from {t1_cache_file}...")
        with open(t1_cache_file, "rb") as f:
            t1_cached = pickle.load(f)
        ref_vol = t1_cached["ref_vol"]
        ref_latents = t1_cached["ref_latents"]
        seg = t1_cached["seg"]
        z_vent = t1_cached["z_vent"]
    else:
        logger.info(f"Synthesizing Canonical T1W Reference Brain: {canonical_t1_ref_prompt}...")
        ref_vol, ref_latents = gen(canonical_t1_ref_prompt)
        logger.info("Running SynthSeg on Canonical T1W reference brain...")
        seg = run_synthseg(ref_vol, logger=logger)
        vent_mask = np.isin(seg, [4, 43, 14, 15]) if seg is not None else None
        if vent_mask is not None:
            vent_counts = [int(np.sum(vent_mask[:, :, z])) for z in range(ref_vol.shape[2])]
            z_vent = int(np.argmax(vent_counts))
        else:
            z_vent = 70
        with open(t1_cache_file, "wb") as f:
            pickle.dump({"ref_vol": ref_vol, "ref_latents": ref_latents, "seg": seg, "z_vent": z_vent}, f)
    logger.info(f"Locked ventricular plane at z={z_vent}")

    # Anchor at reverse step 8 (as in original canonical figure)
    anchor_step = 8
    anchor_latent = ref_latents[anchor_step]

    # Swept TR values starting from 200 ms:
    tr_sweep_ms = [200, 500, 1000, 2000, 4000, 8000]

    results_all = {}
    for p_info in PROMPTS_TO_SCREEN:
        pid = p_info["id"]
        plabel = p_info["label"]
        ptemplate = p_info["prompt"]
        logger.info(f"\nEvaluating Candidate [{pid}] - {plabel}...")

        slices = []
        records = []

        for tr_ms in tr_sweep_ms:
            tr_sec = tr_ms / 1000.0
            p = replace_imaging_param(ptemplate.format(TR=tr_sec), tr=tr_sec)
            vol, _ = gen(p, anchor=anchor_latent, start_step=anchor_step)
            slc = vol[:, :, z_vent]
            p1, p99 = np.percentile(slc, 1), np.percentile(slc, 99)
            norm = np.clip((slc - p1) / max(p99 - p1, 1e-4), 0, 1)
            slices.append((tr_ms, crop_brain_tight(np.rot90(norm))))

            tissues = extract_tissue_intensities(vol, seg_np=seg)
            records.append({"param_ms": tr_ms, **tissues})
            logger.info(f"  TR={tr_ms:4d}ms -> WM={tissues['wm']:.1f}, GM={tissues['gm']:.1f}, CSF={tissues['csf']:.1f}")

        tr_arr = np.array([r["param_ms"] for r in records])
        mean_arr = np.array([r["brain_mean"] for r in records])
        csf_arr = np.array([r["csf"] for r in records])
        rho_brain, _ = spearmanr(tr_arr, mean_arr)
        rho_csf, _ = spearmanr(tr_arr, csf_arr)

        # Contrast order at 200 ms:
        # Correct T1 expectation: WM > GM > CSF
        r200 = records[0]
        t1_order_correct = (r200["wm"] > r200["gm"]) and (r200["gm"] > r200["csf"])
        wm_csf_ratio_200 = r200["wm"] / max(r200["csf"], 1e-3)
        csf_dynamic_range = records[-1]["csf"] - records[0]["csf"]

        results_all[pid] = {
            "label": plabel,
            "slices": slices,
            "records": records,
            "spearman_rho_brain": float(rho_brain),
            "spearman_rho_csf": float(rho_csf),
            "t1_order_correct": bool(t1_order_correct),
            "wm_csf_ratio_200": float(wm_csf_ratio_200),
            "csf_dynamic_range": float(csf_dynamic_range),
            "prompt_template": ptemplate,
        }
        logger.info(
            f"  Summary: rho_brain={rho_brain:.3f}, rho_csf={rho_csf:.3f}, "
            f"WM/CSF(200ms)={wm_csf_ratio_200:.2f}, CSF_DynRange={csf_dynamic_range:.1f}, "
            f"T1_Order_Correct={t1_order_correct}"
        )

    # Save summary cache
    with open(out_dir / "screen_canonical_and_10_tr_results.pkl", "wb") as f:
        pickle.dump(results_all, f)

    # Render Comprehensive Screening Grid (11 rows, 1 curve plot + 6 slice images per row)
    fig = plt.figure(figsize=(22, 3.2 * len(PROMPTS_TO_SCREEN)), dpi=180, facecolor="#ffffff")
    gs = fig.add_gridspec(len(PROMPTS_TO_SCREEN), 7, width_ratios=[1.8, 1, 1, 1, 1, 1, 1])

    for row_idx, p_info in enumerate(PROMPTS_TO_SCREEN):
        pid = p_info["id"]
        res = results_all[pid]
        recs = res["records"]
        tr_vals = [r["param_ms"] for r in recs]

        # 1. Curve plot
        ax_curve = fig.add_subplot(gs[row_idx, 0])
        ax_curve.plot(tr_vals, [r["csf"] for r in recs], "o-", color="#0284c7", lw=2, label="CSF")
        ax_curve.plot(tr_vals, [r["gm"] for r in recs], "s-", color="#16a34a", lw=2, label="Gray Matter")
        ax_curve.plot(tr_vals, [r["wm"] for r in recs], "^-", color="#ea580c", lw=2, label="White Matter")
        title_str = (
            f"[{pid[:3]}] {res['label']}\n"
            f"$\\rho_{{CSF}}$={res['spearman_rho_csf']:.2f} | WM/CSF(200ms)={res['wm_csf_ratio_200']:.2f} | "
            f"CSF $\\Delta$={res['csf_dynamic_range']:.0f}"
        )
        ax_curve.set_title(title_str, fontsize=9.5, fontweight="bold")
        ax_curve.set_xlabel("TR [ms]", fontsize=9)
        ax_curve.set_ylabel("Intensity", fontsize=9)
        ax_curve.set_ylim(150, 950)
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
    out_png = out_dir / "screen_canonical_10_tr_overview.png"
    fig.savefig(str(out_png), bbox_inches="tight")
    logger.info(f"\nCompleted screening of Canonical + 10 prompts! Full visualization saved to: {out_png}")

    # Rank candidates by physical alignment score:
    # Criteria:
    # 1. Correct T1 contrast at 200 ms (WM > CSF)
    # 2. High CSF recovery correlation rho
    # 3. High CSF dynamic range (recovery from dark to bright)
    def rank_score(item):
        d = item[1]
        score = (
            (2.0 if d["t1_order_correct"] else 0.0)
            + d["wm_csf_ratio_200"] * 1.5
            + d["spearman_rho_csf"] * 2.0
            + (d["csf_dynamic_range"] / 100.0)
        )
        return score

    ranked = sorted(results_all.items(), key=rank_score, reverse=True)
    print("\n" + "=" * 80)
    print("RANKED TR PROMPT CANDIDATES (Alignment with MRI Physics):")
    print("=" * 80)
    for rank, (pid, r) in enumerate(ranked, 1):
        print(f"Rank {rank}: [{pid}] {r['label']}")
        print(f"   -> T1 Order at 200ms (WM>GM>CSF): {'YES' if r['t1_order_correct'] else 'NO'}")
        print(f"   -> WM/CSF Ratio at 200ms: {r['wm_csf_ratio_200']:.2f}")
        print(f"   -> CSF Spearman Rho: {r['spearman_rho_csf']:.3f}")
        print(f"   -> CSF Dynamic Range: {r['csf_dynamic_range']:.1f} a.u.")
        print("-" * 80)

    best_id, best_info = ranked[0]
    print(f"\nTop Candidate: {best_id} ({best_info['label']})")


if __name__ == "__main__":
    main()
