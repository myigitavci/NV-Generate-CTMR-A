#!/usr/bin/env python3
import json
import logging
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

def main():
    logger = setup_logging("test_symphony_tr")
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

    # Reference volume
    ref_prompt = "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Avanto, 1.5), Acquisition (Description, Sequence, Variant): (pd+t2_tse_tra, SE, SK_SP_OSP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.086, 4.5, NONE, 150)"
    ref_vol, ref_latents = gen(ref_prompt)
    seg = run_synthseg(ref_vol, logger=logger)
    vent_mask = np.isin(seg, [4, 43, 14, 15])
    vent_counts = [int(np.sum(vent_mask[:, :, z])) for z in range(ref_vol.shape[2])]
    z_vent = int(np.argmax(vent_counts))
    logger.info(f"Ventricular plane locked at z={z_vent}")

    # Full 8-step TR sweep with Symphony (NONE, NONE, NONE)
    tr_sweep_ms = [600, 800, 1400, 2200, 3200, 4500, 6000, 8000]
    base_prompt = "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Symphony, 1.5), Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.012, 0.6, NONE, 90)"

    fig, axes = plt.subplots(1, 8, figsize=(20, 3), dpi=180)
    for idx, tr_ms in enumerate(tr_sweep_ms):
        p = replace_imaging_param(base_prompt, tr=tr_ms / 1000.0)
        vol, _ = gen(p, anchor=ref_latents[6], start_step=6)
        slc = vol[:, :, z_vent]
        p1, p99 = np.percentile(slc, 1), np.percentile(slc, 99)
        norm = np.clip((slc - p1) / max(p99 - p1, 1e-4), 0, 1)
        axes[idx].imshow(crop_brain_tight(np.rot90(norm)), cmap="gray")
        axes[idx].set_title(f"TR = {tr_ms} ms", fontsize=11, fontweight="bold")
        axes[idx].axis("off")
        tissues = extract_tissue_intensities(vol, seg_np=seg)
        logger.info(f"TR={tr_ms}ms -> WM={tissues['wm']:.1f}, GM={tissues['gm']:.1f}, CSF={tissues['csf']:.1f}")

    plt.tight_layout()
    out_png = Path("output/eval_physics/test_symphony_8tr_sweep.png")
    fig.savefig(str(out_png), bbox_inches="tight")
    logger.info(f"Saved to {out_png}")

if __name__ == "__main__":
    main()
