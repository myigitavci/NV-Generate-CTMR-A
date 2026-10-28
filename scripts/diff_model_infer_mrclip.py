"""
diff_model_infer_mrclip.py
===========================
Inference with the MR-CLIP conditioned diffusion UNet.

Usage:
  python -m scripts.diff_model_infer_mrclip \\
      -e configs/environment_maisi_diff_model_rflow-mr-brain.json \\
      -c configs/config_maisi_diff_model_rflow-mr-brain.json \\
      -t configs/config_network_rflow.json \\
      --mrclip-ckpt models/mrclip_phase2_best.pt \\
      --open-clip-src /home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src \\
      --clip-checkpoint /home/mav24/Projects/AMA-CLIP/logs/.../checkpoints/epoch_latest.pt \\
      --metadata-text "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, Prisma, 3.0), Acquisition (Description, Sequence, Variant): (T1w MPRAGE, SPGR, ND), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.00213, 1.9, 0.9, 9)" \\
      --cfg-scale 2.0 \\
      --output-dir output/

  # Or encode from a CSV row index:
  python -m scripts.diff_model_infer_mrclip ... --csv /data/distclip_training/....csv --csv-idx 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism
from tqdm import tqdm

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .mrclip_conditioned_unet import MRCLIPConditionedUNet
from .metadata_conditioning import MetadataEncoder, SequenceLabelEncoder
from .sample import ReconModel
from .utils import define_instance, dynamic_infer


# ---------------------------------------------------------------------------
# MR-CLIP text encoder (standalone, no image branch needed)
# ---------------------------------------------------------------------------

def load_clip_text_encoder(open_clip_src: str, checkpoint_path: str,
                            context_length: int, device: str):
    """Load ViT-B-16 MR-CLIP and return (model, tokenizer) — text-only use."""
    if open_clip_src not in sys.path:
        sys.path.insert(0, open_clip_src)
    from open_clip import create_model_and_transforms, get_tokenizer  # type: ignore

    model, _, _ = create_model_and_transforms(
        "ViT-B-16", device=device, textcontextlength=context_length, force_image_size=224
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = get_tokenizer("ViT-B-16", context_length=context_length)
    return model, tokenizer


@torch.no_grad()
def embed_text(text: str, clip_model, tokenizer, device: str) -> torch.Tensor:
    """Encode a single text string → (1, 512) normalised float tensor."""
    tokens = tokenizer([text]).to(device)
    feat = clip_model.encode_text(tokens)           # (1, 512)
    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    return feat.float()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models_mrclip(args, device, logger, mrclip_ckpt_path: str):
    """Load autoencoder + MRCLIPConditionedUNet from checkpoint."""
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ae_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
    if "unet_state_dict" in ae_ckpt:
        ae_ckpt = ae_ckpt["unet_state_dict"]
    autoencoder.load_state_dict(ae_ckpt)
    logger.info(f"Autoencoder loaded from {args.trained_autoencoder_path}")

    ckpt = torch.load(mrclip_ckpt_path, map_location=device, weights_only=False)
    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(
        unet=base_unet, clip_embed_dim=ckpt.get("conditioning_dim", 512),
    ).to(device)
    state = ckpt.get("model_state_dict", ckpt.get("unet_state_dict", ckpt))
    cleaned_state = {}
    for k, v in state.items():
        cleaned_k = k[len("module."):] if k.startswith("module.") else k
        cleaned_state[cleaned_k] = v

    has_wrapper_keys = any(k.startswith("clip_projection.") or k.startswith("unet.") for k in cleaned_state.keys())
    if has_wrapper_keys:
        missing, unexpected = model.load_state_dict(cleaned_state, strict=True)
    else:
        missing, unexpected = model.unet.load_state_dict(cleaned_state, strict=False)

    logger.info(f"MRCLIPConditionedUNet loaded from {mrclip_ckpt_path}")
    if missing:
        logger.warning(f"  Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"  Unexpected keys: {len(unexpected)}")

    scale_factor = ckpt.get("scale_factor", torch.tensor(1.0))
    if hasattr(scale_factor, "as_tensor"):
        scale_factor = scale_factor.as_tensor()
    logger.info(f"scale_factor = {scale_factor}")

    return autoencoder, model, scale_factor


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_mrclip(
    args,
    device,
    autoencoder,
    model: MRCLIPConditionedUNet,
    scale_factor,
    clip_embedding: torch.Tensor,   # (1, 512)
    output_size: tuple,
    cfg_scale: float,
    logger,
) -> np.ndarray:
    divisor = 2 ** (len(args.diffusion_unet_def["num_channels"]) - 2)
    latent_shape = (
        1, args.latent_channels,
        output_size[0] // divisor,
        output_size[1] // divisor,
        output_size[2] // divisor,
    )

    spacing_raw = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2
    spacing_tensor = torch.from_numpy(spacing_raw[None, :]).half().to(device)

    noise = torch.randn(latent_shape, device=device)
    image = noise

    noise_scheduler = define_instance(args, "noise_scheduler")
    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"]
        )

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    model.eval()

    null_emb = model.get_null_embedding(1, device).to(torch.float16 if args else torch.float32)
    cond_emb = clip_embedding.to(device)

    all_t = noise_scheduler.timesteps
    all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in tqdm(zip(all_t, all_next_t),
                               total=len(all_t), desc="Denoising"):
            if cfg_scale == 0.0:
                # Pure unconditional generation (no text conditioning)
                model_output = model(
                    x=image,
                    timesteps=torch.tensor([t], device=device),
                    clip_embedding=null_emb,
                    spacing_tensor=spacing_tensor,
                )
            elif cfg_scale == 1.0:
                # Standard conditional generation (single-pass)
                model_output = model(
                    x=image,
                    timesteps=torch.tensor([t], device=device),
                    clip_embedding=cond_emb,
                    spacing_tensor=spacing_tensor,
                )
            else:
                # Double-batch: [cond, uncond] -> uncond + s * (cond - uncond)
                x2 = torch.cat([image, image], dim=0)
                t2 = torch.tensor([t, t], device=device)
                emb2 = torch.cat([cond_emb, null_emb], dim=0)
                sp2 = torch.cat([spacing_tensor, spacing_tensor], dim=0)

                out2 = model(
                    x=x2, timesteps=t2,
                    clip_embedding=emb2, spacing_tensor=sp2,
                )
                model_cond, model_uncond = out2.chunk(2)
                model_output = model_uncond + cfg_scale * (model_cond - model_uncond)

            if isinstance(noise_scheduler, RFlowScheduler):
                image, _ = noise_scheduler.step(model_output, t, image, next_t)
            else:
                image, _ = noise_scheduler.step(model_output, t, image)

        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80], sw_batch_size=1,
            progress=True, mode="gaussian", overlap=0.4,
            sw_device=device, device=device,
        )
        image = image.float()
        synthetic = dynamic_infer(inferer, recon_model, image)
        data = synthetic.squeeze().cpu().detach().numpy()

        # MR intensity rescale [0,1] → [0,1000]
        a_min, a_max, b_min, b_max = 0, 1000, 0, 1
        data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        data = np.clip(data, a_min, None)
        return np.int16(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def diff_model_infer_mrclip(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    mrclip_ckpt: str,
    open_clip_src: str | None = None,
    clip_checkpoint: str | None = None,
    metadata_text: str | None = None,
    csv_path: str | None = None,
    csv_idx: int | None = None,
    cfg_scale: float = 2.0,
    output_dir: str = "./output",
    context_length: int = 98,
    random_seed: int | None = None,
    num_gpus: int = 1,
    sequence_label: str | None = None,
) -> str:
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("mrclip_infer")

    seed = random.randint(0, 99999) if random_seed is None else random_seed
    set_determinism(seed)
    logger.info(f"Random seed: {seed}")

    # Structured and sequence checkpoints carry their training preprocessing.
    checkpoint_info = torch.load(mrclip_ckpt, map_location="cpu", weights_only=False)
    conditioning = checkpoint_info.get("conditioning", "mrclip")
    metadata_schema = checkpoint_info.get("metadata_schema")
    sequence_schema = checkpoint_info.get("sequence_schema")
    del checkpoint_info

    if conditioning == "sequence":
        if sequence_label is None and csv_path and csv_idx is not None:
            import pandas as pd
            sequence_label = pd.read_csv(csv_path).iloc[csv_idx]["sequence"]
        if sequence_label is None:
            raise ValueError("Sequence inference requires --sequence-label, or --csv/--csv-idx with a 'sequence' column")
    elif sequence_label is not None:
        raise ValueError("--sequence-label is only used by sequence-conditioned checkpoints")
    elif metadata_text is None:
        if csv_path and csv_idx is not None:
            import pandas as pd
            df = pd.read_csv(csv_path)
            metadata_text = df.iloc[csv_idx]["text"]
            logger.info(f"Using text from CSV row {csv_idx}: {metadata_text[:80]}...")
        else:
            raise ValueError("Provide either --metadata-text or both --csv and --csv-idx")

    if conditioning == "sequence":
        if sequence_schema is None:
            raise ValueError("Sequence checkpoint is missing sequence_schema")
        encoder = SequenceLabelEncoder(sequence_schema)
        clip_emb = torch.from_numpy(encoder.transform([sequence_label])).to(device)
        logger.info(f"High-level sequence label: {sequence_label}")
    elif conditioning == "mlp":
        if metadata_schema is None:
            raise ValueError("MLP checkpoint is missing metadata_schema")
        encoder = MetadataEncoder(metadata_schema)
        clip_emb = torch.from_numpy(encoder.transform([metadata_text])).to(device)
    elif conditioning == "mrclip":
        if not open_clip_src or not clip_checkpoint:
            raise ValueError("MR-CLIP inference requires --open-clip-src and --clip-checkpoint")
        logger.info("Loading MR-CLIP text encoder...")
        clip_model, tokenizer = load_clip_text_encoder(
            open_clip_src, clip_checkpoint, context_length, device
        )
        clip_emb = embed_text(metadata_text, clip_model, tokenizer, device)
    else:
        raise ValueError(f"Unknown checkpoint conditioning: {conditioning}")
    logger.info(f"Encoded embedding: {clip_emb.shape}, norm={clip_emb.norm().item():.3f}")

    # Load diffusion model
    autoencoder, model, scale_factor = load_models_mrclip(args, device, logger, mrclip_ckpt)

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])

    # Encoder/model initialization differs between arms. Match sampling noise.
    set_determinism(seed)
    data = run_inference_mrclip(
        args, device, autoencoder, model, scale_factor,
        clip_emb, output_size, cfg_scale, logger,
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    out_path = (
        f"{output_dir}/{conditioning}_seed{seed}"
        f"_size{'x'.join(str(d) for d in output_size)}"
        f"_cfg{cfg_scale}_{ts}.nii.gz"
    )
    affine = np.diag([*out_spacing, 1.0])
    nib.save(nib.Nifti1Image(data, affine), out_path)
    logger.info(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MR-CLIP conditioned inference")
    p.add_argument("-e", "--env-config", required=True)
    p.add_argument("-c", "--model-config", required=True)
    p.add_argument("-t", "--model-def", required=True)
    p.add_argument("--mrclip-ckpt", required=True,
                   help="Path to fine-tuned MRCLIPConditionedUNet checkpoint")
    p.add_argument("--open-clip-src", default=None,
                   help="Path to open_clip src directory")
    p.add_argument("--clip-checkpoint", default=None,
                   help="Path to MR-CLIP epoch_latest.pt")
    p.add_argument("--metadata-text", type=str, default=None,
                   help="Free-text metadata string describing the desired MRI contrast")
    p.add_argument("--sequence-label", default=None,
                   help="Coarse label for sequence checkpoints: T1W, T2W, FLAIR, PDW, T2STARW")
    p.add_argument("--csv", type=str, default=None,
                   help="CSV path to pull text from by index")
    p.add_argument("--csv-idx", type=int, default=None,
                   help="Row index in CSV to use as metadata text")
    p.add_argument("--cfg-scale", type=float, default=2.0,
                   help="Classifier-free guidance scale (0=unconditional)")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--context-length", type=int, default=98)
    p.add_argument("-g", "--num-gpus", type=int, default=1)

    a = p.parse_args()
    diff_model_infer_mrclip(
        env_config_path=a.env_config,
        model_config_path=a.model_config,
        model_def_path=a.model_def,
        mrclip_ckpt=a.mrclip_ckpt,
        open_clip_src=a.open_clip_src,
        clip_checkpoint=a.clip_checkpoint,
        metadata_text=a.metadata_text,
        csv_path=a.csv,
        csv_idx=a.csv_idx,
        cfg_scale=a.cfg_scale,
        output_dir=a.output_dir,
        random_seed=a.seed,
        context_length=a.context_length,
        num_gpus=a.num_gpus,
        sequence_label=a.sequence_label,
    )
