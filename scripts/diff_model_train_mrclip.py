"""
diff_model_train_mrclip.py
==========================
Fine-tune the MAISI brain-MRI diffusion UNet conditioned on MR-CLIP
text embeddings instead of discrete modality integers.

Two-phase training:
  Phase 1  --phase 1  Freeze UNet, only train the clip_projection MLP
  Phase 2  --phase 2  Unfreeze UNet, train jointly with separate LRs

Usage example:
  # Phase 1  (projection only, ~5 epochs)
  python -m scripts.diff_model_train_mrclip \\
      -e configs/environment_maisi_diff_model_rflow-mr-brain.json \\
      -c configs/config_maisi_diff_model_rflow-mr-brain.json \\
      -t configs/config_network_rflow.json \\
      --embeddings-pkl data/mrclip_embeddings.pkl \\
      --data-json dataset_mrclip.json \\
      --phase 1 --n-epochs 5 --lr 1e-3

  # Phase 2  (full fine-tune)
  python -m scripts.diff_model_train_mrclip \\
      -e configs/environment_maisi_diff_model_rflow-mr-brain.json \\
      -c configs/config_maisi_diff_model_rflow-mr-brain.json \\
      -t configs/config_network_rflow.json \\
      --embeddings-pkl data/mrclip_embeddings.pkl \\
      --data-json dataset_mrclip.json \\
      --phase 2 --n-epochs 100 --lr 1e-5 --proj-lr 1e-4 \\
      --existing-ckpt models/mrclip_phase1_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import monai
import torch
import torch.distributed as dist
from monai.data import DataLoader, partition_dataset
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.transforms import Compose
from monai.utils import first
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .mrclip_conditioned_unet import MRCLIPConditionedUNet
from .utils import define_instance


# ---------------------------------------------------------------------------
# PKL embedding store
# ---------------------------------------------------------------------------

class EmbeddingStore:
    """
    Lightweight wrapper around the pre-computed MR-CLIP embeddings pkl.
    Provides O(1) lookup by integer index.
    """

    def __init__(self, pkl_path: str) -> None:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.embeddings: np.ndarray = data["embeddings"].astype(np.float32)  # (N, 512)
        self.filepaths: list = data.get("filepaths", [])
        print(f"[EmbeddingStore] Loaded {len(self.embeddings)} embeddings from {pkl_path}")

    def __len__(self) -> int:
        return len(self.embeddings)

    def get(self, idx: int) -> torch.Tensor:
        """Return a (512,) float32 tensor."""
        return torch.from_numpy(self.embeddings[idx])


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

def load_filenames_and_indices(data_json_path: str) -> list:
    """
    Load dataset JSON.  Each entry has {"image": "..._emb.nii.gz", "idx": int}.
    Returns a list of dicts ready for MONAI CacheDataset.
    """
    with open(data_json_path) as f:
        js = json.load(f)
    return js.get("training", js)  # fall back to the whole list if no split key


def prepare_data_mrclip(
    train_files: list,
    emb_store: EmbeddingStore,
    embedding_base_dir: str = "./embeddings",
    cache_rate: float = 0.0,
    num_workers: int = 2,
    batch_size: int = 1,
) -> DataLoader:
    """Build a DataLoader that yields {image, clip_embedding, spacing}."""

    def _load_spacing(info_path: str | None) -> torch.Tensor:
        if info_path is not None and os.path.exists(info_path):
            with open(info_path) as f:
                d = json.load(f)
            return torch.FloatTensor(d.get("spacing", [1.0, 1.0, 1.0])) * 1e2
        return torch.FloatTensor([0.94, 0.94, 1.36]) * 1e2

    def _load_clip_embedding(idx: int) -> torch.Tensor:
        return emb_store.get(idx)   # (512,)

    transforms = Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Lambdad(
            keys="spacing",
            func=lambda x: _load_spacing(x),
        ),
        monai.transforms.Lambdad(
            keys="idx",
            func=lambda x: _load_clip_embedding(int(x)),
        ),
    ])

    # Build file-dicts: {image, spacing, idx}
    file_dicts = []
    for item in train_files:
        img_path = item["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(embedding_base_dir, img_path)
        info_path = img_path + ".json"
        if not os.path.exists(img_path):
            continue
        file_dicts.append({
            "image": img_path,
            "spacing": info_path if os.path.exists(info_path) else None,
            "idx": item["idx"],
        })

    ds = monai.data.CacheDataset(
        data=file_dicts,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    return DataLoader(ds, num_workers=4, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_wrapped_unet(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
    phase: int,
    existing_ckpt: str | None,
    clip_embed_dim: int = 512,
    clip_dropout: float = 0.1,
) -> MRCLIPConditionedUNet:
    """
    Load base UNet, wrap with MRCLIPConditionedUNet, optionally restore checkpoint.
    """
    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    base_unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_unet)

    model = MRCLIPConditionedUNet(
        unet=base_unet,
        clip_embed_dim=clip_embed_dim,
        dropout_p=clip_dropout,
    ).to(device)

    # Load weights if checkpoint provided
    if existing_ckpt:
        ckpt = torch.load(existing_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("unet_state_dict", ckpt))
        
        # Strip any DDP 'module.' prefix
        cleaned_state = {}
        for k, v in state.items():
            cleaned_k = k[len("module."):] if k.startswith("module.") else k
            cleaned_state[cleaned_k] = v

        # Check if this is a wrapped checkpoint (contains unet. or clip_projection.)
        has_wrapper_keys = any(k.startswith("clip_projection.") or k.startswith("unet.") for k in cleaned_state.keys())
        if has_wrapper_keys:
            missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        else:
            missing, unexpected = model.unet.load_state_dict(cleaned_state, strict=False)

        logger.info(f"Loaded checkpoint {existing_ckpt}")
        if missing:
            logger.info(f"  Missing keys: {len(missing)}")
        if unexpected:
            logger.info(f"  Unexpected keys: {len(unexpected)}")
    else:
        logger.info("No checkpoint provided — starting from base UNet initialized weights.")

    # Phase-specific freeze strategy
    if phase == 1:
        model.freeze_unet()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Phase 1: UNet frozen. Trainable params: {n_train:,}")
    else:
        model.unfreeze_unet()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Phase 2: Full fine-tune. Trainable params: {n_train:,}")

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    return model


# ---------------------------------------------------------------------------
# Slice extraction helper for MR-CLIP Image Encoder
# ---------------------------------------------------------------------------

def extract_triplane_slices(vol: torch.Tensor, in_channels: int = 3) -> torch.Tensor:
    """
    Extract 3 orthogonal central slices (axial, coronal, sagittal) from 3D MRI volume (B, 1, D, H, W)
    and resize each to (224, 224) for MR-CLIP vision encoder.
    """
    B, C, D, H, W = vol.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    # axial (B, 1, H, W), coronal (B, 1, D, W), sagittal (B, 1, D, H)
    axial = vol[:, :, mid_d, :, :]
    coronal = vol[:, :, :, mid_h, :]
    sagittal = vol[:, :, :, :, mid_w]

    axial_res = torch.nn.functional.interpolate(axial, size=(224, 224), mode="bilinear", align_corners=False)
    coronal_res = torch.nn.functional.interpolate(coronal, size=(224, 224), mode="bilinear", align_corners=False)
    sagittal_res = torch.nn.functional.interpolate(sagittal, size=(224, 224), mode="bilinear", align_corners=False)

    triplanes = torch.cat([axial_res, coronal_res, sagittal_res], dim=0)  # (3*B, 1, 224, 224)

    if in_channels == 3 and triplanes.shape[1] == 1:
        triplanes = triplanes.repeat(1, 3, 1, 1)

    return triplanes


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch_mrclip(
    epoch: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    loss_fn: torch.nn.Module,
    scaler: GradScaler,
    scale_factor: torch.Tensor,
    noise_scheduler,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
    amp: bool = True,
    tb_writer=None,
    csv_file=None,
    global_step_start: int = 0,
    use_mr_clip_loss: bool = False,
    mr_clip_loss_weight: float = 0.05,
    autoencoder: torch.nn.Module | None = None,
    clip_model: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, int]:
    """One training epoch with MR-CLIP embeddings as conditioning and optional consistency loss."""
    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)
    global_step = global_step_start

    # unwrap DDP for attribute access
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    include_body = raw_model.include_top_region_index_input

    model.train()
    for batch in train_loader:
        _iter += 1
        global_step += 1
        images = batch["image"].to(device) * scale_factor
        clip_emb = batch["idx"].to(device)           # (B, 512) — idx key was replaced by emb
        spacing = batch["spacing"].to(device)

        if include_body:
            top_idx = batch.get("top_region_index", torch.zeros(images.shape[0], 4)).to(device)
            bot_idx = batch.get("bottom_region_index", torch.zeros(images.shape[0], 4)).to(device)
        else:
            top_idx = bot_idx = None

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            noise = torch.randn_like(images)

            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(images)
            else:
                timesteps = torch.randint(
                    0, num_train_timesteps, (images.shape[0],), device=device
                ).long()

            noisy = noise_scheduler.add_noise(
                original_samples=images, noise=noise, timesteps=timesteps
            )

            model_out = model(
                x=noisy,
                timesteps=timesteps,
                clip_embedding=clip_emb,
                spacing_tensor=spacing,
                top_region_index_tensor=top_idx,
                bottom_region_index_tensor=bot_idx,
            )

            if isinstance(noise_scheduler, RFlowScheduler):
                target = images - noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                target = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                target = images
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                target = noise_scheduler.get_velocity(images, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction_type: {noise_scheduler.prediction_type}")

            loss_flow = loss_fn(model_out.float(), target.float())
            loss = loss_flow
            loss_clip_val = 0.0

            # Optional MR-CLIP Image-Text Consistency Loss
            if use_mr_clip_loss and autoencoder is not None and clip_model is not None:
                # Estimate clean latent x_1 from velocity: x_1 = x_t + (1 - t) * v
                if isinstance(noise_scheduler, RFlowScheduler):
                    t_expanded = timesteps.view(-1, 1, 1, 1, 1).to(images.dtype)
                    pred_x1 = noisy + (1.0 - t_expanded) * model_out
                else:
                    pred_x1 = model_out

                # Decode latent to 3D image volume
                pred_img = autoencoder.decode_stage_2_outputs(pred_x1 / scale_factor)
                pred_img = torch.clamp(pred_img, 0.0, 1.0)

                # Extract triplane slices and encode with MR-CLIP image encoder
                in_ch = 3
                if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "conv1"):
                    in_ch = getattr(clip_model.visual.conv1, "in_channels", 3)
                slices = extract_triplane_slices(pred_img, in_channels=in_ch)
                img_feats = clip_model.encode_image(slices)
                img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-8)

                # Average triplane features per volume -> (B, 512)
                B = images.shape[0]
                img_feats = img_feats.view(3, B, -1).mean(dim=0)
                img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-8)

                # Normalized target text embeddings
                tgt_feats = clip_emb / (clip_emb.norm(dim=-1, keepdim=True) + 1e-8)

                # Cosine similarity loss (1 - cos_sim)
                loss_clip = 1.0 - torch.sum(img_feats * tgt_feats, dim=-1).mean()
                loss_clip_val = loss_clip.item()
                loss = loss_flow + mr_clip_loss_weight * loss_clip

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        current_lr = optimizer.param_groups[0]["lr"]

        if local_rank == 0:
            if tb_writer is not None and (_iter % 5 == 0 or _iter == len(train_loader)):
                tb_writer.add_scalar("train/loss_total", loss.item(), global_step)
                tb_writer.add_scalar("train/loss_flow", loss_flow.item(), global_step)
                if use_mr_clip_loss:
                    tb_writer.add_scalar("train/loss_clip", loss_clip_val, global_step)
                tb_writer.add_scalar("train/lr", current_lr, global_step)
                tb_writer.flush()

            if csv_file is not None and (_iter % 5 == 0 or _iter == len(train_loader)):
                with open(csv_file, "a") as f_csv:
                    f_csv.write(f"{epoch+1},{_iter},{global_step},{loss.item():.6f},{loss_flow.item():.6f},{loss_clip_val:.6f},{current_lr:.8e}\n")

            if _iter % 10 == 0 or _iter == 1 or _iter == len(train_loader):
                msg = f"[{str(datetime.now())[:19]}] epoch {epoch+1}, iter {_iter}/{len(train_loader)}, loss: {loss.item():.4f}"
                if use_mr_clip_loss:
                    msg += f" (flow: {loss_flow.item():.4f}, clip: {loss_clip_val:.4f})"
                msg += f", lr: {current_lr:.2e}"
                logger.info(msg)

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=dist.ReduceOp.SUM)

    return loss_torch, global_step


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint_mrclip(
    epoch: int,
    model: torch.nn.Module,
    loss: float,
    scale_factor: torch.Tensor,
    ckpt_folder: str,
    filename: str,
    num_train_timesteps: int,
) -> None:
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    sf_val = float(scale_factor.item() if hasattr(scale_factor, "item") else scale_factor)
    os.makedirs(ckpt_folder, exist_ok=True)
    out_path = os.path.join(ckpt_folder, filename)
    torch.save(
        {
            "epoch": epoch + 1,
            "loss": float(loss),
            "scale_factor": torch.tensor(sf_val, dtype=torch.float32),
            "num_train_timesteps": num_train_timesteps,
            "model_state_dict": raw.state_dict(),
        },
        out_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def diff_model_train_mrclip(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    embeddings_pkl: str,
    data_json: str,
    phase: int = 1,
    n_epochs: int | None = None,
    lr: float | None = None,
    proj_lr: float | None = None,
    existing_ckpt: str | None = None,
    num_gpus: int = 1,
    amp: bool = True,
    clip_dropout: float = 0.1,
    ckpt_suffix: str = "",
    tb_dir: str = "./runs/mrclip_train",
    csv_log: str = "./logs/train_loss.csv",
    use_mr_clip_loss: bool = False,
    mr_clip_loss_weight: float = 0.05,
    clip_checkpoint: str | None = None,
    open_clip_src: str | None = None,
    clip_context_length: int = 98,
) -> None:
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("mrclip_train")

    # CLI overrides
    if n_epochs is not None:
        args.diffusion_unet_train["n_epochs"] = n_epochs
    if lr is not None:
        args.diffusion_unet_train["lr"] = lr

    # Set up TensorBoard and CSV logging (only on rank 0)
    tb_writer = None
    if local_rank == 0:
        logger.info(f"Phase {phase} | epochs={args.diffusion_unet_train['n_epochs']} | "
                    f"lr={args.diffusion_unet_train['lr']}")
        if use_mr_clip_loss:
            logger.info(f"MR-CLIP consistency loss enabled (weight={mr_clip_loss_weight})")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

        if tb_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_sub = os.path.join(tb_dir, f"phase{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                tb_writer = SummaryWriter(log_dir=tb_sub)
                logger.info(f"TensorBoard logging to: {tb_sub}")
            except ImportError:
                logger.warning("TensorBoard not available, skipping SummaryWriter.")

        if csv_log:
            Path(csv_log).parent.mkdir(parents=True, exist_ok=True)
            if not os.path.exists(csv_log):
                with open(csv_log, "w") as f_csv:
                    f_csv.write("epoch,iter,global_step,loss_total,loss_flow,loss_clip,lr\n")
            logger.info(f"CSV logging to: {csv_log}")

    # Load embeddings pkl
    emb_store = EmbeddingStore(embeddings_pkl)

    # Build DataLoader
    train_items = load_filenames_and_indices(data_json)
    if dist.is_initialized():
        train_items = partition_dataset(
            data=train_items, shuffle=True,
            num_partitions=dist.get_world_size(), even_divisible=True
        )[local_rank]

    train_loader = prepare_data_mrclip(
        train_items,
        emb_store,
        embedding_base_dir=getattr(args, "embedding_base_dir", "./embeddings"),
        cache_rate=args.diffusion_unet_train.get("cache_rate", 0.0),
        batch_size=args.diffusion_unet_train["batch_size"],
    )
    logger.info(f"Train batches: {len(train_loader)}")

    # Load model
    model = load_wrapped_unet(
        args, device, logger,
        phase=phase,
        existing_ckpt=existing_ckpt or getattr(args, "existing_ckpt_filepath", None),
        clip_dropout=clip_dropout,
    )
    noise_scheduler = define_instance(args, "noise_scheduler")

    # Load Autoencoder and MR-CLIP vision model if MR-CLIP loss is enabled
    autoencoder = None
    clip_model = None
    if use_mr_clip_loss:
        logger.info("Loading Autoencoder for MR-CLIP loss decoding...")
        autoencoder = define_instance(args, "autoencoder_def").to(device)
        ae_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
        if "unet_state_dict" in ae_ckpt:
            ae_ckpt = ae_ckpt["unet_state_dict"]
        autoencoder.load_state_dict(ae_ckpt)
        autoencoder.eval()
        for p in autoencoder.parameters():
            p.requires_grad_(False)

        if clip_checkpoint:
            from .embed_with_mrclip import load_mrclip
            src = open_clip_src or "/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src"
            logger.info(f"Loading MR-CLIP model from {clip_checkpoint}...")
            clip_model, _ = load_mrclip(src, clip_checkpoint, clip_context_length, str(device))
            clip_model.eval()
            for p in clip_model.parameters():
                p.requires_grad_(False)
        else:
            logger.warning("--use-mr-clip-loss enabled but --clip-checkpoint not provided!")

    # Scale factor
    check = first(train_loader)
    scale_factor = 1.0 / torch.std(check["image"].to(device))
    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=dist.ReduceOp.AVG)
    logger.info(f"scale_factor = {scale_factor:.4f}")

    # Optimizer — two-LR for phase 2
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    if phase == 2 and proj_lr is not None:
        param_groups = raw_model.trainable_parameter_groups(
            unet_lr=args.diffusion_unet_train["lr"],
            proj_lr=proj_lr,
        )
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.diffusion_unet_train["lr"],
        )

    total_steps = (
        args.diffusion_unet_train["n_epochs"]
        * len(train_loader.dataset)
        / args.diffusion_unet_train["batch_size"]
    )
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=int(total_steps), power=2.0
    )
    loss_fn = torch.nn.L1Loss()
    scaler = GradScaler("cuda")

    suffix = f"_phase{phase}{ckpt_suffix}"
    ckpt_name = args.model_filename.replace(".pt", f"{suffix}.pt")

    global_step = 0
    for epoch in range(args.diffusion_unet_train["n_epochs"]):
        loss_t, global_step = train_one_epoch_mrclip(
            epoch, model, train_loader, optimizer, lr_scheduler,
            loss_fn, scaler, scale_factor, noise_scheduler,
            args.noise_scheduler["num_train_timesteps"],
            device, logger, local_rank, amp=amp,
            tb_writer=tb_writer,
            csv_file=csv_log if local_rank == 0 else None,
            global_step_start=global_step,
            use_mr_clip_loss=use_mr_clip_loss,
            mr_clip_loss_weight=mr_clip_loss_weight,
            autoencoder=autoencoder,
            clip_model=clip_model,
        )
        loss_t = loss_t.tolist()
        if local_rank == 0:
            avg = loss_t[0] / loss_t[1]
            logger.info(f"Epoch {epoch+1} avg loss: {avg:.4f}")
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss_epoch", avg, epoch + 1)
            save_checkpoint_mrclip(
                epoch, model, avg, scale_factor,
                args.model_dir, ckpt_name,
                args.noise_scheduler["num_train_timesteps"],
            )

    if tb_writer is not None:
        tb_writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MR-CLIP conditioned diffusion model training")
    parser.add_argument("-e", "--env-config", required=True,
                        help="Environment config JSON")
    parser.add_argument("-c", "--model-config", required=True,
                        help="Model training config JSON")
    parser.add_argument("-t", "--model-def", required=True,
                        help="Network definition config JSON")
    parser.add_argument("--embeddings-pkl", required=True,
                        help="Path to mrclip_embeddings.pkl")
    parser.add_argument("--data-json", required=True,
                        help="Path to dataset_mrclip.json")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="Training phase: 1=projection only, 2=full fine-tune")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override UNet learning rate")
    parser.add_argument("--proj-lr", type=float, default=1e-3,
                        help="Learning rate for clip_projection (phase 2 only)")
    parser.add_argument("--existing-ckpt", type=str, default=None,
                        help="Path to existing checkpoint to resume from")
    parser.add_argument("-g", "--num-gpus", type=int, default=1)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--clip-dropout", type=float, default=0.1,
                        help="CFG embedding dropout during training")
    parser.add_argument("--ckpt-suffix", type=str, default="",
                        help="Extra suffix to append to checkpoint filename")
    parser.add_argument("--tb-dir", type=str, default="./runs/mrclip_train",
                        help="Directory for TensorBoard event logs (pass '' to disable)")
    parser.add_argument("--csv-log", type=str, default="./logs/train_loss.csv",
                        help="Path to CSV loss log file (pass '' to disable)")
    parser.add_argument("--mr-clip-loss", action="store_true",
                        help="Enable auxiliary MR-CLIP image-text cosine consistency loss")
    parser.add_argument("--mr-clip-loss-weight", type=float, default=0.05,
                        help="Weight lambda for the MR-CLIP consistency loss (default: 0.05)")
    parser.add_argument("--clip-checkpoint", type=str, default=None,
                        help="Path to MR-CLIP checkpoint for image-text consistency loss")
    parser.add_argument("--open-clip-src", type=str,
                        default="/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src",
                        help="Path to open_clip src directory")
    parser.add_argument("--clip-context-length", type=int, default=98,
                        help="Tokenizer context length for MR-CLIP")

    a = parser.parse_args()
    diff_model_train_mrclip(
        env_config_path=a.env_config,
        model_config_path=a.model_config,
        model_def_path=a.model_def,
        embeddings_pkl=a.embeddings_pkl,
        data_json=a.data_json,
        phase=a.phase,
        n_epochs=a.n_epochs,
        lr=a.lr,
        proj_lr=a.proj_lr,
        existing_ckpt=a.existing_ckpt,
        num_gpus=a.num_gpus,
        amp=a.amp,
        clip_dropout=a.clip_dropout,
        ckpt_suffix=a.ckpt_suffix,
        tb_dir=a.tb_dir,
        csv_log=a.csv_log,
        use_mr_clip_loss=a.mr_clip_loss,
        mr_clip_loss_weight=a.mr_clip_loss_weight,
        clip_checkpoint=a.clip_checkpoint,
        open_clip_src=a.open_clip_src,
        clip_context_length=a.clip_context_length,
    )
