"""
embed_with_mrclip.py
====================
Generate per-scan MR-CLIP embeddings from the MIDI dataset using the
metadata text from the CSV.  Outputs:
  - data/mrclip_embeddings.pkl   : {'embeddings': ndarray(N,512,f32),
                                    'filepaths': [str,...],
                                    'texts': [str,...]}
  - dataset_mrclip.json          : MAISI-style data list for fine-tuning

Usage (with maisi conda env active, from repo root):
  python -m scripts.embed_with_mrclip \\
      --csv /data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv \\
      --data-root /data/mr_niftis \\
      --clip-checkpoint /home/mav24/Projects/AMA-CLIP/logs/<run>/checkpoints/epoch_latest.pt \\
      --open-clip-src /home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src \\
      --out-pkl data/mrclip_embeddings.pkl \\
      --out-json dataset_mrclip.json \\
      --batch-size 64 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MR-CLIP text embeddings for MIDI dataset")
    p.add_argument(
        "--csv", required=True,
        help="Path to metadata CSV (filepath, text, label, orientation, pair, sequence columns)",
    )
    p.add_argument(
        "--data-root", default="/data/mr_niftis",
        help="Root prefix under which the NIfTI files live on this machine.",
    )
    p.add_argument(
        "--clip-checkpoint", required=True,
        help="Path to the MR-CLIP epoch_latest.pt checkpoint",
    )
    p.add_argument(
        "--open-clip-src", required=True,
        help="Path to the open_clip src directory "
             "(e.g. /home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src)",
    )
    p.add_argument("--out-pkl", default="data/mrclip_embeddings.pkl",
                   help="Output pickle path")
    p.add_argument("--out-json", default="dataset_mrclip.json",
                   help="Output dataset JSON for MAISI training")
    p.add_argument("--embedding-base-dir", default="./embeddings",
                   help="Base dir where VAE latent embeddings (_emb.nii.gz) are stored")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Text encoding batch size")
    p.add_argument("--context-length", type=int, default=98,
                   help="Tokenizer context length (must match CLIP checkpoint)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="Fraction of scans to hold out for validation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# MR-CLIP loading  (mirrors MaRaI/src/dist_clip/modules/model_new.py)
# ---------------------------------------------------------------------------

def load_mrclip(
    open_clip_src: str,
    checkpoint_path: str,
    context_length: int,
    device: str,
):
    """Load a ViT-B-16 MR-CLIP model + tokenizer from a training checkpoint."""
    if open_clip_src not in sys.path:
        sys.path.insert(0, open_clip_src)

    from open_clip import create_model_and_transforms, get_tokenizer  # type: ignore

    model, _, _ = create_model_and_transforms(
        "ViT-B-16",
        device=device,
        textcontextlength=context_length,
        force_image_size=224,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    # strip 'module.' prefix if saved from DDP
    if next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    # filter mismatched shapes (e.g. 3-channel vs 1-channel conv)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    skipped = [k for k in sd if k not in filtered]
    if skipped:
        print(f"[CLIP] Skipping {len(skipped)} mismatched keys: {skipped[:5]}"
              f"{'...' if len(skipped) > 5 else ''}")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[CLIP] {len(missing)} keys not loaded from checkpoint")

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    tokenizer = get_tokenizer("ViT-B-16", context_length=context_length)
    print(f"[CLIP] Loaded from {checkpoint_path}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Embedding loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts(
    texts: list,
    model,
    tokenizer,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Returns (N, 512) float32 array of L2-normalised text embeddings."""
    all_embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
        batch = texts[i: i + batch_size]
        tokens = tokenizer(batch).to(device)
        feats = model.encode_text(tokens)                   # (B, 512)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        all_embeds.append(feats.float().cpu().numpy())
    return np.concatenate(all_embeds, axis=0)               # (N, 512) float32


# ---------------------------------------------------------------------------
# Filepath resolution
# ---------------------------------------------------------------------------

def resolve_filepath(raw_path: str, data_root: str) -> str:
    """
    CSV stores paths like /nfs/home/mavci/... which live under
    /data/mr_niftis on this machine.
    """
    if os.path.exists(raw_path):
        return raw_path
    # prepend data_root to the raw path
    candidate = data_root + raw_path          # /data/mr_niftis/nfs/home/...
    if os.path.exists(candidate):
        return candidate
    # strip leading /nfs and prepend data_root
    stripped = raw_path.lstrip("/")
    candidate2 = os.path.join(data_root, stripped)
    if os.path.exists(candidate2):
        return candidate2
    return raw_path   # return as-is; will be flagged later


# ---------------------------------------------------------------------------
# Dataset JSON builder
# ---------------------------------------------------------------------------

def build_dataset_json(
    valid_rows: pd.DataFrame,
    val_fraction: float,
) -> dict:
    """
    Build MAISI-style dataset JSON with per-scan index into the pkl.
      { "training": [{"image": "..._emb.nii.gz", "idx": int}, ...],
        "validation": [...] }
    The integer 'idx' is the row index in the pkl embeddings array.
    """
    random.seed(42)
    n_val = max(1, int(len(valid_rows) * val_fraction))
    all_idx = list(range(len(valid_rows)))
    random.shuffle(all_idx)
    val_idx = set(all_idx[:n_val])

    training, validation = [], []
    for i, row in valid_rows.iterrows():
        emb_path = row["emb_path"]
        entry = {"image": emb_path, "idx": int(row["pkl_idx"])}
        if i in val_idx:
            validation.append(entry)
        else:
            training.append(entry)

    return {"training": training, "validation": validation}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Reading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  Total rows: {len(df)}")

    # Resolve filepaths on this machine
    print("Resolving filepaths...")
    df["resolved_path"] = df["filepath"].apply(
        lambda p: resolve_filepath(str(p), args.data_root)
    )

    # Load MR-CLIP
    model, tokenizer = load_mrclip(
        args.open_clip_src,
        args.clip_checkpoint,
        args.context_length,
        args.device,
    )

    # Encode all texts
    texts = df["text"].tolist()
    print(f"Encoding {len(texts)} metadata strings with MR-CLIP...")
    embeddings = encode_texts(texts, model, tokenizer, args.device, args.batch_size)
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Save pkl  (row i of embeddings corresponds to row i of CSV)
    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving embeddings pkl to {out_pkl}...")
    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,                        # (N, 512) float32
                "filepaths": df["resolved_path"].tolist(),
                "texts": texts,
                "sequences": df["sequence"].tolist() if "sequence" in df.columns else [],
                "labels": df["label"].tolist() if "label" in df.columns else [],
            },
            f,
            protocol=4,
        )
    print(f"Saved {out_pkl}")

    # Build dataset JSON — only include rows with existing VAE _emb.nii.gz
    print("Building dataset JSON (filtering to existing VAE embeddings)...")
    valid_rows_list = []
    missing_emb = 0
    for pkl_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Checking emb")):
        raw = row["resolved_path"]
        # Derive _emb.nii.gz path
        stem = raw
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        emb_path = stem + "_emb.nii.gz"
        # Expand relative to embedding_base_dir if not absolute
        if not os.path.isabs(emb_path):
            emb_path = os.path.join(args.embedding_base_dir, emb_path)
        if os.path.exists(emb_path):
            valid_rows_list.append({**row.to_dict(), "emb_path": emb_path, "pkl_idx": pkl_idx})
        else:
            missing_emb += 1

    print(f"  Valid (emb found): {len(valid_rows_list)}, Missing emb: {missing_emb}")

    if valid_rows_list:
        valid_df = pd.DataFrame(valid_rows_list).reset_index(drop=True)
        dataset = build_dataset_json(valid_df, args.val_fraction)
        out_json = Path(args.out_json)
        with open(out_json, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved dataset JSON: {out_json}")
        print(f"  Training entries: {len(dataset['training'])}")
        print(f"  Validation entries: {len(dataset['validation'])}")
    else:
        print("WARNING: No VAE embedding files found — dataset JSON not written.")
        print("  Run the MAISI VAE encoding step first, or check --embedding-base-dir.")

    print("Done.")


if __name__ == "__main__":
    main()
