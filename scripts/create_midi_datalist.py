"""
create_midi_datalist.py
========================
Creates a MAISI-compatible dataset JSON from the MIDI metadata CSV.
This JSON is needed both for:
  1. VAE encoding step (dataset_raw.json)
  2. MR-CLIP fine-tuning (dataset_mrclip.json, storing pkl idx + emb paths)

Usage:
  python -m scripts.create_midi_datalist \\
      --csv /data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv \\
      --embeddings-pkl data/mrclip_embeddings.pkl \\
      --data-base-dir /data/mr_niftis \\
      --embedding-base-dir ./embeddings \\
      --out-raw-json dataset_raw.json \\
      --out-mrclip-json dataset_mrclip.json \\
      --val-fraction 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import nibabel as nib
import pandas as pd

# ---------------------------------------------------------------------------
# Sequence → MAISI modality mapping
# ---------------------------------------------------------------------------
SEQUENCE_TO_MODALITY = {
    "T1W":     "mri_t1",
    "T2W":     "mri_t2",
    "FLAIR":   "mri_flair",
    "PDW":     "mri_pd",
    "T2STARW": "mri_t2",
}


def sequence_to_modality(seq: str) -> str:
    return SEQUENCE_TO_MODALITY.get(str(seq).upper().strip(), "mri")


# ---------------------------------------------------------------------------
# Filepath resolution
# ---------------------------------------------------------------------------

def resolve_path(raw: str, data_base_dir: str) -> str | None:
    """Try raw path, then prepend data_base_dir. Return None if not found."""
    if os.path.exists(raw):
        return raw
    alt = data_base_dir + raw
    if os.path.exists(alt):
        return alt
    alt2 = os.path.join(data_base_dir, raw.lstrip("/"))
    if os.path.exists(alt2):
        return alt2
    return None


def to_relative(abs_path: str, data_base_dir: str) -> str:
    """Strip data_base_dir prefix so path is relative to it."""
    if abs_path.startswith(data_base_dir):
        rel = abs_path[len(data_base_dir):]
        return rel.lstrip("/")
    return abs_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--embeddings-pkl", required=True,
                   help="mrclip_embeddings.pkl (matches CSV row order)")
    p.add_argument("--data-base-dir", default="/data/mr_niftis",
                   help="Root where NIfTI images live")
    p.add_argument("--embedding-base-dir", default="./embeddings",
                   help="Where VAE _emb.nii.gz files are stored")
    p.add_argument("--out-raw-json", default="dataset_raw.json",
                   help="Output JSON for VAE encoding step")
    p.add_argument("--out-mrclip-json", default="dataset_mrclip.json",
                   help="Output JSON for MR-CLIP fine-tuning")
    p.add_argument("--val-fraction", type=float, default=0.05)
    args = p.parse_args()

    random.seed(42)

    df = pd.read_csv(args.csv)
    print(f"CSV rows: {len(df)}")

    # Load pkl to confirm row count matches
    with open(args.embeddings_pkl, "rb") as f:
        pkl = pickle.load(f)
    assert len(pkl["embeddings"]) == len(df), (
        f"PKL has {len(pkl['embeddings'])} entries but CSV has {len(df)} rows."
    )
    print(f"PKL entries: {len(pkl['embeddings'])} ✓ (matches CSV)")

    raw_entries = []
    mrclip_entries = []
    missing_nifti = 0

    for pkl_idx, (_, row) in enumerate(df.iterrows()):
        abs_path = resolve_path(str(row["filepath"]), args.data_base_dir)
        if abs_path is None:
            missing_nifti += 1
            continue

        rel_path = to_relative(abs_path, args.data_base_dir)
        modality = sequence_to_modality(row.get("sequence", ""))

        # 1. Entry for VAE encoding
        raw_entries.append({"image": rel_path, "modality": modality})

        # 2. Derive embedding path relative to embedding_base_dir
        stem = rel_path
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        emb_rel_path = stem + "_emb.nii.gz"
        full_emb_path = os.path.join(args.embedding_base_dir, emb_rel_path)

        if os.path.exists(full_emb_path):
            # Also ensure sidecar .json with spacing exists for MONAI dataloader
            sidecar_json = full_emb_path + ".json"
            if not os.path.exists(sidecar_json):
                try:
                    img = nib.load(abs_path)
                    hdr = img.header
                    zooms = [float(z) for z in hdr.get_zooms()[:3]]
                except Exception:
                    zooms = [1.0, 1.0, 1.0]
                with open(sidecar_json, "w") as fj:
                    json.dump({"spacing": zooms, "modality": modality}, fj)

            # Store path relative to embedding_base_dir
            mrclip_entries.append({"image": emb_rel_path, "idx": pkl_idx})

    print(f"NIfTI files found: {len(raw_entries)}/{len(df)}, missing: {missing_nifti}")
    print(f"VAE embeddings already exist: {len(mrclip_entries)}/{len(raw_entries)}")

    # Write raw JSON for VAE
    idx_list = list(range(len(raw_entries)))
    random.shuffle(idx_list)
    n_val = max(1, int(len(raw_entries) * args.val_fraction))
    raw_json = {
        "training":   [raw_entries[i] for i in idx_list[n_val:]],
        "validation": [raw_entries[i] for i in idx_list[:n_val]],
    }
    with open(args.out_raw_json, "w") as f:
        json.dump(raw_json, f, indent=2)
    print(f"Saved {args.out_raw_json} (train={len(raw_json['training'])}, val={len(raw_json['validation'])})")

    # Write mrclip JSON for training
    if mrclip_entries:
        idx2 = list(range(len(mrclip_entries)))
        random.shuffle(idx2)
        n_val2 = max(1, int(len(mrclip_entries) * args.val_fraction))
        mrclip_json = {
            "training":   [mrclip_entries[i] for i in idx2[n_val2:]],
            "validation": [mrclip_entries[i] for i in idx2[:n_val2]],
        }
        with open(args.out_mrclip_json, "w") as f:
            json.dump(mrclip_json, f, indent=2)
        print(f"Saved {args.out_mrclip_json} (train={len(mrclip_json['training'])}, val={len(mrclip_json['validation'])})")
    else:
        print(f"Skipping {args.out_mrclip_json} — no _emb.nii.gz files found yet.")


if __name__ == "__main__":
    main()
