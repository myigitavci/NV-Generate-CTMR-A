import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from scripts.embed_with_mrclip import load_mrclip
from scripts.evaluate_mrrate_systematic import load_and_preprocess_real, encode_volume_with_clip, encode_prompt

def resolve_filepath(raw_path: str, data_root: str | None = None) -> str | None:
    if not raw_path or str(raw_path).lower() in ["nan", "none"]:
        return None
    if os.path.exists(raw_path):
        return raw_path
    if data_root:
        c1 = os.path.join(data_root, raw_path.lstrip("/"))
        if os.path.exists(c1):
            return c1
        c2 = os.path.join(data_root, os.path.basename(raw_path))
        if os.path.exists(c2):
            return c2
    common = ["/data/mr_niftis", "/data/mrrate"]
    for cr in common:
        c = os.path.join(cr, os.path.basename(raw_path))
        if os.path.exists(c):
            return c
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Empirical Diagnostic for MIDI CLIP model with native MIDI prompts")
    p.add_argument(
        "--clip-checkpoint",
        default="/home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt",
        help="Path to CLIP checkpoint",
    )
    p.add_argument(
        "--open-clip-src",
        default="/home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src",
        help="Path to open_clip src directory",
    )
    p.add_argument(
        "--csv",
        default="/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv",
        help="Path to MIDI CSV",
    )
    p.add_argument("--data-root", default="/data/mr_niftis")
    p.add_argument("--context-length", type=int, default=98)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clip_ckpt = args.clip_checkpoint

    print("=" * 75)
    print("EMPIRICAL DIAGNOSTIC: MIDI CLIP MODEL WITH NATIVE MIDI PROMPTS")
    print("=" * 75)
    print(f"Device: {device}")
    print(f"Checkpoint: {clip_ckpt}")
    print(f"CSV: {args.csv}")

    clip_model, tokenizer = load_mrclip(args.open_clip_src, clip_ckpt, args.context_length, str(device))
    clip_model.eval()

    # Load MIDI CSV
    target_dim = (256, 256, 128)
    test_cases = []
    
    if os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        print(f"\nLoaded MIDI CSV with {len(df)} rows. Columns: {list(df.columns)}")
        
        # Find 2 valid cases with existing niftis on disk
        for idx, row in df.iterrows():
            fp = row.get("filepath", None)
            res = resolve_filepath(str(fp), args.data_root)
            if res and os.path.exists(res):
                prompt = str(row.get("text", ""))
                seq = str(row.get("sequence", row.get("label", "MRI")))
                test_cases.append({"idx": idx, "path": res, "prompt": prompt, "seq": seq})
                if len(test_cases) >= 2:
                    break
    else:
        print(f"\nMIDI CSV not found at {args.csv}, falling back to default cohort CSV")

    # Fallback if no nifti resolved
    if len(test_cases) == 0:
        print("Could not resolve nifti from MIDI CSV, using sample MIDI prompts:")
        prompt_1 = "A brain MRI, axial plane, T1-weighted MPRAGE sequence showing cerebral cortex, ventricles and white matter."
        prompt_2 = "A brain MRI, axial plane, T2-weighted sequence showing hyperintense CSF and clear basal ganglia."
        real_scan_1 = None
        real_scan_2 = None
    else:
        print(f"\nResolved {len(test_cases)} native MIDI cases:")
        for i, tc in enumerate(test_cases):
            print(f"  Case {i+1} (Row {tc['idx']}, {tc['seq']}): {tc['path']}")
            print(f"  Prompt {i+1}: {tc['prompt'][:100]}...")
        prompt_1 = test_cases[0]["prompt"]
        prompt_2 = test_cases[1]["prompt"]
        _, real_scan_1 = load_and_preprocess_real(test_cases[0]["path"], target_dim)
        _, real_scan_2 = load_and_preprocess_real(test_cases[1]["path"], target_dim)

    prompt_mismatched = "A photo of a golden retriever dog playing in a park on a sunny day."

    print("\n--- Prompts Tested ---")
    print(f"Prompt 1 (Matching Case 1): {prompt_1[:85]}...")
    print(f"Prompt 2 (Matching Case 2): {prompt_2[:85]}...")
    print(f"Prompt 3 (Unrelated Dog):   {prompt_mismatched}")

    emb_p1 = encode_prompt(prompt_1, clip_model, tokenizer, device)
    emb_p2 = encode_prompt(prompt_2, clip_model, tokenizer, device)
    emb_dog = encode_prompt(prompt_mismatched, clip_model, tokenizer, device)

    # Test volumes
    test_vols = {}
    if real_scan_1 is not None:
        test_vols[f"Real Case 1 ({test_cases[0]['seq']})"] = real_scan_1
    if real_scan_2 is not None:
        test_vols[f"Real Case 2 ({test_cases[1]['seq']})"] = real_scan_2

    test_vols["All-Zeros (0.0)"] = torch.zeros((1, 1, *target_dim), device=device)
    test_vols["Tiny Noise [0, 0.005]"] = torch.rand((1, 1, *target_dim), device=device) * 0.005
    test_vols["Gaussian Noise [0, 1]"] = torch.clamp(torch.randn((1, 1, *target_dim), device=device) * 0.25 + 0.5, 0.0, 1.0)
    test_vols["Constant Half (0.5)"] = torch.ones((1, 1, *target_dim), device=device) * 0.5

    print("\n" + "=" * 95)
    print(f"{'Input Volume':<25} | {'Sim Prompt 1':<15} | {'Sim Prompt 2':<16} | {'Sim Unrelated Dog':<18}")
    print("=" * 95)

    emb_dict = {}
    for name, vol in test_vols.items():
        emb = encode_volume_with_clip(vol, clip_model, device)
        emb_dict[name] = emb
        s_p1 = float(torch.sum(emb * emb_p1).item())
        s_p2 = float(torch.sum(emb * emb_p2).item())
        s_dog = float(torch.sum(emb * emb_dog).item())
        print(f"{name:<25} | {s_p1:<15.4f} | {s_p2:<16.4f} | {s_dog:<18.4f}")

    if real_scan_1 is not None:
        print("\n" + "=" * 95)
        print(f"{'Input Volume':<25} | {'Sim to Real Case 1':<20} | {'Sim to Real Case 2':<20}")
        print("=" * 95)
        e_r1 = emb_dict[f"Real Case 1 ({test_cases[0]['seq']})"]
        e_r2 = emb_dict.get(f"Real Case 2 ({test_cases[1]['seq']})", e_r1)
        for name, emb in emb_dict.items():
            s_r1 = float(torch.sum(emb * e_r1).item())
            s_r2 = float(torch.sum(emb * e_r2).item())
            print(f"{name:<25} | {s_r1:<20.4f} | {s_r2:<20.4f}")

if __name__ == "__main__":
    main()
