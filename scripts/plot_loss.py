"""
plot_loss.py
============
Quick utility to plot training loss curves from the CSV log.

Usage:
  python -m scripts.plot_loss --csv logs/train_loss.csv --out logs/loss_curve.png
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves from CSV log")
    parser.add_argument("--csv", default="./logs/train_loss.csv", help="Path to train_loss.csv")
    parser.add_argument("--out", default="./logs/loss_curve.png", help="Output plot image path")
    parser.add_argument("--smooth", type=int, default=20, help="Rolling average window size")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} does not exist.")
        return

    df = pd.read_csv(args.csv)
    if len(df) == 0:
        print("Log is empty.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss curve
    ax1.plot(df["global_step"], df["loss"], alpha=0.3, color="tab:blue", label="Step Loss (raw)")
    if len(df) >= args.smooth:
        ax1.plot(df["global_step"], df["loss"].rolling(args.smooth).mean(), color="tab:blue", linewidth=2, label=f"Step Loss ({args.smooth}-pt MA)")
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("L1 Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Learning rate secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df["global_step"], df["lr"], color="tab:orange", linestyle="--", label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("MR-CLIP Diffusion UNet Training Loss")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved loss plot to: {args.out}")


if __name__ == "__main__":
    main()
