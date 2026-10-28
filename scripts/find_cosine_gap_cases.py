import os
import pandas as pd

def main():
    csv_path = "output/systematic_evaluation_mrrate_p2/metrics_per_sample.csv"
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)
    
    # Filter for CFG 4.0 (the primary operational guidance scale)
    cfg4 = df[df["cfg_scale"] == 4.0].copy()

    # Aggregate by case_idx
    case_stats = cfg4.groupby(["case_idx", "modality", "scanner"]).agg({
        "sim_real_text": "first",
        "sim_text": ["mean", "std"],
        "sim_real": ["mean", "std"],
        "vol_mean": "mean",
        "vol_std": "mean"
    }).reset_index()

    case_stats.columns = ["case_idx", "modality", "scanner", "sim_real_text", "sim_text_mean", "sim_text_std", "sim_real_mean", "sim_real_std", "vol_mean", "vol_std"]
    case_stats["gap"] = case_stats["sim_real_text"] - case_stats["sim_text_mean"]
    sorted_cases = case_stats.sort_values(by="gap", ascending=False)

    out_file = "output/systematic_evaluation_mrrate_p2/gap_analysis_report.txt"
    with open(out_file, "w") as f:
        f.write("=" * 110 + "\n")
        f.write("ANALYSIS: CASES WHERE REAL SIMILARITY IS HIGH BUT SYNTHETIC IS LOW (CFG = 4.0)\n")
        f.write("=" * 110 + "\n")
        f.write(f"Total Cases Evaluated: {len(case_stats)}\n")
        f.write(f"Cases with (Real - Synthetic) Gap > 0.15: {(case_stats['gap'] > 0.15).sum()}\n")
        f.write(f"Cases with (Real - Synthetic) Gap > 0.10: {(case_stats['gap'] > 0.10).sum()}\n")
        f.write(f"Cases where Synthetic >= Real: {(case_stats['gap'] <= 0.0).sum()}\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Case':<9} | {'Modality':<8} | {'Real<->Text':<11} | {'Syn<->Text':<11} | {'Gap (R-S)':<10} | {'Syn<->Real':<10} | {'VolMean':<8} | {'Scanner'}\n")
        f.write("-" * 110 + "\n")

        for idx, r in sorted_cases.head(25).iterrows():
            c_id = int(r['case_idx'])
            f.write(f"Case #{c_id:03d} | {r['modality']:<8} | {r['sim_real_text']:<11.4f} | {r['sim_text_mean']:<11.4f} | {r['gap']:<+10.4f} | {r['sim_real_mean']:<10.4f} | {r['vol_mean']:<8.1f} | {r['scanner']}\n")

        f.write("\n" + "=" * 110 + "\n")
        f.write("MODALITY BREAKDOWN (AVERAGES AT CFG 4.0):\n")
        f.write("=" * 110 + "\n")
        mod_summary = case_stats.groupby("modality").agg({
            "sim_real_text": "mean",
            "sim_text_mean": "mean",
            "sim_real_mean": "mean",
            "gap": "mean",
            "vol_mean": "mean"
        }).reset_index()
        f.write(mod_summary.to_string(index=False) + "\n")

    print(f"Saved analysis to {out_file}", flush=True)

if __name__ == "__main__":
    main()
