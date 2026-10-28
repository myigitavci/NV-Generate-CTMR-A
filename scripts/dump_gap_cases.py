import pandas as pd
from scripts.evaluate_mrrate_systematic import extract_modality, resolve_filepath

def main():
    csv_path = "data/batch00_metadata_processed_ready_age_sex.csv"
    metrics_path = "output/systematic_evaluation_mrrate_p2/metrics_per_sample.csv"
    
    df_metrics = pd.read_csv(metrics_path)
    df_meta = pd.read_csv(csv_path)

    df_meta['modality'] = df_meta.apply(extract_modality, axis=1)
    df_meta['resolved_path'] = df_meta['filepath'].apply(lambda p: resolve_filepath(str(p), '/data/mrrate'))

    modalities = [m for m in ['T1W', 'T2W', 'FLAIR', 'T2STARW', 'PDW'] if m in df_meta['modality'].values]
    per_mod = 25
    selected_rows = []
    for mod in modalities:
        m_df = df_meta[df_meta['modality'] == mod]
        m_with_real = m_df[m_df['resolved_path'].notna()]
        m_without_real = m_df[m_df['resolved_path'].isna()]
        picked = []
        if len(m_with_real) > 0:
            picked.append(m_with_real.sample(n=min(len(m_with_real), per_mod), random_state=42))
        needed = per_mod - sum(len(p) for p in picked)
        if needed > 0 and len(m_without_real) > 0:
            picked.append(m_without_real.sample(n=min(len(m_without_real), needed), random_state=42))
        if picked:
            selected_rows.append(pd.concat(picked))

    df_sel = pd.concat(selected_rows).reset_index(drop=True)

    cfg4 = df_metrics[df_metrics['cfg_scale'] == 4.0]
    case_stats = cfg4.groupby('case_idx').agg({
        'sim_real_text': 'first',
        'sim_text': 'mean',
        'sim_real': 'mean',
        'vol_mean': 'mean',
        'vol_std': 'mean'
    }).reset_index()

    case_stats['gap'] = case_stats['sim_real_text'] - case_stats['sim_text']
    sorted_cases = case_stats.sort_values(by='gap', ascending=False)

    out_file = "output/systematic_evaluation_mrrate_p2/gap_detailed_cases.txt"
    with open(out_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("DETAILED BREAKDOWN OF TOP DISCREPANCY CASES (REAL HIGH, SYNTHETIC LOW)\n")
        f.write("=" * 100 + "\n\n")

        for idx, r in sorted_cases.head(15).iterrows():
            c_id = int(r['case_idx'])
            row_meta = df_sel.iloc[c_id]
            f.write(f"--- CASE #{c_id:03d} ---\n")
            f.write(f"Modality: {row_meta['modality']}\n")
            f.write(f"Real <-> Text Cosine Sim:      {r['sim_real_text']:.4f}\n")
            f.write(f"Synthetic <-> Text Cosine Sim: {r['sim_text']:.4f}\n")
            f.write(f"Gap (Real - Syn):              {r['gap']:+.4f}\n")
            f.write(f"Synthetic <-> Real Cosine Sim: {r['sim_real']:.4f}\n")
            f.write(f"Volume Mean Intensity:         {r['vol_mean']:.1f}\n")
            f.write(f"Filepath: {row_meta['filepath']}\n")
            f.write(f"Prompt Text:\n  \"{row_meta['text']}\"\n\n")

    print(f"Wrote details to {out_file}", flush=True)

if __name__ == "__main__":
    main()
