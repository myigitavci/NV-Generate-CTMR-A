import pandas as pd

df = pd.read_csv("/data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

flairs = df[df["sequence"].str.contains("FLAIR", case=False, na=False) | df["text"].str.contains("FLAIR", case=False, na=False)]
print("Total FLAIR rows:", len(flairs))
with open("scratch_flair_prompts.txt", "w") as f:
    f.write(f"Columns: {df.columns.tolist()}\n\n")
    for idx, r in flairs.head(50).iterrows():
        f.write(f"--- ID: {idx} | Seq: {r.get('sequence')}\n")
        f.write(f"    Text: {r.get('text')}\n\n")

print("WROTE scratch_flair_prompts.txt")
