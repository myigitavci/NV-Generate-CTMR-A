import pickle

with open('output/systematic_evaluation_30casesnew2/systematic_embeddings.pkl', 'rb') as f:
    embs = pickle.load(f)

texts_by_mod = {}
for e in embs:
    if e.get('type') == 'text':
        mod = e.get('modality')
        prompt = e.get('text_prompt')
        case_idx = e.get('case_idx')
        if mod not in texts_by_mod:
            texts_by_mod[mod] = []
        texts_by_mod[mod].append((case_idx, prompt))

with open('scratch_prompts_found.txt', 'w') as out:
    for m, items in texts_by_mod.items():
        out.write(f"=== Modality: {m} (Total {len(items)}) ===\n")
        for c_idx, p in items[:5]:
            out.write(f"  Case {c_idx}: {p}\n")
