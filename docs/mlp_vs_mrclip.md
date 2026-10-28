# Sequence labels versus structured metadata MLP versus MR-CLIP

All three arms use `scripts.diff_model_train_mrclip`, the same pretrained base UNet,
training JSON, flow loss, conditioning dropout, scheduler, and training phases.
No VAE re-encoding or new CSV is needed. The MLP arm uses the `texts` in the
existing PKL, aligned by the same `idx` as MR-CLIP. It does not use the PKL's
MR-CLIP embedding values.

The `sequence` arm takes **only the coarse string labels from PKL `sequences`**:
T1W, T2W, FLAIR, PDW, and T2STARW. It never reads the PKL numeric `labels`
(which are finer protocol categories), metadata texts, or MR-CLIP features as
conditioning. A one-hot label vector goes through the same two-layer adapter
and time-embedding injection point as the other arms. Its training-only class
vocabulary is saved as `sequence_schema` and restored in phase 2 and inference.
Unknown, missing, or unsupported labels raise an error rather than silently
falling back to another class. Existing spacing/body geometry inputs are shared
across all arms; "sequence only" refers to acquisition/contrast conditioning.

## Conditioning

- Categorical fields: plane, manufacturer, scanner model, acquisition description,
  sequence, and variant. Use one-hot encoding with missing and unknown buckets.
- Numeric fields: field strength, TE, TR, TI, and flip angle. Use training-only
  mean/std standardization and an explicit missingness flag per field.
- Vocabulary and statistics are fitted only on `training` indices and saved in
  each checkpoint. Phase 2 and inference restore these settings.
- Timing values stay in the source text's units (normally seconds); flip angles
  stay in degrees. Do not convert a sweep from milliseconds twice.
- The structured vector goes through `Linear(input_dim, time_embed_dim)`, SiLU,
  and `Linear(time_embed_dim, time_embed_dim)`. This is the same two-layer adapter
  architecture and time-embedding injection point used for MR-CLIP.
- On the local dataset the structured dimension is 273, versus 512 for MR-CLIP.
  Consequently adapter parameter counts differ; report the actual counts from
  the saved manifests rather than claiming identical capacity. MR-CLIP also has
  external pretraining; the experiment compares conditioning representations.

## Submit training on RunAI

First sync the modified source files to the NFS project directory used by your
jobs. The launcher changes to that directory inside the container, so it uses
the updated NFS source even if the image contains an older copy.

From the project checkout on your cluster login node:

```bash
bash scripts/submit_conditioning_ablation.sh mlp 1 42 100
bash scripts/submit_conditioning_ablation.sh mrclip 1 42 100
bash scripts/submit_conditioning_ablation.sh sequence 1 42 100
```

These submit separate one-A100 jobs: 100 epochs, adapter-only training at 1e-3,
seed 42, dropout 0.1, initialized from the same original pretrained checkpoint:

`/nfs/home/mavci/projects/AMA-CLIP/codes/NV-Generate-CTMR-A/models/diff_unet_3d_rflow-mr-brain_v1.pt`

After **the phase 1 jobs finish**, submit phase 2 if your reported MR-CLIP model
uses full fine-tuning:

```bash
bash scripts/submit_conditioning_ablation.sh mlp 2 42 100
bash scripts/submit_conditioning_ablation.sh mrclip 2 42 100
bash scripts/submit_conditioning_ablation.sh sequence 2 42 100
```

Phase 2 starts each arm from its own phase 1 checkpoint. All arms run 100 epochs
with backbone LR 1e-5 and adapter LR 1e-4. These are fresh fine-tuning stages,
not optimizer-state resumes. To compare with an existing phase 2 experiment,
match its actual epoch counts and learning rates in all arms instead.

The launcher preserves the resource flags from the previous training command.
To print the full RunAI command without submitting:

```bash
DRY_RUN=1 bash scripts/submit_conditioning_ablation.sh mlp 1 42 100
```

Override the default project root or container image using
`ABLATION_PROJECT_ROOT` and `ABLATION_IMAGE`. Different seeds create different
job names, checkpoint directories, and logs. Reusing the same arm/phase/seed
targets the same outputs, so use a different seed or project root for an
independent rerun.

## Outputs and reproducibility

Relative to the NFS project directory, seed 42 checkpoints are:

| Arm | Phase 1 | Phase 2 |
| --- | --- | --- |
| MLP | `models/conditioning_ablation/mlp_seed42/diff_unet_3d_rflow-mr-brain_v1_phase1_mlp_seed42.pt` | `models/conditioning_ablation/mlp_seed42/diff_unet_3d_rflow-mr-brain_v1_phase2_mlp_seed42.pt` |
| MR-CLIP | `models/conditioning_ablation/mrclip_seed42/diff_unet_3d_rflow-mr-brain_v1_phase1_seed42.pt` | `models/conditioning_ablation/mrclip_seed42/diff_unet_3d_rflow-mr-brain_v1_phase2_seed42.pt` |
| Sequence | `models/conditioning_ablation/sequence_seed42/diff_unet_3d_rflow-mr-brain_v1_phase1_sequence_seed42.pt` | `models/conditioning_ablation/sequence_seed42/diff_unet_3d_rflow-mr-brain_v1_phase2_sequence_seed42.pt` |

Each is the **last epoch**, not a validation-selected best checkpoint. Alongside
it, a `.pt.json` manifest records the conditioning schema, training split hash,
initialization checkpoint path, hyperparameters, and trainable parameter count.
TensorBoard logs and CSV losses are separated by arm and seed.

The RNG is seeded before initialization and reset after model construction to
match shuffled batches, initial scale estimation, training noise, and CFG
dropout draws across input dimensions. This does not guarantee bitwise equality
of all GPU kernels. Missing latent files now fail explicitly rather than
silently reducing the dataset. The existing training split is reused; this
change does not establish patient-level separation of the existing splits.

Keep `--mr-clip-loss` disabled for all comparison arms. Compare the same
training stage and sampling conditions; do not compare a phase 1 MLP with a
phase 2 MR-CLIP model.

The sequence baseline has five input dimensions on the current dataset. Its
adapter has fewer parameters than either metadata arm; the manifest records
the actual trainable count. Use this comparison to assess the information
provided by coarse labels versus acquisition metadata, and report the counts.

## Generate MLP images

The existing inference CLI detects the conditioning type from the checkpoint.
For MLP, no `--open-clip-src` or `--clip-checkpoint` is needed:

```bash
python -m scripts.diff_model_infer_mrclip \
  -e configs/environment_dgx_diff_model_rflow-mr-brain.json \
  -c configs/config_maisi_diff_model_rflow-mr-brain.json \
  -t configs/config_network_rflow.json \
  --mrclip-ckpt models/conditioning_ablation/mlp_seed42/diff_unet_3d_rflow-mr-brain_v1_phase2_mlp_seed42.pt \
  --metadata-text 'A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, Signa_HDxt, 1.5), Acquisition (Description, Sequence, Variant): (Ax PD/T2, SE, SK), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.10184, 5.9, NONE, 90)' \
  --seed 1234 --cfg-scale 2 --output-dir output/conditioning_ablation/mlp
```

For MR-CLIP, use its checkpoint and supply its text encoder arguments as before.
Use identical prompts, image dimensions, seed values, CFG scales, and sampling
steps in both arms. The inference RNG is reset after model/text-encoder loading
so the requested seed selects the same initial diffusion noise.

## Generate sequence-only images

Use the same inference CLI with the sequence checkpoint and an explicit label.
No metadata text or MR-CLIP installation is needed:

```bash
python -m scripts.diff_model_infer_mrclip \
  -e configs/environment_dgx_diff_model_rflow-mr-brain.json \
  -c configs/config_maisi_diff_model_rflow-mr-brain.json \
  -t configs/config_network_rflow.json \
  --mrclip-ckpt models/conditioning_ablation/sequence_seed42/diff_unet_3d_rflow-mr-brain_v1_phase2_sequence_seed42.pt \
  --sequence-label T2W \
  --seed 1234 --cfg-scale 2 --output-dir output/conditioning_ablation/sequence
```

Alternatively, `--csv path.csv --csv-idx 42` reads the CSV `sequence` column for
sequence checkpoints. Metadata text is ignored for sequence checkpoints; the
CLI never infers a coarse label from acquisition parameters. A TE/TR sweep at
fixed sequence label, spacing, and noise seed therefore produces the same
sequence-baseline output, giving a control for acquisition-aware methods.

Legacy evaluation scripts that construct a fixed 512-dimensional wrapper are
not adapted by this change. In particular, `eval_ablation_comparison.py` is not
a valid MLP comparison: its raw-metadata branch assumes MR-CLIP inputs, and its
missing-checkpoint branch contains placeholder values. Do not use those values
as experimental evidence. Use the updated inference CLI to generate images.

## Local checks

```bash
python -m unittest discover -s tests -v
bash -n scripts/submit_conditioning_ablation.sh
```

The CPU tests cover parsing, train-only preprocessing, unknown/missing values,
row alignment, independence from MR-CLIP feature values, adapter gradients with
a frozen MAISI UNet, CFG null conditioning, checkpoint roundtrips, and both
training stages with matched diffusion noise in all three arms. They also check
that sequence features are independent of detailed metadata and fine numeric
labels, and that sequence inference needs neither text nor MR-CLIP. Full A100 training must run on
the cluster.
