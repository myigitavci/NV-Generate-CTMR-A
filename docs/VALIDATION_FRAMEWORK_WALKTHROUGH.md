# Comprehensive Validation Framework Walkthrough

This document serves as the permanent reference guide for the 4-module validation suite for the 3D MR-CLIP brain MRI diffusion synthesis framework on the **MIDI** dataset.

All modules enforce **whole-brain axial prompts** (`plane axial`) to eliminate partial-slab / half-brain artifacts, format figures with publication-quality resolution, and output quantitative CSV and JSON records.

---

## 1. Overview of the 4 Evaluation Modules

```
                        ┌─────────────────────────────────────────────────────────┐
                        │   MR-CLIP 3D Brain MRI Validation Framework (MIDI)      │
                        └────────────────────────────┬────────────────────────────┘
                                                     │
         ┌───────────────────┬───────────────────────┴───────────────┬───────────────────┐
         ▼                   ▼                                       ▼                   ▼
┌──────────────────┐┌──────────────────────────────┐┌─────────────────────────┐┌─────────────────────────┐
│ Module 1:        ││ Module 2:                    ││ Module 3:               ││ Module 4:               │
│ Physics Fidelity ││ Scanner Disentanglement      ││ Image Quality & FID     ││ Novelty vs Memorization │
│                  ││                              ││                         ││                         │
│ • T2* mono-exp   ││ • Vendor Invariance          ││ • 2.5D Slice FID        ││ • Latent NN distance    │
│ • T1 recovery    ││   (Siemens vs GE vs Philips) ││   (Axial, Coronal, Sag) ││   to 49k training scans │
│ • FLAIR CSF null ││ • 3D SSIM & NMI              ││ • Volumetric Sharpness  ││ • Intra-prompt multi-   │
│ • Latent Lipsch. ││ • 1.5T vs 3.0T SNR/CNR       ││ • TV, Entropy, Gradient ││   seed diversity (SSIM) │
└──────────────────┘└──────────────────────────────┘└─────────────────────────┘└─────────────────────────┘
```

---

## 2. Implemented Modules & Technical Specifications

### Module 1: Physics Fidelity & Contrast Equations
- **Script**: `scripts/eval_physics_fidelity.py`
- **Scientific Rationale**: Confirm that generated brain volumes adhere to physical MRI Bloch decay and recovery equations rather than acting as an unconstrained image interpolator.
- **Evaluations**:
  1. **$T_2^*$ Relaxation Decay**: Sweeps Echo Time ($TE \in [10, 120]\,\text{ms}$) under fixed TR. Fits mono-exponential decay $S(TE) = S_0 \exp(-TE / T_2^*)$ to estimate tissue relaxation constants for CSF, Gray Matter (GM), and White Matter (WM).
  2. **$T_1$ Recovery & Contrast Inversion**: Sweeps Repetition Time ($TR \in [500, 3000]\,\text{ms}$) under short TE. Measures GM/WM signal recovery and verifies contrast inversion (WM bright at short TR, converging at long TR).
  3. **FLAIR CSF Null Point**: Sweeps Inversion Time ($TI \in [1000, 3500]\,\text{ms}$) to find the nulling minimum where $|1 - 2\exp(-TI/T_1)| \to 0$ (~$2200-2500\,\text{ms}$).
  4. **Latent Trajectory Continuity**: Quantifies Lipschitz gradient continuity ($\|\Delta x\| / \|\Delta z_t\|$).
- **Outputs**:
  - `output/eval_physics/t2_relaxation_curves.png`
  - `output/eval_physics/t1_recovery_curves.png`
  - `output/eval_physics/flair_nulling_curve.png`
  - `output/eval_physics/physics_quantitative_metrics.csv`

---

### Module 2: Scanner & Hardware Disentanglement
- **Script**: `scripts/eval_hardware_disentanglement.py`
- **Scientific Rationale**: Prove that the patient's individual anatomy (brain morphology) is decoupled from acquisition hardware (vendor brand, scanner model, field strength).
- **Prompt Rules & In-Distribution Scanners**:
  - Uses canonical MIDI prompt structure: `A brain MRI, plane axial, Age: 45, Gender: F, Scanner (Manufacturer, Model, Field Strength): ({vendor}, {model}, {field}), Acquisition (Description, Sequence, Variant): (TRA, ...), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): ({te}, {tr}, {ti}, {fa})`.
  - **Canonical Units**: `TE` (seconds), `TR` (seconds), `TI` (seconds or `NONE`), `FA` (degrees).
  - **Verified In-Distribution 3.0T Scanners**:
    - **Siemens**: `MAGNETOM_Vida`, 3.0T
    - **GE**: `Discovery_MR750`, 3.0T (verified 3T cohort)
    - **Philips**: `Achieva_dStream`, 3.0T
  - **Verified In-Distribution 1.5T Scanners**:
    - **Siemens**: `MAGNETOM_Sola`, 1.5T
    - **GE**: `Signa_HDxt`, 1.5T
- **Evaluations**:
  1. **Vendor & Model Invariance (Fixed Seed)**:
     - Conditions on Siemens MAGNETOM Vida 3.0T, GE Discovery MR750 3.0T, and Philips Achieva dStream 3.0T while fixing the latent noise seed.
     - Quantifies 3D Structural Similarity (SSIM) and Normalized Mutual Information (NMI) across vendor pairs.
     - Generates visual tri-planar comparisons, absolute difference maps ($|\text{Siemens} - \text{GE}|$, $|\text{Siemens} - \text{Philips}|$), and mid-axial intensity profile traces.
  2. **Field Strength SNR & CNR Scaling (1.5T vs 3.0T)**:
     - Compares 1.5T vs 3.0T for Siemens (`MAGNETOM_Sola` vs `MAGNETOM_Vida`) and GE (`Signa_HDxt` vs `Discovery_MR750`) across T1W and T2W.
     - Calculates tissue Signal-to-Noise Ratio: $\text{SNR} = \mu_{\text{tissue}} / \sigma_{\text{background}}$.
     - Calculates Gray-to-White Matter Contrast-to-Noise Ratio: $\text{CNR} = |\mu_{\text{GM}} - \mu_{\text{WM}}| / \sigma_{\text{background}}$.
     - Verifies physical field scaling: $\text{CNR}_{3.0\text{T}} / \text{CNR}_{1.5\text{T}} > 1.0$.
- **Outputs**:
  - `output/eval_hardware/vendor_invariance_panel.png`
  - `output/eval_hardware/field_strength_snr_cnr_comparison.png`
  - `output/eval_hardware/hardware_disentanglement_metrics.csv`

---

### Module 3: Statistical Image Quality & 2.5D Slice FID
- **Script**: `scripts/eval_quality_and_fid.py`
- **Scientific Rationale**: Benchmark synthetic volume distributions directly against real, unseen test scans from the MIDI dataset.
- **Evaluations**:
  1. **Tri-Planar 2.5D Slice FID**:
     - Extracts multi-slice cuts across Axial ($Z$), Coronal ($Y$), and Sagittal ($X$) planes.
     - Computes Fréchet feature distance $\text{FID}_{\text{Axial}}$, $\text{FID}_{\text{Coronal}}$, $\text{FID}_{\text{Sagittal}}$ against real test scans.
  2. **3D Medical Image Quality Metrics**:
     - **Volumetric Sharpness**: Laplacian variance $\text{Var}(\nabla^2 I)$ across brain parenchyma.
     - **Total Variation (TV)**: Spatial gradient energy measuring high-frequency noise / artifact level.
     - **Information Entropy**: $-\sum p \log_2 p$ measuring tissue structural information content.
     - **Edge Gradient Energy**: Sobel gradient magnitude at tissue interfaces.
- **Outputs**:
  - `output/eval_quality/fid_and_quality_summary.json`
  - `output/eval_quality/quality_metrics_distribution.png`
  - `output/eval_quality/real_vs_synthetic_slice_gallery.png`

---

### Module 4: Novelty vs Memorization Audit
- **Script**: `scripts/eval_memorization_novelty.py`
- **Scientific Rationale**: Statistically prove that the diffusion model is synthesizing novel brain anatomies and has not memorized training patient scans.
- **Evaluations**:
  1. **Nearest-Neighbor (NN) Distance Audit**:
     - Measures cosine distance from synthetic scans to their nearest match in the training set ($N=49,451$ precomputed scans in `data/image_embeddings.pkl`).
     - Measures baseline real-to-real distances between unseen real test scans and the training database.
     - Demonstrates that $d_{\text{min}}(\text{synth}) \approx d_{\text{min}}(\text{real})$, confirming natural inter-subject anatomical variation.
  2. **Intra-Prompt Multi-Seed Diversity**:
     - Fixes prompt and samples across multiple random seeds.
     - Calculates pairwise 3D SSIM and diversity score ($1 - \text{SSIM}$) to prove lack of mode collapse.
  3. **Visual Top-3 Nearest Neighbor Retrieval**:
     - For generated volumes, retrieves and displays the top-3 closest training scans from the dataset to demonstrate visual distinction in cortical gyri and ventricles.
- **Outputs**:
  - `output/eval_memorization/nearest_neighbor_distribution.png`
  - `output/eval_memorization/intra_prompt_diversity_panel.png`
  - `output/eval_memorization/top_nn_visual_montage.png`
  - `output/eval_memorization/memorization_audit_metrics.csv`

---

## 3. Ready-to-Run CLI Commands

> **Compute Note**:
> Always specify `CUDA_VISIBLE_DEVICES=1` (Quadro RTX 8000, 48GB VRAM) to ensure sufficient memory during the 3D autoencoder decoding phase at $(256, 256, 128)$.

```bash
# ------------------------------------------------------------
# 1. Physics Fidelity (Bloch curves, T2* fit, FLAIR null point)
# ------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python3 -m scripts.eval_physics_fidelity \
  --output-dir output/eval_physics \
  --cfg-scale 3.0

# ------------------------------------------------------------
# 2. Scanner & Hardware Disentanglement (Vendor SSIM + 1.5T/3T)
# ------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python3 -m scripts.eval_hardware_disentanglement \
  --output-dir output/eval_hardware \
  --cfg-scale 3.0

# ------------------------------------------------------------
# 3. Statistical Image Quality & Slice FID vs Unseen Real Scans
# ------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python3 -m scripts.eval_quality_and_fid \
  --output-dir output/eval_quality \
  --num-samples 15 \
  --cfg-scale 3.0

# ------------------------------------------------------------
# 4. Novelty vs Memorization Audit (Nearest-Neighbor & Diversity)
# ------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python3 -m scripts.eval_memorization_novelty \
  --output-dir output/eval_memorization \
  --num-queries 10 \
  --num-intra-seeds 5 \
  --cfg-scale 3.0

# ------------------------------------------------------------
# 5. Scanner Clustering & Manifold Analysis (Experiment 4)
# ------------------------------------------------------------
# Evaluates multi-scanner latent clustering (t-SNE and PCA) with:
#  - Distinct marker shapes per modality (T1W: circle, T2W: square, FLAIR: triangle, PDW: diamond, T2STARW: plus)
#  - Vendor-grouped coherent color palettes (Siemens blues, GE reds/oranges, Philips greens)
#  - Harmonized labels: 'GE Signa HDx/t', stripped underscores from 'Discovery MR XXX'
CUDA_VISIBLE_DEVICES=1 python3 -m scripts.run_mrclip_validation_experiments \
  -e configs/environment_dgx_diff_model_rflow-mr-brain.json \
  -c configs/config_maisi_diff_model_rflow-mr-brain.json \
  -t configs/config_network_rflow.json \
  --mrclip-ckpt models/diff_unet_3d_rflow-mr-brain_v1_phase2.pt \
  --clip-checkpoint /home/mav24/projects/MR-Styler/model/vitb-registered-with-planes-g1-lr-1e4-b-300-wd-02-warmup-2000-epoch-100-augnew2-text-dropout-tokenlength-98-et-20-rt-20-3D-anat-oasisgpu1clip001999-128-final/checkpoints/epoch_125.pt \
  --open-clip-src /home/mav24/Projects/AMA-CLIP/open_clip_dgx/open_clip/src \
  --csv /data/distclip_training/mr_all_3D_final_et_20_rt_20_simplified_anatomical_cleaned_with_pair_ids_orientation_test_unseen_OASIS_desktop.csv \
  --experiments 4 \
  --n-per-scanner 20 \
  --output-dir output/validation_experiments
```
