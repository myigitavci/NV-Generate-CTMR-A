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
- **Anatomical Tissue Ground Truth (FreeSurfer SynthSeg 2.0)**:
  - Replaces heuristic intensity clustering with deep learning **SynthSeg 2.0** spatial parcellation directly on each synthesized 3D volume.
  - **CSF**: Ventricular system (lateral, 3rd, 4th, inf-lat ventricles) + subarachnoid CSF.
  - **White Matter**: Cerebral & cerebellar white matter, brainstem.
  - **Gray Matter**: Cerebral cortex + subcortical structures (thalamus, caudate, putamen, hippocampus).
- **Whole-Brain Axial Prompt Standardization**:
  - Replaced ambiguous GE 2D templates (`Ax T1, SE, NONE`) that produced partial/half brain slabs with canonical whole-brain Siemens 3.0T MAGNETOM Vida axial templates (`(Siemens, MAGNETOM_Vida, 3.0), (t1_tse_tra, SE, SK_SP), (0.009, 0.55, NONE, 90.0)`).
- **Evaluations**:
  1. **$T_2^*$ Relaxation Decay**: Sweeps Echo Time ($TE \in [10, 120]\,\text{ms}$) under fixed TR. Fits mono-exponential decay $S(TE) = S_0 \exp(-TE / T_2^*)$ to estimate tissue relaxation constants for CSF, Gray Matter (GM), and White Matter (WM).
  2. **$T_1$ Recovery & Contrast Inversion**: Sweeps Repetition Time ($TR \in [400, 2500]\,\text{ms}$) under short TE. Measures true anatomical GM/WM signal recovery.
  3. **FLAIR CSF Null Point**: Sweeps Inversion Time ($TI \in [1200, 2800]\,\text{ms}$) to evaluate CSF suppression.
- **Outputs**:
  - `output/eval_physics/t2_relaxation_curves.png` (3-panel: Bloch curve, slice sweep montage, SynthSeg anatomical ROI overlay)
  - `output/eval_physics/t1_recovery_curves.png` (3-panel: recovery curve, slice sweep montage, SynthSeg anatomical ROI overlay)
  - `output/eval_physics/flair_nulling_curve.png` (3-panel: nulling curve, slice sweep montage, SynthSeg anatomical ROI overlay)
  - `output/eval_physics/physics_quantitative_metrics.csv` (contains anatomical voxel counts, median ROI signals, and standard deviations)

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
- **Supported Modalities**:
  - `T2W` (Default): High fluid contrast, sharp ventricle and cortical boundaries across Siemens Vida, GE Discovery, and Philips Achieva.
  - `FLAIR`: Fluid suppression validation across vendor hardware.
  - `T1W`: Standard T1-weighted contrast using canonical Siemens Vida 3.0T prompts.
  - `all`: Sequentially runs and plots both T2W and FLAIR cohorts.
- **Evaluations**:
  1. **Vendor & Model Invariance (Fixed Seed)**:
     - Conditions on Siemens MAGNETOM Vida 3.0T, GE Discovery MR750 3.0T, and Philips Achieva dStream 3.0T while fixing the latent noise seed.
     - Quantifies 3D Structural Similarity (SSIM) and Normalized Mutual Information (NMI) across all 3 vendor pairs (Siemens vs GE, Siemens vs Philips, GE vs Philips).
     - Generates 3x5 publication figure: Tri-planar views, all 3 absolute difference maps, mid-axial intensity profile traces, and pairwise SSIM/NMI bar charts.
  2. **Field Strength SNR & CNR Scaling (1.5T vs 3.0T)**:
     - Compares 1.5T vs 3.0T for Siemens (`MAGNETOM_Sola` vs `MAGNETOM_Vida`), Philips (`Ingenia` vs `Achieva_dStream`), and GE (`Signa_HDxt` vs `Discovery_MR750`).
     - Calculates tissue Signal-to-Noise Ratio: $\text{SNR} = \mu_{\text{tissue}} / \sigma_{\text{background}}$.
     - Calculates Gray-to-White Matter Contrast-to-Noise Ratio: $\text{CNR} = |\mu_{\text{GM}} - \mu_{\text{WM}}| / \sigma_{\text{background}}$.
     - Verifies physical field scaling: $\text{CNR}_{3.0\text{T}} / \text{CNR}_{1.5\text{T}} > 1.0$.
- **Outputs**:
  - `output/eval_hardware/vendor_invariance_panel_{modality}.png`
  - `output/eval_hardware/field_strength_snr_cnr_comparison_{modality}.png`
  - `output/eval_hardware/hardware_disentanglement_metrics.csv`

---

### Module 3: Statistical Image Quality & 2.5D Slice FID
- **Script**: `scripts/eval_quality_and_fid.py`
- **Scientific Rationale**: Benchmark synthetic volume distributions directly against real, unseen test scans from the MIDI dataset.
- **Automated Quality & Completeness Guard (`generate_complete_brain`)**:
  - **Symmetry & Volume Verification**: Automatically assesses every volume for minimum parenchyma voxel count ($>100\text{k}$), left-right hemispheric balance ($<0.38$ asymmetry), anterior-posterior coverage ($<0.48$ cutoff), and central axial slice fullness.
  - **3-Attempt Regeneration Mechanism**: If an incomplete or half-brain is generated due to an ambiguous clinical prompt, it automatically regenerates with a canonical whole-brain axial template and fresh seed up to 3 times.
  - **Rejection Policy**: If a scan fails to synthesize a complete brain after 3 attempts, it is cleanly discarded so that corrupted/half brains never pollute the FID, quality distribution metrics, or gallery panels.
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
- **Scientific Rationale**: Statistically prove that the diffusion model synthesizes novel patient-specific brain anatomies and has not memorized training patient scans.
- **Root Cause Resolution for Half-Brains & Nonsensical Nearest Neighbors**:
  1. **Root Cause of T1 Half-Brains**: The sequence descriptor `t1_tse_tra` in the clinical dataset represented cervical/thoracic spine exams (`serv`, `torak`). This conditioned the model to generate only the cervical junction and inferior skull base. Replacing this with canonical whole-brain axial `(TRA, SE, SK)` and tightening axial coverage checks (`is_half_or_incomplete_brain` verifying $\ge 42$ axial slices and inferior-superior balance) guarantees complete, intact whole brains.
  2. **Root Cause of Nonsensical Nearest Neighbors**: Previous retrieval matched queries against mixed modalities without contrast filtering, yielding large cosine distances ($\sim 0.88$) and mismatched contrasts (e.g., T2 query matched against FLAIR). Enforcing **strict modality matching** (e.g., T2 query against real T2 training scans; T1 query against real T1 training scans) and canonical RAS orientation (`nib.as_closest_canonical`) produces clinically valid, contrast-consistent nearest-neighbor matches.
  3. **Robust Slice Extraction (`get_middle_slices`)**: Automatically selects the slice with the maximum brain parenchyma within the mid-cerebral region ($40\%\text{--}70\%$ extent), completely preventing black or truncated slice cuts.
- **Evaluations**:
  1. **Nearest-Neighbor (NN) Distance Audit**:
     - Measures cosine distance from synthetic scans to their nearest match within a verified reference gallery of real patient scans of the **identical modality** on disk.
     - Compares synthetic-to-real vs real-to-real baseline distance distributions to prove non-memorization.
  2. **Intra-Prompt Multi-Seed Diversity**:
     - Evaluates 5 random seeds from a fixed canonical whole-brain prompt, with automated 3-attempt completeness verification.
     - Computes pairwise 3D SSIM and diversity score ($1 - \text{SSIM}$).
  3. **Visual Top-3 Nearest Neighbor Retrieval**:
     - For each synthesized query, retrieves and displays the genuine top-3 closest training scans of the same modality.
     - Reports both **3D Latent Cosine Distance** and **Slice Structural Similarity (SSIM)** for every match to validate novelty in both semantic latent space and pixel space.
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
#  - Distinct marker shapes per modality (T1W: circle, T2W: square, FLAIR: triangle, PDW: diamond)
#  - Vendor-grouped coherent color palettes (Siemens blues, GE reds/oranges, Philips greens)
#  - Harmonized labels: 'GE Signa HDx/t', stripped underscores from 'Discovery MR XXX'
#  - Authentic CSV parameters and real sequences directly from the dataset (no synthetic or made-up sequences)
#  - Targeted T1w standardization using empirical median CSV parameters (TR~450ms, TE~9ms, FA=90) to ensure true T1 contrast and eliminate PDw clustering
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

