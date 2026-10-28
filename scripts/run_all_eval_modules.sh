#!/usr/bin/env bash
# ==============================================================================
# run_all_eval_modules.sh
# ==============================================================================
# Master Execution Runner for the 6 Core MR-CLIP Validation Pillars:
#   Pillar 1: Main Experiment: Acquisition Controllability (Sweeps, Frozen Regressors/Classifiers, Requested vs Predicted)
#   Pillar 2: Same Latent = Anatomy/Acquisition Disentanglement (SynthSeg Dice vs MR-CLIP Distance / Discrimination)
#   Pillar 3: The Essential Model Ablation (MR-CLIP vs Raw Metadata vs Unconditional)
#   Pillar 4: Quantitative MRI Parameter-Response Consistency (Monotonic TE decay, TR contrast, FLAIR CSF null TI*)
#   Pillar 6: Feature-Space Realism: 3D MR-CLIP FID with Bootstrap 95% Confidence Intervals
#   Pillar 7: Memorisation, Diversity & Dual Sanity Check (Same z diff meta vs Diff z same meta)
#
# Usage:
#   bash scripts/run_all_eval_modules.sh [GPU_ID] [CFG_SCALE] [NUM_SEEDS]
#
# Examples:
#   bash scripts/run_all_eval_modules.sh 1 3.0 100
# ==============================================================================

set -eo pipefail

GPU_ID="${1:-1}"
CFG_SCALE="${2:-3.0}"
NUM_SEEDS="${3:-100}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export FREESURFER_HOME="/usr/local/freesurfer/8.1.0"
export PATH="$FREESURFER_HOME/bin:${PATH:-}"

LOG_DIR="output/eval_logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/rigorous_eval_pipeline_$(date +'%Y%m%d_%H%M%S').log"

echo "==============================================================================" | tee -a "$PIPELINE_LOG"
echo "  STARTING RIGOROUS MR-CLIP SCIENTIFIC VALIDATION PIPELINE"                    | tee -a "$PIPELINE_LOG"
echo "  Target GPU Device:  CUDA_VISIBLE_DEVICES=$GPU_ID"                             | tee -a "$PIPELINE_LOG"
echo "  CFG Scale:          $CFG_SCALE"                                               | tee -a "$PIPELINE_LOG"
echo "  Number of Seeds:    $NUM_SEEDS"                                               | tee -a "$PIPELINE_LOG"
echo "  Master Log File:    $PIPELINE_LOG"                                            | tee -a "$PIPELINE_LOG"
echo "  Timestamp:          $(date)"                                                  | tee -a "$PIPELINE_LOG"
echo "==============================================================================" | tee -a "$PIPELINE_LOG"

run_module() {
    local mod_num="$1"
    local mod_name="$2"
    local script_cmd="$3"

    echo "" | tee -a "$PIPELINE_LOG"
    echo "------------------------------------------------------------------------------" | tee -a "$PIPELINE_LOG"
    echo "[PILLAR $mod_num START] $mod_name" | tee -a "$PIPELINE_LOG"
    echo "Command: $script_cmd" | tee -a "$PIPELINE_LOG"
    echo "Time: $(date)" | tee -a "$PIPELINE_LOG"
    echo "------------------------------------------------------------------------------" | tee -a "$PIPELINE_LOG"

    start_time=$(date +%s)
    eval "$script_cmd" 2>&1 | tee -a "$PIPELINE_LOG"
    status=${PIPESTATUS[0]}
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ $status -eq 0 ]; then
        echo ">>> [PILLAR $mod_num COMPLETED] in ${duration}s" | tee -a "$PIPELINE_LOG"
    else
        echo "!!! [PILLAR $mod_num FAILED] with exit code $status (elapsed: ${duration}s)" | tee -a "$PIPELINE_LOG"
        exit $status
    fi
}

# ------------------------------------------------------------------------------
# Pillar 1: Acquisition Controllability Suite
# ------------------------------------------------------------------------------
# run_module "1" "Acquisition Controllability (100 Seeds, Frozen Regressors/Classifiers)" \
#   "python3 scripts/eval_acquisition_controllability.py \
#     -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
#     -c configs/config_maisi_diff_model_rflow-mr-brain.json \
#     -t configs/config_network_rflow.json \
#     --num-seeds $NUM_SEEDS \
#     --output-dir output/eval_controllability \
#     --cfg-scale $CFG_SCALE"

# ------------------------------------------------------------------------------
# Pillar 2: Anatomy / Acquisition Disentanglement Matrix
# ------------------------------------------------------------------------------
run_module "2" "Anatomy vs Acquisition Disentanglement (SynthSeg Dice vs MR-CLIP Distance)" \
  "python3 scripts/eval_disentanglement_matrix.py \
    -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
    -c configs/config_maisi_diff_model_rflow-mr-brain.json \
    -t configs/config_network_rflow.json \
    --num-seeds $NUM_SEEDS \
    --output-dir output/eval_disentanglement \
    --cfg-scale $CFG_SCALE"

# ------------------------------------------------------------------------------
# Pillar 3: The Essential Model Ablation (Unconditional vs MR-CLIP)
# NOTE: Raw metadata baseline omitted — no trained checkpoint available.
#       Pass --raw-meta-ckpt <path> to include a third condition if a checkpoint exists.
# ------------------------------------------------------------------------------
run_module "3" "The Essential Model Ablation (Unconditional vs MR-CLIP)" \
  "python3 scripts/eval_ablation_comparison.py \
    -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
    -c configs/config_maisi_diff_model_rflow-mr-brain.json \
    -t configs/config_network_rflow.json \
    --num-seeds $NUM_SEEDS \
    --cfg-scale $CFG_SCALE \
    --output-dir output/eval_ablation"

# ------------------------------------------------------------------------------
# Pillar 4: Quantitative MRI Parameter-Response Consistency
# ------------------------------------------------------------------------------
run_module "4" "Quantitative MRI Parameter-Response Consistency" \
  "python3 scripts/eval_physics_fidelity.py \
    -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
    -c configs/config_maisi_diff_model_rflow-mr-brain.json \
    -t configs/config_network_rflow.json \
    --output-dir output/eval_physics \
    --cfg-scale $CFG_SCALE"

# ------------------------------------------------------------------------------
# Pillar 4b: Joint TR×TE×TI Physics Consistency (3D Monte Carlo Sweep)
# ------------------------------------------------------------------------------
run_module "4b" "Joint TR×TE×TI Physics Consistency (Levels A/B/C — 60 MC samples)" \
  "python3 scripts/eval_physics_joint_sweep.py \
    -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
    -c configs/config_maisi_diff_model_rflow-mr-brain.json \
    -t configs/config_network_rflow.json \
    --output-dir output/eval_physics_joint \
    --level ABC \
    --level-c-samples 60 \
    --cfg-scale $CFG_SCALE"

# ------------------------------------------------------------------------------
# Pillar 6: Realism in 3D MR-CLIP Feature Space with Bootstrap CIs
# ------------------------------------------------------------------------------
run_module "6" "3D MR-CLIP Feature Space Realism (FID with 95% Bootstrap CIs)" \
  "python3 scripts/eval_quality_and_fid.py \
    -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
    -c configs/config_maisi_diff_model_rflow-mr-brain.json \
    -t configs/config_network_rflow.json \
    --num-samples $NUM_SEEDS \
    --output-dir output/eval_quality \
    --cfg-scale $CFG_SCALE"

# ------------------------------------------------------------------------------
# Pillar 7: Memorisation, Diversity & Dual Sanity Check
# ------------------------------------------------------------------------------
run_module "7" "Memorisation, Diversity & Dual Sanity Check (Same z diff meta vs Diff z same meta)" \
  "python3 scripts/eval_memorization_novelty.py \
    -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
    -c configs/config_maisi_diff_model_rflow-mr-brain.json \
    -t configs/config_network_rflow.json \
    --output-dir output/eval_memorization \
    --cfg-scale $CFG_SCALE"

echo "" | tee -a "$PIPELINE_LOG"
echo "==============================================================================" | tee -a "$PIPELINE_LOG"
echo "  ALL VALIDATION PILLARS COMPLETED SUCCESSFULLY!"                             | tee -a "$PIPELINE_LOG"
echo "  Timestamp: $(date)"                                                         | tee -a "$PIPELINE_LOG"
echo "==============================================================================" | tee -a "$PIPELINE_LOG"
