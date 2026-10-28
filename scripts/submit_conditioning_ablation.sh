#!/usr/bin/env bash
# Submit from the cluster login node after syncing the modified source to NFS.
# Usage: bash scripts/submit_conditioning_ablation.sh {sequence|mlp|mrclip} {1|2} [seed] [epochs]
# DRY_RUN=1 prints the exact command without submitting a job.
set -euo pipefail

conditioning="${1:?Usage: $0 sequence-or-mlp-or-mrclip phase seed epochs}"
phase="${2:?Provide phase 1 or 2}"
seed="${3:-42}"
epochs="${4:-100}"
case "$conditioning" in sequence|mlp|mrclip) ;; *) echo "Expected sequence, mlp, or mrclip" >&2; exit 2 ;; esac
case "$phase" in 1|2) ;; *) echo "Expected phase 1 or 2" >&2; exit 2 ;; esac
if [[ ! "$seed" =~ ^[0-9]+$ || ! "$epochs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Seed must be nonnegative and epochs must be positive integers" >&2
    exit 2
fi

project="${ABLATION_PROJECT_ROOT:-/nfs/home/mavci/projects/AMA-CLIP/codes/NV-Generate-CTMR-A}"
image="${ABLATION_IMAGE:-aicregistry:5000/mavci:nv_generate}"
model_dir="$project/models/conditioning_ablation/${conditioning}_seed${seed}"
checkpoint_tag=""
if [[ "$conditioning" != mrclip ]]; then checkpoint_tag="_${conditioning}"; fi
if [[ "$phase" == 1 ]]; then
    checkpoint="$project/models/diff_unet_3d_rflow-mr-brain_v1.pt"
    learning_rate=1e-3
else
    checkpoint="$model_dir/diff_unet_3d_rflow-mr-brain_v1_phase1${checkpoint_tag}_seed${seed}.pt"
    learning_rate=1e-5
fi

command=(
    runai training standard submit "${ABLATION_JOB_NAME:-nvgenerate-${conditioning}-p${phase}-s${seed}}"
    --image "$image" --image-pull-policy Always
    --gpu-devices-request 1 --large-shm --backoff-limit 0 --host-ipc --run-as-user
    --node-type A100 --host-path path=/nfs,mount=/nfs,readwrite
    --cpu-core-request 16 --cpu-memory-request 32G
    --command -- bash -c 'export MPLCONFIGDIR="$(mktemp -d /tmp/nvgenerate-mpl.XXXXXX)" && cd "$1" && shift && exec python -m scripts.diff_model_train_mrclip "$@"'
    bash "$project"
    -e "$project/configs/environment_dgx_diff_model_rflow-mr-brain.json"
    -c "$project/configs/config_maisi_diff_model_rflow-mr-brain.json"
    -t "$project/configs/config_network_rflow.json"
    --conditioning "$conditioning"
    --embeddings-pkl "$project/data/mrclip_embeddings.pkl"
    --data-json "$project/data/dataset_mrclip.json"
    --existing-ckpt "$checkpoint"
    --phase "$phase" --n-epochs "$epochs" --lr "$learning_rate" --proj-lr 1e-4
    --num-gpus 1 --seed "$seed" --clip-dropout 0.1
    --model-dir "$model_dir" --ckpt-suffix "_seed${seed}"
    --tb-dir "$project/runs/conditioning_ablation/${conditioning}_seed${seed}"
    --csv-log "$project/logs/conditioning_ablation/${conditioning}_seed${seed}_phase${phase}.csv"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
