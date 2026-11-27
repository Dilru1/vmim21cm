#!/bin/bash
#SBATCH --job-name=s3_mcmc
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# usage: sbatch slurm/stage3.sbatch CONFIG CSEED NSEED [extra vmim21cm stage3 args]
# One job per run: all SBC targets sampled together (batched; ~20-30 min for 908 targets).
# For an array split instead, add e.g. --array=0-5 on the command line.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
sbi_env cpu
export CUDA_VISIBLE_DEVICES=""
CFG=$1 CS=$2 NS=$3; shift 3
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
  set -- --task "$SLURM_ARRAY_TASK_ID" --ntasks "$N" "$@"
fi
python -u -m vmim21cm stage3 "$CFG" --cseed "$CS" --nseed "$NS" "$@"
