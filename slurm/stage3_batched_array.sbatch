#!/bin/bash
#SBATCH --job-name=s3b_mcmc
#SBATCH --output=logs/stage3/s3ba_%A_%a.out
#SBATCH --error=logs/stage3/s3ba_%A_%a.err
#SBATCH --partition=medium
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=tycho[06,09-14,16-17,20-24,27-30,32,53-54,56,58-60,83,91]

set -euo pipefail
mkdir -p logs
module purge; module load anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/ddehiwalage-don/conda_envs/torch_nadam

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""

CONFIG=${1:?pass a config}; shift
SPLITS=1
python -u stage3_mcmc.py "$CONFIG" --task-num ${SLURM_ARRAY_TASK_ID} --task-nb ${SLURM_ARRAY_TASK_COUNT} --batched --device cpu "$@"
