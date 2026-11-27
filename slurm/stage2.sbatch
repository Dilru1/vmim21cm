#!/bin/bash
#SBATCH --job-name=s2_nle
#SBATCH --output=logs/stage2/s2_%j.out
#SBATCH --error=logs/stage2/s2_%j.err
#SBATCH --partition=medium
#SBATCH --time=17:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --exclude=tycho[05-06,09-14,16-17,20-24,27-36,38-45,48-49,53-54,56-60,83,91]

set -euo pipefail
mkdir -p logs/stage2

module purge; module load anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_gpu

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export CUDA_VISIBLE_DEVICES=""

CONFIG=${1:?pass a config}
shift

python -u stage2_nle.py "$CONFIG" "$@"
