#!/bin/bash
#SBATCH --job-name=s1_merge
#SBATCH --output=logs/stage1/s1_merge_%j.out
#SBATCH --error=logs/stage1/s1_merge_%j.err
#SBATCH --partition=medium
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
mkdir -p logs
module purge; module load anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/ddehiwalage-don/conda_envs/torch_nadam

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""

CONFIG=${1:?usage: sbatch slurm/stage1_merge.sbatch configs/arm_xxx.yaml}

python -u merge_export_shards.py "$CONFIG" "${@:2}"
