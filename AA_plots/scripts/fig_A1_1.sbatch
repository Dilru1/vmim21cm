#!/bin/bash
#SBATCH --job-name=figA1_1_mi
#SBATCH --output=logs/aa_plots/figA1_%j.out
#SBATCH --error=logs/aa_plots/figA1_%j.err
#SBATCH --partition=medium
#SBATCH --time=60
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/aa_plots
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
export MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""
export SBI_EVAL_WORKERS=${SBI_EVAL_WORKERS:-16}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
/data/ddehiwalage-don/miniconda3/envs/torch_gpu/bin/python -u AA_plots/scripts/fig_A1_1_summary_mi_paths.py "$@"
