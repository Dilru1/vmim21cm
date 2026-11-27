#!/bin/bash
#SBATCH --job-name=bs_table
#SBATCH --output=logs/diag/bs_table_%j.out
#SBATCH --partition=medium
#SBATCH --time=120
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
cd /obs/ddehiwalage-don/21cm/sbi_project_FIN
export SBI_EVAL_WORKERS=16 MPLBACKEND=Agg
/data/ddehiwalage-don/conda_envs/torch_nadam/bin/python -u paper_plot_final/bs_table.py "$@"
