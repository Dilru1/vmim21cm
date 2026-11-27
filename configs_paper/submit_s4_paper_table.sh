#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
L=configs_paper/stage4_lists
s4() { local name=$1 out=$2; shift 2
  sbatch --job-name=$name --mem=64G slurm/stage4.sbatch "$*" $out --n-corner 0 </dev/null; }

s4 s4_ref    paper_out/reference $PWD/$L/ref_pdfps_gmm.list $PWD/$L/pdf_ps_nsf_L8h64.list
s4 s4_patha  paper_out/patha     $(ls $PWD/$L/patha_*.list)
s4 s4_pathb  paper_out/pathb     $(ls $PWD/$L/pathb_*.list)
