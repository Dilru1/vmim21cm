#!/bin/bash
set -euo pipefail
cd /obs/ddehiwalage-don/21cm/sbi_project_FIN
DRY=${DRY:-1}
S3EXC="tycho[06,09-14,16-17,20-24,27-30,32,53-54,56,58-60,83,91],tycho35,tycho68,tycho70,tycho88"
CFG=configs_paper/patha_pureMSE/arm_mlp_mse_t4.yaml
sub() { if [[ $DRY == 1 ]]; then echo "    sbatch $*" >&2; echo 000; else sbatch "$@" | awk '{print $NF}'; fi; }
for cs in 43 44 45; do
  cov=(-o compressor.init_seed=$cs -o compressor.tag_arm_with_seed=true)
  j1=$(sub --job-name=s1mse_s$cs --time=10:00:00 slurm/stage1_mlp.sbatch $CFG "${cov[@]}")
  echo "== c$cs s1 $j1"
  for ns in 42 43 44; do
    ov=(-o nle.model=nsf -o nle.batch_size=32 "${cov[@]}" -o nle.init_seed=$ns -o nle.seed_subdir=true)
    j2=$(sub --job-name=s2mse_s${cs}_n$ns --dependency=afterok:$j1 slurm/stage2.sbatch $CFG "${ov[@]}")
    j3=$(sub --job-name=s3mse_s${cs}_n$ns --dependency=afterok:$j2 --time=10:00:00 --exclude=$S3EXC slurm/stage3.sbatch $CFG "${ov[@]}")
    echo "  n$ns s2 $j2 s3 $j3"
  done
done
