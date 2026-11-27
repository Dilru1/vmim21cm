#!/bin/bash
set -euo pipefail
cd /obs/ddehiwalage-don/21cm/sbi_project_FIN
DRY=${DRY:-1}
S3EXC="tycho[06,09-14,16-17,20-24,27-30,32,53-54,56,58-60,83,91],tycho35,tycho68,tycho88"
sub() { if [[ $DRY == 1 ]]; then echo "    sbatch $*" >&2; echo 000; else sbatch "$@" | awk '{print $NF}'; fi; }
cell() {
  local k=$1 cs=$2 ns_list=$3 do1=$4 cfg=configs_paper/patha_pureVMIM3/arm_mlp_vmim_t$1_mix2.yaml
  local cov=(-o compressor.init_seed=$cs -o compressor.tag_arm_with_seed=true) dep=()
  echo "== k$k c$cs"
  if [[ $do1 == 1 ]]; then
    j1=$(sub --job-name=s1mlp_s$cs --time=10:00:00 slurm/stage1_mlp.sbatch $cfg "${cov[@]}")
    echo "  s1 $j1"; dep=(--dependency=afterok:$j1)
  fi
  for ns in ${ns_list//,/ }; do
    ov=(-o nle.model=nsf -o nle.batch_size=32 "${cov[@]}" -o nle.init_seed=$ns -o nle.seed_subdir=true)
    j2=$(sub --job-name=s2_nsf_s${cs}_n$ns "${dep[@]}" slurm/stage2.sbatch $cfg "${ov[@]}")
    j3=$(sub --job-name=s3_nsf_s${cs}_n$ns --dependency=afterok:$j2 --time=10:00:00 --exclude=$S3EXC slurm/stage3.sbatch $cfg "${ov[@]}")
    echo "  n$ns s2 $j2 s3 $j3"
  done
}
for k in 4 8 16; do
  cell $k 45 42,43,44 1
  cell $k 43 43,44 0
  cell $k 44 43,44 0
done
