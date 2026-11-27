#!/bin/bash

set -euo pipefail

CSEEDS_RAW=""; NSEEDS_RAW=""; USER_OV=(); POS=()
while (( $# )); do
  case "$1" in
    --seed.compressor|--cseeds) CSEEDS_RAW="$2"; shift 2 ;;
    --seed.nle|--nseeds)        NSEEDS_RAW="$2"; shift 2 ;;
    -o)                         USER_OV+=(-o "$2"); shift 2 ;;
    -o*)                        USER_OV+=(-o "${1#-o}"); shift ;;
    *)                          POS+=("$1"); shift ;;
  esac
done
set -- ${POS[@]+"${POS[@]}"}

CONFIG=${1:?usage: submit_nle_grid.sh config.yaml ["gmm nsf"] ["std raw"] [STAGES] [--seed.compressor L] [--seed.nle L]}
FAMILIES=(${2:-gmm maf nsf})
SCALINGS=(${3:-std raw})
STAGES=${4:-123}

NSHARD=${NSHARD:-5}
MAXPAR=${MAXPAR:-2}
DRYRUN=${DRYRUN:-0}
BS_GMM=${BS_GMM:-}; BS_MAF=${BS_MAF:-32}; BS_NSF=${BS_NSF:-32}
bs_for() { case "$1" in gmm) echo "${BS_GMM}";; maf) echo "${BS_MAF}";; nsf) echo "${BS_NSF}";; *) echo "";; esac; }

IFS=',' read -r -a CSEEDS <<< "${CSEEDS_RAW}"
IFS=',' read -r -a NSEEDS <<< "${NSEEDS_RAW}"
(( ${#CSEEDS[@]} )) || CSEEDS=("")       # "" = use the yaml seed, no tagging
(( ${#NSEEDS[@]} )) || NSEEDS=("")

DO1=0; DO1_MODE="full"; DO2=0; DO3=0
case "$STAGES" in
  *1t*) DO1=1; DO1_MODE="train"  ;;
  *1e*) DO1=1; DO1_MODE="export" ;;
  *1*)  DO1=1; DO1_MODE="full"   ;;
esac
[[ "$STAGES" == *2* ]] && DO2=1
[[ "$STAGES" == *3* ]] && DO3=1
(( DO1 + DO2 + DO3 )) || { echo "STAGES='$STAGES' selects nothing (try 123, 23, 1t, 1e23)"; exit 1; }

for f in slurm/stage1_cnn.sbatch slurm/stage1_export.sbatch slurm/stage1_merge.sbatch \
         slurm/stage2.sbatch slurm/stage3.sbatch; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
done
if [[ $DO1 -eq 1 && "$DO1_MODE" != "export" ]] && ! grep -q -- '--no-export' slurm/stage1_cnn.sbatch; then
  echo "ERROR: slurm/stage1_cnn.sbatch does not pass --no-export, so the training" >&2
  echo "       job would export all ~9.1M rows itself and the array below would" >&2
  echo "       redo the same work -- the most expensive step, paid twice." >&2
  exit 1
fi

sub() {
  if [[ "$DRYRUN" == "1" ]]; then echo "    sbatch $*" >&2; echo "000000"; return; fi
  sbatch "$@" | awk '{print $NF}'          # no --parsable: needs Slurm >= 2.5
}

arm_yaml=$(grep -E "^arm_name:" "$CONFIG" | head -1 | awk '{print $2}')
echo "config=$CONFIG  arm(yaml)=$arm_yaml"
echo "stages=$STAGES  stage1=$([[ $DO1 -eq 1 ]] && echo $DO1_MODE || echo skip)"\
     " stage2=$([[ $DO2 -eq 1 ]] && echo yes || echo skip)"\
     " stage3=$([[ $DO3 -eq 1 ]] && echo yes || echo skip)"
echo "compressor seeds: ${CSEEDS[*]:-<yaml>}     NLE seeds: ${NSEEDS[*]:-<yaml>}"
echo "grid: families=[${FAMILIES[*]}] scalings=[${SCALINGS[*]}]  shards=$NSHARD%$MAXPAR"
echo "NOTE only init_seed is overridden; split_seed stays at the yaml value, so"
echo "     every ensemble member trains on the SAME train/val partition."
[[ "$DRYRUN" == "1" ]] && echo "MODE: DRYRUN"
echo

JOBS=(); n=0
for cs in "${CSEEDS[@]}"; do
  cov=()
  if [[ -n "$cs" ]]; then
    cov=(-o "compressor.init_seed=${cs}" -o "compressor.tag_arm_with_seed=true")
    tag="_s${cs}"
  else
    tag=""
  fi
  echo "===== compressor seed ${cs:-<yaml>}  -> arm ${arm_yaml}${tag}"

  DEP=""
  if [[ $DO1 -eq 1 ]]; then
    if [[ "$DO1_MODE" != "export" ]]; then
      j=$(sub --job-name="s1t${tag}" slurm/stage1_cnn.sbatch "$CONFIG" \
              ${cov[@]+"${cov[@]}"} ${USER_OV[@]+"${USER_OV[@]}"})
      echo "  train  : $j"; JOBS+=("$j"); DEP="--dependency=afterok:$j"
    fi
    if [[ "$DO1_MODE" != "train" ]]; then
      jx=$(sub --job-name="s1e${tag}" $DEP --array=0-$((NSHARD-1))%${MAXPAR} \
               slurm/stage1_export.sbatch "$CONFIG" \
               ${cov[@]+"${cov[@]}"} ${USER_OV[@]+"${USER_OV[@]}"})
      echo "  export : $jx  [array 0-$((NSHARD-1))%${MAXPAR}]"; JOBS+=("$jx")
      jm=$(sub --job-name="s1m${tag}" --dependency=afterok:"$jx" \
               slurm/stage1_merge.sbatch "$CONFIG" \
               ${cov[@]+"${cov[@]}"} ${USER_OV[@]+"${USER_OV[@]}"})
      echo "  merge  : $jm"; JOBS+=("$jm"); DEP="--dependency=afterok:$jm"
    fi
  fi

  if (( DO2 == 0 && DO3 == 0 )); then
    echo
    continue
  fi
  for ns in "${NSEEDS[@]}"; do
    nov=()
    [[ -n "$ns" ]] && nov=(-o "nle.init_seed=${ns}" -o "nle.seed_subdir=true")
    for fam in "${FAMILIES[@]}"; do
      for sc in "${SCALINGS[@]}"; do
        [[ "$sc" == "raw" ]] && flag="--raw-t" || flag=""
        ov=(-o "nle.model=${fam}")
        bs=$(bs_for "$fam"); [[ -n "$bs" ]] && ov+=(-o "nle.batch_size=${bs}")
        ov+=(${cov[@]+"${cov[@]}"} ${nov[@]+"${nov[@]}"} ${USER_OV[@]+"${USER_OV[@]}"})

        J2=""
        if [[ $DO2 -eq 1 ]]; then
          J2=$(sub --job-name="s2_${fam}${tag}_n${ns:-y}" $DEP \
                   slurm/stage2.sbatch "$CONFIG" "${ov[@]}" $flag)
          JOBS+=("$J2")
        fi
        if [[ $DO3 -eq 1 ]]; then
          d3=(); [[ -n "$J2" ]] && d3=(--dependency=afterok:"$J2")
          J3=$(sub --job-name="s3_${fam}${tag}_n${ns:-y}" "${d3[@]}" \
                   slurm/stage3.sbatch "$CONFIG" "${ov[@]}" $flag)
          JOBS+=("$J3")
          echo "  ${fam}/${sc} nle_seed=${ns:-<yaml>} : s2=$J2 s3=$J3"
        else
          echo "  ${fam}/${sc} nle_seed=${ns:-<yaml>} : s2=$J2"
        fi
        n=$((n+1))
      done
    done
  done
  echo
done

echo "submitted ${#JOBS[@]} jobs across $n stage-2/3 cells"
echo "watch : squeue -u \$USER -o '%.10i %.9P %.16j %.2t %.10M %R'"
echo "cancel: scancel ${JOBS[*]}"
