#!/bin/bash
#SBATCH --job-name=s1_exp
#SBATCH --output=logs/stage1/s1_exp_%A_%a.out
#SBATCH --error=logs/stage1/s1_exp_%A_%a.err
#SBATCH --partition=medium
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4%2
#SBATCH --nodelist=tycho[91,58]

set -euo pipefail
mkdir -p logs
NODE=$(hostname -s)
if [[ "$NODE" == "tycho91" ]]; then
    module purge; module load cuda/12.0 anaconda/2024.10-1
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch_gpu
else
    module purge; module load cuda/11.8 anaconda/2024.10-1
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate /data/ddehiwalage-don/conda_envs/torch_nadam
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CONFIG=${1:?usage: sbatch slurm/stage1_export.sbatch configs/arm_xxx.yaml}

TOTAL=${TOTAL_NNOISE:-1000}

NSHARD=${SLURM_ARRAY_TASK_COUNT:-$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))}
IDX=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))

if (( NSHARD > TOTAL )); then
    echo "ERROR: $NSHARD shards for only $TOTAL noise reps" >&2; exit 1
fi

PER=$(( TOTAL / NSHARD ))
START=$(( IDX * PER ))
END=$(( START + PER ))
if (( IDX == NSHARD - 1 )); then END=$TOTAL; fi

echo "node=$NODE  gpus=${CUDA_VISIBLE_DEVICES:-none}  cpus=${SLURM_CPUS_PER_TASK}"
echo "shard ${IDX}/${NSHARD}  noise[${START},${END})  of TOTAL=${TOTAL}"

python -u stage1_compress.py "$CONFIG" --export-only \
    --noise-start "$START" --noise-end "$END" \
    -o compressor.save_batch_size=512 "${@:2}"
