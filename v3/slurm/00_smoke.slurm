#!/bin/bash
#SBATCH -J ujepa-smoke
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1g.18gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

# First thing to run on Baramati. Reports the slice, the derived dtype, whether
# the cards pair, and which stage 1 prerequisites are missing.
#
#   mkdir -p logs && sbatch v3/slurm/00_smoke.slurm
#
# Exit 2 means no CUDA. Exit 1 means the harness is fine and stage 1 is blocked
# on easyeditor or the probe sets.
#
# aicoeserver05 carries the 1g.24gb flavour instead; swap the gres line for
# --gres=gpu:1g.24gb:1 to land there.

set -u

echo "JOB START : $(date)"
echo "HOST      : $(hostname)"
echo "JOB ID    : ${SLURM_JOB_ID}"
echo "GPUS      : ${CUDA_VISIBLE_DEVICES:-unset}"

module purge
module load miniconda3
source /home/apps/codes/miniconda3/etc/profile.d/conda.sh
conda activate "${UJEPA_ENV:-torch-gpu-pip}"

export HF_HOME="${HF_HOME:-/home/${USER}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

nvidia-smi

python v3/scripts/00_smoke_gpu.py
status=$?

echo "JOB END   : $(date), smoke exit ${status}"
exit ${status}
