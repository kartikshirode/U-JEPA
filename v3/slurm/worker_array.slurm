#!/bin/bash
#SBATCH -J ujepa
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1g.18gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/ujepa-%A_%a.out
#SBATCH --error=logs/ujepa-%A_%a.err

# One array task per MIG slice, each running its own shard of the grid. No
# collectives and no shared state beyond the output directory, where each cell
# writes one JSON named by its own hash. Two tasks can never target the same
# file.
#
#   mkdir -p logs
#   sbatch v3/slurm/worker_array.slurm grids/rq1_pilot.json runs/rq1_pilot
#   sbatch v3/slurm/worker_array.slurm grids/rq2_pilot.json runs/rq2_pilot
#
# The grid decides which experiment runs; an `arm` key of rq2 picks the gated
# driver. One script, because the queue shape is identical either way.
#
# The array range is 0 based so SLURM_ARRAY_TASK_ID is the shard index directly.
# Change --array to match how many slices you can actually hold; aicoeserver03
# and 04 have 14 of this flavour each and other people use them.

set -u

GRID="${1:-grids/rq1_pilot.json}"
OUT="${2:-runs/rq1_pilot}"

echo "JOB START : $(date)"
echo "HOST      : $(hostname)"
echo "ARRAY     : ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT}"
echo "GPUS      : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "GRID      : ${GRID} -> ${OUT}"

module purge
module load miniconda3
source /home/apps/codes/miniconda3/etc/profile.d/conda.sh
conda activate "${UJEPA_ENV:-torch-gpu-pip}"

# Compute nodes have no outbound network. Everything must already be cached by
# 03_prefetch.py, run on the login node, into a directory on shared /home.
export HF_HOME="${HF_HOME:-/home/${USER}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export U_JEPA_V3_PROBE_DIR="${U_JEPA_V3_PROBE_DIR:-/home/${USER}/probes}"

if [ ! -f "${U_JEPA_V3_PROBE_DIR}/sst.json" ]; then
  echo "FATAL probe sets missing at ${U_JEPA_V3_PROBE_DIR}"
  echo "Build them on the login node: python v3/scripts/01_build_probes.py --out ${U_JEPA_V3_PROBE_DIR}"
  exit 1
fi

mkdir -p "${OUT}"

# CUDA_VISIBLE_DEVICES is already narrowed to this task's slice. The worker
# leaves it alone under Slurm, and --node and --of come from the array.
python -m u_jepa_v3.runs.worker --grid "${GRID}" --out "${OUT}"
status=$?

echo "JOB END   : $(date), worker exit ${status}"
exit ${status}
