#!/bin/bash
#SBATCH --job-name=ppa-dataset-prep  # job short name
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task
#SBATCH --mem-per-cpu=1G         # memory per cpu-core
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=EMAIL
#SBATCH --account=##             # sponsored account job should be tracked under
# Send SIGTERM to the batch shell (B:) 120s before the wall-clock limit so the
# python process can catch it and stop cleanly at the next work boundary.
# Keep this lead time larger than the longest single work takes to process.
#SBATCH --signal=B:SIGTERM@120

# Template for batch running corppa dataset preparation
# (src/corppa/utils/dataset_prep.py) with slurm.
# Assumes a conda environment named corppa is set up with required dependencies.
#
# The python script installs SIGINT/SIGTERM handlers that stop cleanly at the
# next work boundary, so output always ends on a whole work. Combined with the
# --signal directive above and the --continue flag, a job that runs out of wall
# time stops gracefully and can be resumed where it left off.
#
# Update before using:
# - EMAIL for slurm notification
# - account (sponsored account to track the job under)
# - customize path for working directory (set username if using Princeton HPC)
#   (and make sure the directory exists)
# - set INPUT, IMAGE_DIR, and OUTPUT_DIR paths for the run
# - configure the time appropriately for the run
# - to resume an interrupted run, add --continue to the command below

module purge
module load anaconda3/2026.7
conda activate corppa

# change working directory for data output
cd /scratch/network/<USER>/corppa

# multithreaded polars / rapidfuzz honor the allocated CPUs
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"

INPUT=/scratch/network/<USER>/corppa/ppa_pages.jsonl.gz
IMAGE_DIR=/scratch/network/<USER>/corppa/images
OUTPUT_DIR=/scratch/network/<USER>/corppa/dataset

# Use `exec` so requested slurm SIGTERM signal will propagate to
# the python script, so the script can finish processing the current
# work pages before exiting cleaning, to support resuming with --continue flag.
# Using --no-progress to suppress the tqdm progress bar in output log file.
# Add --continue to resume a previously interrupted run.
exec python -m corppa.utils.dataset_prep \
    --no-progress \
    --log-level info \
    --log-file "dataset_prep_${SLURM_JOB_ID}.log" \
    "$INPUT" "$IMAGE_DIR" "$OUTPUT_DIR"
