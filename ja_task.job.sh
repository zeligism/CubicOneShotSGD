#!/bin/bash
#SBATCH --job-name=cubic-oneshot
#SBATCH --ntasks=1
#SBATCH --partition gpu
#SBATCH --time=0:10:00

source activate torch
echo "Submitting jobs in job array file: '${JA_FILE}'"
JA_FILE="${JA_FILE:-_experiments.ja}"

# Get the array_id'th line of job array script and run
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}"p "${JA_FILE}")
echo "job_${SLURM_ARRAY_TASK_ID}: ${COMMAND}"

# Run command
time $COMMAND
