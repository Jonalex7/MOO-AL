#!/bin/bash
# Submission script for nic5
#
#SBATCH --job-name=batchAL_fourbranch
#SBATCH --time=36:00:00 # hh:mm:ss
#SBATCH --array=0-11    # 36 combinations in total (1 * 3 * 4)
#SBATCH --ntasks=1
#SBATCH --mem=10G        # Memory to allocate per allocated CPU core
#SBATCH --partition=batch

# Define arrays for the arguments
# LS_OPTIONS=("four_branch" "himmelblau" "pushover_frame")
LS_OPTIONS=("four_branch")
ACQ_F_OPTIONS=("correlation_det" "correlation_eigen" "u_function")
ACQ_B_OPTIONS=(3 5 7 10)

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$(( ${#LS_OPTIONS[@]} * ${#ACQ_F_OPTIONS[@]} * ${#ACQ_B_OPTIONS[@]} ))

# Get the combination based on SLURM_ARRAY_TASK_ID
LS_INDEX=$(( SLURM_ARRAY_TASK_ID / ( ${#ACQ_F_OPTIONS[@]} * ${#ACQ_B_OPTIONS[@]} ) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % ( ${#ACQ_F_OPTIONS[@]} * ${#ACQ_B_OPTIONS[@]} ) ))
ACQ_F_INDEX=$(( REMAINDER / ${#ACQ_B_OPTIONS[@]} ))
ACQ_B_INDEX=$(( REMAINDER % ${#ACQ_B_OPTIONS[@]} ))

# Get the values for each argument
LS=${LS_OPTIONS[$LS_INDEX]}
ACQ_F=${ACQ_F_OPTIONS[$ACQ_F_INDEX]}
ACQ_B=${ACQ_B_OPTIONS[$ACQ_B_INDEX]}

#SBATCH --mail-user=jmoran@uliege.be
#SBATCH --mail-type=ALL

# Run the Python script with the selected arguments
python main_batchactive.py --ls=$LS --acq_f=$ACQ_F --acq_b=$ACQ_B
