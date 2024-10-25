#!/bin/bash
# Submission script for nic5
#
#SBATCH --job-name=ALMO2D
#SBATCH --time=40:00:00 # hh:mm:ss
#SBATCH --array=0-179    # 12 combinations * 15 repetitions
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5  # 10 cores 
#SBATCH --mem=10G        # Memory to allocate per allocated CPU core
#SBATCH --partition=batch
#SBATCH --mail-user=jmoran@uliege.be
#SBATCH --mail-type=ALL

module load Python/3.9.6-GCCcore-11.2.0
source env/bin/activate

# Define arrays for the arguments
CASE_STUDY_OPTIONS=("four_branch_6" "four_branch_7" "himmelblau" "hat")
AL_STRATEGY_OPTIONS=("u" "eff" "mo")
REPEATS=15  # Number of repetitions for each experiment setting

# Calculate the number of options
NUM_CASE_STUDY=${#CASE_STUDY_OPTIONS[@]}
NUM_AL_STRATEGY=${#AL_STRATEGY_OPTIONS[@]}

# Total combinations
TOTAL_COMBINATIONS=$((NUM_CASE_STUDY * NUM_AL_STRATEGY))

# Ensure SLURM_ARRAY_TASK_ID is set
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set."
    exit 1
fi

# Calculate indices
idx_case_study=$(( SLURM_ARRAY_TASK_ID / (NUM_AL_STRATEGY * REPEATS) ))
idx_al_strategy=$(( (SLURM_ARRAY_TASK_ID / REPEATS) % NUM_AL_STRATEGY ))
repeat_num=$(( SLURM_ARRAY_TASK_ID % REPEATS ))

# Assign argument values
CASE_STUDY=${CASE_STUDY_OPTIONS[$idx_case_study]}
AL_STRATEGY=${AL_STRATEGY_OPTIONS[$idx_al_strategy]}

# Use the repetition number as the output number (starting from 1)
OUTPUT_NUMBER=$((repeat_num + 1))

echo "Running with --case_study=$CASE_STUDY --al_strategy=$AL_STRATEGY --output=$OUTPUT_NUMBER"
wandb offline
# Run the Python script with the selected arguments
python main.py --case_study "$CASE_STUDY" --al_strategy "$AL_STRATEGY" --output "$OUTPUT_NUMBER"
