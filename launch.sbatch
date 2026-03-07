#!/bin/bash
# Submission script for Lyra
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

source ~/miniconda3/bin/activate
conda activate py39_base
source env/bin/activate

CPU_LIMIT=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Define arrays for the arguments
# CASE_STUDY_OPTIONS=("four_branch_6" "four_branch_7" "himmelblau" "hat" "nonlinear_oscillator" "high_dimensional" "2dof_oscillator")
CASE_STUDY_OPTIONS=("nonlinear_oscillator" "high_dimensional" "2dof_oscillator")
AL_STRATEGY_OPTIONS=("moo")
REPEATS=15  # Number of repetitions for each experiment setting
# OUTPUT_PREFIX="N_it1" # Prefix for output naming

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
# Run the Python script with the selected arguments

# OUTPUT_NAME="${OUTPUT_PREFIX}_${OUTPUT_NUMBER}"
python main.py \
    --config default_moor \
    --case_study "$CASE_STUDY" \
    --al_strategy "$AL_STRATEGY" \
    --cpu_workers "$CPU_LIMIT" \
    --output "$OUTPUT_NUMBER"
