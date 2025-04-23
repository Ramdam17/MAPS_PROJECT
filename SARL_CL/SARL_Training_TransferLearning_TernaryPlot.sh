#!/bin/bash

# Command line arguments
setting=${1:-default_setting}
step_current=${2:-base_number_steps}

substrate="space_invaders"

# Define hyperparameters
#CHANGE THIS TO YOUR DIRECTORY,  /$local_repo_directory/SARL/MinAtar
general_dir="/home/juan-david-vargas-mazuera/ICML-RUNS/WorkshopPaper/REPO/MinAtar"

# For conda to work in scripts, you need to initialize it first
eval "$(conda shell.bash hook)"
conda activate marl

export PYTHONPATH="$PYTHONPATH:$general_dir"

load_dir='/home/juan-david-vargas-mazuera/ICML-RUNS/WorkshopPaper/REPO/MinAtar/results_curriculum/breakout_setting1_steps50000_1000_data_and_weights'

# Create a log directory for all runs
log_dir="$general_dir/results_curriculum/transfer_learning_logs3"
mkdir -p "$log_dir"

# Combinations of weights to iterate through
declare -a task_loss=(10 10 10 10 10 10 10 10 10 20 20 20 20 20 20 20 20 30 30 30 30 30 30 30 40 40 40 40 40 40 50 50 50 50 50 60 60 60 60 70 70 70 80 80 90 100 0 0)
declare -a weight_reg_loss=(10 20 30 40 50 60 70 80 90 10 20 30 40 50 60 70 80 10 20 30 40 50 60 70 10 20 30 40 50 60 10 20 30 40 50 10 20 30 40 10 20 30 10 20 10 0 100 0)
declare -a feature_loss=(80 70 60 50 40 30 20 10 0 70 60 50 40 30 20 10 0 60 50 40 30 20 10 0 50 40 30 20 10 0 40 30 20 10 0 30 20 10 0 20 10 0 10 0 0 0 0 100)


#declare -a task_loss=(10 10 10 10 10 10 10 10 10 30 30 30 30 30 30 30 50 50 50 50 50 70 70 70 90 100 0 0)
#declare -a weight_reg_loss=(10 20 30 40 50 60 70 80 90 10 20 30 40 50 60 70 10 20 30 40 50 10 20 30 10 0 100 0)
#declare -a feature_loss=(80 70 60 50 40 30 20 10 0 60 50 40 30 20 10 0 40 30 20 10 0 20 10 0 0 0 0 100)

# Number of combinations
num_combinations=${#task_loss[@]}

echo "EVALUATION CURRICULUM SPACE INVADERS BEFORE BREAKOUT"

seeds=(1 2 3 4 5)  # You can change these to your preferred seed values

for seed in "${seeds[@]}"; do
    echo "Running experiment with seed $seed..." >> "$log_dir/original_evaluation_before_transfer_learning.log"

    CUDA_VISIBLE_DEVICES=0,1 python "$general_dir/examples/maps.py" \
    -ema 25 \
    -cascade 50 \
    -g breakout \
    -seed "$seed" \
    -setting "$setting" \
    -steps "$step_current"  \
    -l "$load_dir" \
    -c \
    -ce \
    -anet \
    >> "$log_dir/original_evaluation_before_transfer_learning.log" 2>&1

    echo "Completed run with seed $seed" >> "$log_dir/original_evaluation_before_transfer_learning.log"
done


for seed in "${seeds[@]}"; do
    log_file="$log_dir/transfer_learning_seed_${seed}_setting${setting}_${step_current}steps.log"

    echo "Running experiment with seed $seed..." >> "$log_file" 2>&1

    # Loop through all combinations
    for (( i=0; i<$num_combinations; i++ )); do
        w1=${task_loss[$i]}
        w2=${weight_reg_loss[$i]}
        w3=${feature_loss[$i]}
        
        # Create a unique identifier for this run (only based on w1, w2, w3)
        run_id="task${w1}_reg${w2}_feat${w3}"
        
        # Create a specific log file for this run
        
        echo "Starting run with weights: task_loss=${w1}, weight_reg_loss=${w2}, feature_loss=${w3} (retention=${ret})"
        echo "Log file: $log_file"
        
        
        echo "TRAINING CURRICULUM BREAKOUT WITH WEIGHTS: task=${w1}, reg=${w2}, feat=${w3}" >> "$log_file"
        
        # Training phase
        CUDA_VISIBLE_DEVICES=0,1 python "$general_dir/examples/maps.py" \
            -ema 25 \
            -cascade 50 \
            -g "$substrate" \
            -seed "$seed" \
            -setting "$setting" \
            -steps "$step_current" \
            -l "$load_dir" \
            -c \
            -anet \
            -w1 "$w1" \
            -w2 "$w2" \
            -w3 "$w3" \
            >> "$log_file" 2>&1
        
        # Define checkpoint directory for this specific run (only based on w1, w2, w3)
        checkpoint_dir="${general_dir}/${substrate}_setting${setting}_steps${step_current}_${seed}_data_and_weights"
        
        echo "EVALUATION CURRICULUM SPACE INVADERS AFTER BREAKOUT WITH WEIGHTS: task=${w1}, reg=${w2}, feat=${w3}" >> "$log_file"
        
        # Evaluation phase
        CUDA_VISIBLE_DEVICES=0,1 python "$general_dir/examples/maps.py" \
            -ema 25 \
            -cascade 50 \
            -g breakout \
            -seed "$seed" \
            -setting "$setting" \
            -steps "$step_current" \
            -l "$checkpoint_dir" \
            -c \
            -ce \
            -anet \
            -w1 "$w1" \
            -w2 "$w2" \
            -w3 "$w3" \
            >> "$log_file" 2>&1
        
        echo "Completed run with weights: task_loss=${w1}, weight_reg_loss=${w2}, feature_loss=${w3}"
        echo "----------------------------------------"

        mv $checkpoint_dir "$general_dir/results_curriculum/Losses_Plot_${substrate}_setting${setting}_w1_${w1}_w2_${w2}_w3_${w3}_steps${step_current}_${seed}_${run_id}_data_and_weights"
    done
    echo "Completed run with seed $seed" >> "$log_file" 2>&1

done

echo "All combinations completed!"