#!/bin/bash

seed=${1:-default_seed}
setting=${2:-default_setting}
base=100000

#CHANGE THIS TO YOUR DIRECTORY,  /$local_repo_directory/SARL/MinAtar
#CHANGE THIS TO YOUR DIRECTORY,  /$local_repo_directory/SARL/MinAtar
general_dir="/home/juan-david-vargas-mazuera/ICML-RUNS/conference_paper/know_thyself/MAPS_PROJECT/SARL_CL/MinAtar"

#module for computing cluster
#module load gcc python/3.11 opencv mpi4py arrow cuda cudnn rust
#source /home/juan-david-vargas-mazuera/ICML-RUNS/conference_paper/know_thyself/MAPS_PROJECT/MAPS/bin/activate
conda activate MAPS


export PYTHONPATH="$PYTHONPATH:$general_dir"

# Define arrays
settings=("breakout" "space_invaders" "seaquest" "freeway" )
load_path=("None" "breakout" "space_invaders" "seaquest")
steps=($base $((base*2)) $((base*3)) $((base*4)) )
steps_load=(0 $base $((base*2)) $((base*3)) )

echo "current setting: $setting"
echo "current seed: $seed"

# Define log file path once
log_file="$general_dir/results_curriculum/AAA_curriculum2_learning_seed_${seed}_setting${setting}_${base}steps.log"
destination_dir="$general_dir/results_curriculum/curriculum_setting${setting}_${base}steps_${seed}_data_and_weights"

rm $log_file

# Simpler array iteration using index
for i in "${!settings[@]}"; do
    env_current=${settings[i]}
    step_current=${steps[i]}
    load_current=${load_path[i]}
    step_load_current=${steps_load[i]}

    # Correct string concatenation in bash
    load_dir="${general_dir}/${load_current}_setting${setting}_steps${step_load_current}_${seed}_data_and_weights"
    current_dir="${general_dir}/${env_current}_setting${setting}_steps${step_current}_${seed}_data_and_weights"


    echo "current env: $env_current"
    echo "current steps: $step_current"
    echo "load directory: $load_dir"
    echo "current directory: $current_dir"

    # Correct if condition syntax
    if [ "$load_current" = "None" ]; then
        CUDA_VISIBLE_DEVICES=0,1 python "$general_dir/examples/maps.py" \
            -ema 25 \
            -cascade 50 \
            -g "$env_current" \
            -seed "$seed" \
            -setting "$setting" \
            -steps "$step_current" \
            -anet \
            >> "$log_file" 2>&1
    else 
        CUDA_VISIBLE_DEVICES=0,1 python "$general_dir/examples/maps.py" \
            -ema 25 \
            -cascade 50 \
            -g "$env_current" \
            -seed "$seed" \
            -setting "$setting" \
            -steps "$step_current" \
            -l "$load_dir" \
            -c \
            -anet \
            >> "$log_file" 2>&1
        
        mv $load_dir $general_dir/results_curriculum

    fi

    case "$env_current" in
        "breakout")
            evaluation_list=("breakout" )
            ;;

        "space_invaders")
            evaluation_list=("breakout" "space_invaders"  )
            ;;
        "seaquest")
            evaluation_list=("breakout" "space_invaders" "seaquest")
            ;;
        "freeway")
            evaluation_list=("breakout" "space_invaders" "seaquest" "freeway")
            ;;
        "asterix")
            evaluation_list=("breakout" "space_invaders" "seaquest" "freeway" "asterix")
            ;;
        *)
            echo "Unknown environment: $environment"
            exit 1
            ;;
    esac        


    for j in "${!evaluation_list[@]}"; do
        env_evaluation=${evaluation_list[j]}
        echo "Doing evaluation steps of curriculum learning approach, average over 100 seeds"
        echo "CURRENT ENV EVALUATION: $env_evaluation"


        CUDA_VISIBLE_DEVICES=0,1 python "$general_dir/examples/maps.py" \
            -ema 25 \
            -cascade 50 \
            -g "$env_evaluation" \
            -seed "$seed" \
            -setting "$setting" \
            -steps "$step_current" \
            -l "$current_dir" \
            -c \
            -ce \
            -anet \
            >> "$log_file" 2>&1
    done

    if [ "$i" -eq "$((${#settings[@]} - 1))" ]; then
        mv "$current_dir" "$destination_dir"
    fi

done