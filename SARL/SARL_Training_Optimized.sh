#!/bin/bash

#EXAMPLE: 

#sbatch train_minatar.sh BREAK

#max memory 62G
#CPUS per task 6

environment=${1:-default_environment}
seed=${2:-default_seed}
setting_current=${3:-setting_input}

base=2000000


# Define hyperparameters
#seeds=(1 2 3)

#CHANGE THIS TO YOUR DIRECTORY,  /$local_repo_directory/SARL/MinAtar
#CHANGE THIS TO YOUR DIRECTORY,  /$local_repo_directory/SARL/MinAtar
general_dir="/home/juan-david-vargas-mazuera/ICML-RUNS/conference_paper/know_thyself/MAPS_PROJECT/SARL/MinAtar"

#module for computing cluster
#module load gcc python/3.11 opencv mpi4py arrow cuda cudnn rust
#source /home/juan-david-vargas-mazuera/ICML-RUNS/conference_paper/know_thyself/MAPS_PROJECT/MAPS/bin/activate

#conda activate MAPS


export PYTHONPATH="$PYTHONPATH:$general_dir"

# Set agents and substrate name based on the environment
case "$environment" in
    "BREAK")
        echo "RUNNING break out"
        substrate="breakout"
        ;;
    "SEA")
        echo "RUNNING sea quest"
        substrate="seaquest"
        ;;
    "SPACE")
        echo "RUNNING space invaders"
        substrate="space_invaders"
        ;;
    "AST")
        echo "RUNNING asterix"
        substrate="asterix"
        ;;
    "FREE")
        echo "RUNNING free way"
        substrate="freeway"
        ;;
    *)
        echo "Unknown environment: $environment"
        exit 1
        ;;
esac

echo "Substrate: $substrate"
#echo "Hidden: $hidden"

# Double loop through settings and seeds
echo "current setting: $setting_current"

seed_current=$seed
echo "  current seed: $seed_current"

CUDA_VISIBLE_DEVICES=0,1 python $general_dir/examples/maps_v2.py \
    -ema 25 \
    -cascade 50 \
    -g $substrate \
    -seed $seed_current \
    -setting $setting_current \
    -steps "$base" \
    #> $general_dir/logs/AAA_${substrate}_Regular_seed_${seed}_setting${setting_current}_${base}steps.log
