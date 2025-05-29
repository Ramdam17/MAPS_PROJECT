#!/bin/bash

#EXAMPLE: 

#sbatch train_minatar.sh BREAK

#max memory 62G
#CPUS per task 6

environment=${1:-default_environment}
base=2000000
seeds=(1 2 3 4 5)


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
for (( index=1; index<=${#seeds[@]}; index++ )); do
    setting_current=${seeds[index - 1]}
    echo "current seed: $setting_current"
    
    
    CUDA_VISIBLE_DEVICES=0,1 python $general_dir/examples/AC_lambda.py \
        -g $substrate \
        -seed $setting_current \
        -steps "$base" \
        #>> $general_dir/logs/AAA_Regular_${substrate}_setting7_seed${setting_current}_${base}steps.log
done