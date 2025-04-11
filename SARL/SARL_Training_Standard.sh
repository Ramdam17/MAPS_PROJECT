#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --account=def-irina
#SBATCH --mem=3000M
#SBATCH --job-name=MINATAR-GPU-CEDAR-JUAN-SETTING-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=outputs/minatar_%N-%j.out
#SBATCH --error=errors/minatar_%N-%j.err
#SBATCH --mail-user=juan.davidvargas19022001@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#EXAMPLE: 

#sbatch train_minatar.sh BREAK

#max memory 62G
#CPUS per task 6

environment=${1:-default_environment}
seed=${2:-default_seed}
base=2000000


# Define hyperparameters
settings=(1)
#seeds=(1 2 3)

general_dir="/home/$USER/projects/def-gdumas85/$USER/MinAtar"

module load gcc python/3.11 opencv mpi4py arrow cuda cudnn rust

source /home/$USER/projects/def-gdumas85/$USER/MAPS/bin/activate

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
for (( index=1; index<=${#settings[@]}; index++ )); do
    setting_current=${settings[index - 1]}
    echo "current setting: $setting_current"
    
    seed_current=$seed
    echo "  current seed: $seed_current"
    
    CUDA_VISIBLE_DEVICES=0,1 python $general_dir/examples/maps.py \
        -ema 25 \
        -cascade 50 \
        -g $substrate \
        -seed $seed_current \
        -setting $setting_current \
        -steps "$base" \
        > $general_dir/logs/AAA_${substrate}_Regular_seed_${seed}_setting${setting_current}_${base}steps.log
done