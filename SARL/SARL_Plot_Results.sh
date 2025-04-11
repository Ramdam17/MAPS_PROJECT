# Define hyperparameters
general_dir="/home/juan-david-vargas-mazuera/ICML-RUNS/WorkshopPaper/REPO/MinAtar"

# For conda to work in scripts, you need to initialize it first
eval "$(conda shell.bash hook)"
conda activate marl

export PYTHONPATH="$PYTHONPATH:$general_dir"

log_file=$general_dir/results/AAA_log_plot.log
rm $log_file

python examples/maps.py -pl -plfile /home/juan-david-vargas-mazuera/ICML-RUNS/WorkshopPaper/REPO/MinAtar/results/ 