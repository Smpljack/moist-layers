#!/usr/bin/bash

#SBATCH -J moist_layers           # Specify job name
#SBATCH -p compute          # Use partition shared
#SBATCH -N 1               # Specify number of nodes (1 for serial applications!)
#SBATCH -n 1               # Specify max. number of tasks to be invoked
#SBATCH -t 08:00:00        # Set a limit on the total run time
#SBATCH --mem=0            # 0=All memory 
#SBATCH -A um0878          # Charge resources on this project account
#SBATCH --output=logs/%x_%A_%a.o   # File name for standard output
#SBATCH --error=logs/%x_%A_%a.err   # File name for error output

module load python3


python calc_eml_chars_for_monsoon.py --time=$1
