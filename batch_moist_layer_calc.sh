#!/usr/bin/bash

#SBATCH -J monthly_stat           # Specify job name
#SBATCH -p compute          # Use partition shared
#SBATCH -N 1               # Specify number of nodes (1 for serial applications!)
#SBATCH -n 1               # Specify max. number of tasks to be invoked
#SBATCH -t 08:00:00        # Set a limit on the total run time
#SBATCH --mem=0            # 0=All memory 
#SBATCH --constraint=256G # Constrain to big memory nodes
#SBATCH -A um0878          # Charge resources on this project account
#SBATCH --output=logs/%x_%A_%a.o   # File name for standard output
#SBATCH --error=logs/%x_%A_%a.err   # File name for error output

module load python3


~/.local/share/jupyter/kernels/bin/python monthly_statistics.py --time=$1 --variable=$2 --region=$3 --stat_type=$4
