#!/bin/bash
#SBATCH --job-name=fp_finder
#SBATCH --gres gpu:1
#SBATCH -p short,user-overcap,overcap
#SBATCH -A overcap
#SBATCH --output=/srv/flash1/jye72/projects/embodied-recall/slurm_logs/fp_finder-%j.out
#SBATCH --error=/srv/flash1/jye72/projects/embodied-recall/slurm_logs/fp_finder-%j.err
all_args=("$@")
echo ${all_args[@]}
python -u fp_finder.py ${all_args[@]}
