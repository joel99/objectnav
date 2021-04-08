#!/bin/bash
#SBATCH --job-name=on_ddppo
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 5
#SBATCH --ntasks-per-node 8
#SBATCH --partition=long
#SBATCH --constraint=titan_x
#SBATCH --output=slurm_logs/ddppo-%j.out
#SBATCH --error=slurm_logs/ddppo-%j.err

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# Unused?
# SBATCH --mem=60GB
# SBATCH --time=12:00
# SBATCH --signal=USR1@600

hostname
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
if [[ $# -lt 2 ]]
then
    echo "Expect two args to specify config file and suffix"
elif [[ $# -eq 2 ]]
then
    srun -u python -u -m habitat_baselines.run \
        --exp-config habitat_baselines/config/objectnav/$1.on.yaml --run-suffix $2 \
        --run-type train
elif [[ $# -eq 3 ]]
then
    srun -u python -u -m habitat_baselines.run \
        --run-type train --exp-config habitat_baselines/config/objectnav/$1.on.yaml --run-suffix $2 \
        --ckpt-path ~/share/objectnav/$1/$1.$3.pth
fi
