#!/bin/bash
#SBATCH --job-name=gc
#SBATCH --gres gpu:1
hostname
if [[ $# -lt 2 ]]
then
    echo "Expect two args to specify config file and suffix"
elif [[ $# -eq 2 ]]
then
    python -u habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/objectnav/$1.on.yaml --run-suffix $2 --wipe-only True
elif [[ $# -eq 3 ]]
then
    python -u habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/objectnav/$1.on.yaml --run-suffix $2 --ckpt-path ~/share/objectnav/$1/$1.$3.pth --wipe-only True
fi