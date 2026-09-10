#!/bin/bash -l

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --qos=standard
#SBATCH --time=0-72:00:00
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --account=khalizov
#SBATCH --mem=20G
#SBATCH --job-name="mace-run"

### Configure the environment ###
# It loads all the necessary modules and already has the required python packages pre-installed
source /project/khalizov/mace-environment/mace-env.sh

for i in $(seq 2.2 0.2 12.0); do
    srun python 04a_run_md.py --restraint-r0 $i
done
