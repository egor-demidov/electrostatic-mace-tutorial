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
#SBATCH --output="mace-run-2026-08-31.output"

### Configure the environment ###
# It loads all the necessary modules and already has the required python packages pre-installed
source /project/khalizov/mace-environment/mace-env.sh

# Run part 1: water molecule minimization
cd 01_molecule_minimization
srun python 01a_minimiza_water_with_mace.py
cd ..

# Run part 2:
#   - HgCl2 molecule minimization
#   - Solution minimization and MD
cd 02_bulk_solution_nvt
srun python 02a_minimize_HgCl2.py
srun python 02c_run_nvt.py
cd ..
