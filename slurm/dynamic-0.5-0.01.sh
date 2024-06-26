#!/bin/bash


#SBATCH --job-name="hash"
#SBATCH --partition=cybersecurity
#SBATCH --nodelist=cyberstation1.csit.local
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=24G            
#SBATCH --gres=gpu:1
#SBATCH --error=slurm/output/job_error_%j.log  
#SBATCH --output=slurm/output/output%j.out
#SBATCH --chdir=../

# Load any required modules (environments, libraries etc.)
eval "$(conda 'shell.bash' 'hook' 2> /dev/null)" # initialize conda
conda activate spikingjelly

srun python dynamic.py --dataset mnist --cupy --epochs 10 --train_epochs 1 --alpha 0.5 --beta 0.01

