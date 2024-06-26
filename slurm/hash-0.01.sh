#!/bin/bash


#SBATCH --job-name="hash"
#SBATCH --partition=cybersecurity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=24G            
#SBATCH --gres=gpu:1
#SBATCH --error=slurm/output/job_error_%j.log  
#SBATCH --output=slurm/output/output%j.out
#SBATCH --chdir=../

# Load any required modules (environments, libraries etc.)
eval "$(conda 'shell.bash' 'hook' 2> /dev/null)" # initialize conda
conda activate spikingjelly

srun python main.py --dataset mnist --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type static --cupy --epochs 10

# srun python main.py --dataset mnist --trigger_size 0.01 --epsilon 0.1 --type static --cupy --epochs 10
# srun python memtest.py