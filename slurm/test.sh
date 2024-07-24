#!/bin/bash


#SBATCH --job-name="Backdoor Experiment"
#SBATCH --partition=cybersecurity
#SBATCH --nodelist=cyberstation2.csit.local
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:00
#SBATCH --mem=54G
#SBATCH --gres=gpu:1
#SBATCH --error=slurm/output/job_error_%j.log  
#SBATCH --output=slurm/output/output%j.out
#SBATCH --chdir=../

# Load any required modules (environments, libraries etc.)
eval "$(conda 'shell.bash' 'hook' 2> /dev/null)" # initialize conda
conda activate spikingjelly2

srun python main.py --dataset caltech --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type static --cupy --epochs 30



# srun python main.py --dataset mnist --trigger_size 0.01 --epsilon 0.1 --type static --cupy --epochs 10
# srun python memtest.py

# srun --partition=cybersecurity --nodelist=cyberstation1.csit.local --nodes=1 --ntasks-per-node=1 --time=01:00:00 --mem=24G --gres=gpu:1 --pty bash -i