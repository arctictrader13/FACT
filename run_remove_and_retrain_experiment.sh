#!/bin/bash

#SBATCH --job-name=lgpu0009_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=100000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/slurm_test_grad_rand.out

module purge
module load 2019
module load CUDA/10.0.130
module load Miniconda2

source activate dl

cp -r $HOME/git_repo/FACT/ $TMPDIR
cd FACT 
python3 remove_and_retrain.py --device=cuda --batch_size=2 --max_train_steps=30 --lr_decresing_step=3

source deactivate dl

