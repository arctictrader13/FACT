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
python3 remove_and_retrain.py --device=cpu --batch_size=20 --max_train_steps=100 --lr_decresing_step=10

#rm  $HOME/git_repo/FACT/results/remove_and_retrain/*
cp results/remove_and_retrain/*  $HOME/git_repo/FACT/results/remove_and_retrain/

source deactivate dl

