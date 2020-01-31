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

#time python3 remove_and_retrain.py --device=cuda --batch_size=10 --epochs=5 --print_step=1000 --grads fullgrad inputgrad random --phase=create_modified_datasets
time python3 remove_and_retrain.py --device=cuda --batch_size=10 --epochs=5 --print_step=500 --grads fullgrad inputgrad random --phase=train_on_modified_datasets


source deactivate dl

