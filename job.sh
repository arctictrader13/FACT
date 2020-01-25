#!/bin/bash

#SBATCH --job-name=lgpu0009_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=100000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/slurm_test_least.out

module purge
module load 2019
module load CUDA/10.0.130

source activate dl
srun python3 pixel_perturbation.py --grads fullgrad random gradcam --batch_size 5 --n_images 500 --k 0.01 --n_random_runs 5 --most_salient False --model vgg --model_type 16 --target_layer features


