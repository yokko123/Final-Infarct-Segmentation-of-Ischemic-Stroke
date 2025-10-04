#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -w gorina9
#SBATCH --time=30:00:00
#SBATCH --job-name=nnUNet
#SBATCH --output=nnUNet.out
#SBATCH --error=nnUNet.err


nnUNetv2_train 001 3d_fullres 10 -num_gpus 1