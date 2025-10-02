#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --job-name=nnUNet
#SBATCH --output=nnUNet.out
#SBATCH --error=nnUNet.err


nnUNetv2_train 050 3d_fullres 4 -num_gpus 4