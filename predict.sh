#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -w gorina9
#SBATCH --time=30:00:00
#SBATCH --job-name=nnUNet
#SBATCH --output=nnUNet.out
#SBATCH --error=nnUNet.err 


# nnUNetv2_train 010 3d_fullres 4 -num_gpus 4
nnUNetv2_predict -i /home/stud/sazidur/bhome/ELE670_project/nnUNetFrame/dataset/nnUNet_raw/Dataset001_ISLES24_6ch_f/imagesTs -o /home/stud/sazidur/bhome/ELE670_project/preds_001_fullres -d 001 -c 3d_fullres -f 10 --save_probabilities    