#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --job-name=nnUNet
#SBATCH --output=nnUNet.out
#SBATCH --error=nnUNet.err


# nnUNetv2_train 010 3d_fullres 4 -num_gpus 4
nnUNetv2_predict -i /home/stud/sazidur/bhome/ELE670_project/nnUNetFrame/dataset/nnUNet_raw/Dataset010_ISLES24_multi/imagesTs -o /home/stud/sazidur/bhome/ELE670_project/preds_010_fullres -d 010 -c 3d_fullres -f 4 --save_probabilities    