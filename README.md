# Final Infarct Segmentation of Ischemic Stroke

This repository contains implementations for final infarct segmentation in ischemic stroke patients using multi-modal imaging data from the ISLES 2024 dataset.

## Project Overview

The project aims to predict final infarct lesions using six imaging modalities:
- NCCT (Non-Contrast CT)
- CTA (Computed Tomography Angiography)
- CBV (Cerebral Blood Volume)
- CBF (Cerebral Blood Flow)
- MTT (Mean Transit Time)
- Tmax (Time to maximum)

## Data Preprocessing

### ISLES 2024 Dataset Preparation
The preprocessing pipeline (`scripts/prepare_isles24.py`) includes:
- Brain extraction with lesion-aware skull stripping
- Intensity windowing
- Label verification and coverage checks

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yokko123/Final-Infarct-Segmentation-of-Ischemic-Stroke.git
cd Final-Infarct-Segmentation-of-Ischemic-Stroke
```

2. Install requirements and create the conda environment:
```bash
conda env create -f environment.yml
```

3. Configure nnUNet:
```bash
export nnUNet_raw="/path/to/nnUNet/raw/data"
```

## Data Processing

Preprocess ISLES 2024 dataset:
```bash
python scripts/prepare_isles24.py \
  --in-roots /path/to/isles24_train-1 /path/to/isles24_train-2 \
  --dataset-id 001 \
  --dataset-name ISLES24_6ch \
  --train-frac 0.8 \
  --save-brainmasks
```
Then run, 
```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```
this will create the preprocessed dataset ready for training. 
## Training
Now run the training, Fix the paths before running the training. 
 ```bash
sbatch train.sh
```
## Predict
```bash
sbatch predict.sh
```

## Generate Metrics
```bash
python scripts/evaluation.py
```
## Acknowledgments

- ISLES 2024 Challenge organizers and dataset providers
