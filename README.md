# Overcoming Data Starvation in MS Lesion Segmentation: A Task-Aware Longitudinal Deep Learning Pipeline

This repository contains the code, models, and results for my VIP Project focused on segmenting new Multiple Sclerosis (MS) lesions from longitudinal T1/FLAIR MRI scans using a clinical cohort from the American University of Beirut Medical Center (AUBMC).

## Project Overview
Due to hardware constraints that made full 3D nnU-Net training impossible, this project utilizes a **2D Triplanar U-Net (ResNet-34)** architecture. 

The project demonstrates that under strict data constraints (n=40 training patients), architectural brute force (like 2.5D context or Attention gates) leads to data starvation. Instead, a **Tier 1 Task-Aware Refactoring** yields a massive +6.00% absolute Dice improvement.

### The Tier 1 Task-Aware Pipeline includes:
1. **Histogram Matching (SimpleITK):** Eliminating scanner drift between FLAIR1 and FLAIR2.
2. **The Difference Channel:** Explicitly feeding `[FLAIR1, FLAIR2, FLAIR2 - FLAIR1]` to the network.
3. **Balanced Sampling:** A 50/50 foreground/background sampler to fix class imbalance.
4. **Focal Tversky Loss:** To heavily penalize False Negatives.
5. **Test-Time Augmentation (TTA):** 4-way flip ensemble for inference stability.

## Repository Structure
* `1_Baseline_Triplanar_UNet.py`: The baseline script (BCE+Dice loss) achieving 32.81% Mean Dice.
* `2_Tier1_Task_Aware_Pipeline.py`: The final optimized script achieving 38.81% Mean Dice.
* `Master_Thesis_Data_Complete.xlsx`: Full training logs and per-patient results for the Baseline.
* `Tier_1_Task_Aware_Results.xlsx`: Full training logs and per-patient results for the Task-Aware model.

## Requirements
To run these scripts, the following libraries are required:
* `torch`
* `segmentation_models_pytorch`
* `nibabel`
* `SimpleITK`
* `scipy`# MS-Lesion-Segmentation-AUBMC
