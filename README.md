# Landvisor Technical Assessment: Sparse Point Supervision for Remote Sensing

# Landvisor Technical Assessment: Sparse Point Supervision

## Overview

This repository contains my submission for the Landvisor semantic segmentation technical assessment. The objective is to train a segmentation model (U-Net with a pre-trained ResNet-34 backbone) on remote sensing imagery using **sparse point supervision** (incomplete tagging) rather than full dense ground-truth masks.

To achieve this, the project implements a custom **Partial Cross-Entropy (PCE) Loss** leveraging PyTorch's `ignore_index` to mask unannotated pixels during backpropagation.

## Project Structure

\`\`\`text
Zayan_Landvisor_Assessment/
├── data/ # Contains a tiny toy dataset (2 imgs) for rapid reviewer testing
├── src/
│ ├── dataset.py # Custom Dataset with dynamic point simulation & downsampling
│ ├── losses.py # Implementation of Partial Cross-Entropy Loss
│ ├── model.py # U-Net architecture (ResNet-34 encoder)
│ └── utils.py # Metrics (mIoU) and visualization logic
├── train.py # Main training and validation pipeline
├── experiments.py # Automated ablation study (Testing 1, 5, 10, 50 points)
├── requirements.txt # Environment dependencies
├── Technical_Report.pdf # Detailed methodology, experiment analysis, and results
└── README.md # Project documentation
\`\`\`

## ⚠️ Note on Local Compute Limitations

This pipeline was developed and tested locally on a CPU. To ensure reasonable iteration times during development, the following pragmatic constraints were added to the code:

1. **Dynamic Downsampling:** High-res images are dynamically resized to `512x512` in `dataset.py`.
2. **Epoch Caps:** Training loops in `train.py` and `experiments.py` are capped at 3-5 epochs.

_If reviewing on a GPU cluster, feel free to remove the resize transformations in `dataset.py` and increase the epochs to 30+ for full convergence._

## Quick Start (For Reviewers)

To respect your time, this repository includes a tiny "toy dataset" inside the `data/` folder so you can verify the pipeline compiles and runs without downloading the full 10GB LoveDA dataset.

### 1. Environment Setup

\`\`\`bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

### 2. Run the Baseline Training

To run a quick sanity check on the pipeline:
\`\`\`bash
python train.py
\`\`\`
_This will execute a short training loop and save `best_model.pth`._

### 3. Run the Ablation Study

To execute the automated experiment evaluating the impact of Point Density (1 vs 5 vs 10 vs 50 points):
\`\`\`bash
python experiments.py
\`\`\`
_This generates an `experiment_results.csv` file mapping point density to validation mIoU._

## Using the Full Dataset

If you wish to reproduce the full results detailed in the `Technical_Report.pdf`:

1. Download the full [LoveDA Dataset](https://zenodo.org/records/5706578).
2. Extract it to the `data/` directory.
3. Update the dataset paths in `train.py` and `experiments.py` to point to the full data folders.

## Key Technical Features

- **Dynamic Point Simulation:** The `LoveDASparseDataset` class dynamically samples $N$ points per class during runtime, allowing seamless experimentation.
- **Partial Cross-Entropy:** Implemented in `src/losses.py`, leveraging PyTorch's `ignore_index` to mask out unlabelled pixels during backpropagation.
- **Transfer Learning:** Utilizes an ImageNet-pretrained ResNet-34 encoder via `segmentation_models_pytorch` to guarantee a strong feature-extraction baseline.
