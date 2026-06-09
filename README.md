# U-Net for Skin Lesion Segmentation

## Overview

This project implements a **U-Net deep learning model with ResNet-34 encoder** for automated skin lesion segmentation on the **ISIC 2018 dataset**. The model achieved a **Dice coefficient of 0.865 and IoU of 0.779** on the held-out test set (1,000 images), exceeding the target threshold of 0.83 in the proposal.

**Key Achievements:**
- ✓ Target Dice ≥ 0.83 **ACHIEVED** (0.8647 on test set)
- ✓ Strong generalization with 4-7% validation-to-test degradation
- ✓ 5.1× improvement over traditional baseline (Otsu's thresholding)
- ✓ 86% of test images with Dice ≥ 0.80 (reliable performance)

## Features

- **U-Net Architecture**: Encoder-decoder with skip connections using pretrained ResNet-34
- **Transfer Learning**: ImageNet-pretrained weights for faster convergence
- **Data Augmentation**: Spatial (flips, rotation, elastic) and pixel-level (brightness/contrast) augmentation
- **Loss Function**: Combined BCEWithDiceLoss for handling class imbalance
- **Baseline Comparison**: Otsu's automatic thresholding for reference
- **Comprehensive Evaluation**: 
  - Quantitative metrics (Dice, IoU)
  - Qualitative analysis (success/failure cases)
  - Overlay visualizations and error analysis
  - Per-image performance distribution

## Dataset

**ISIC 2018 Skin Lesion Dataset**
- Total images: 2,594 dermoscopic images with pixel-level masks
- Train split: 1,594 images
- Validation split: 100 images
- Test split: 1,000 images
- Resolution: Resized to 256×256 pixels
- Task: Binary segmentation (lesion vs. background)

The dataset can be downloaded at: [ISIC 2018 Dataset](https://challenge.isic-archive.com/data/#2018)

## Project Structure

```
skin-lession-segmentation/
├── README.md                                         # This file
├── environment.yml                                   # Conda environment specification (If using conda)
├── requirements.txt                                  # Python requirements
├── LICENSE.txt                                       # Project and dataset license
├── ATTRIBUTION.txt                                   # Dataset attribution
│
├── data/
│   ├── train/
│   │   ├── images/                                   # 1,594 training images
│   │   └── masks/                                    # 1,594 training masks
│   ├── val/
│   │   ├── images/                                   # 100 validation images
│   │   └── masks/                                    # 100 validation masks
│   ├── test/
│   │   ├── images/                                   # 1,000 test images
│   │   └── masks/                                    # 1,000 test masks
│   └── models/
│       ├── best_model.pth                            # Trained model weights
│       └── notebooks/
│           ├── unet_rn34_pipeline.ipynb              # Main training & evaluation notebook                        
│           └── src/
│               ├── dataset.py                        # SegmentationDatasets class
│               ├── loss.py                           # DiceLoss & BCEWithDiceLoss
│               ├── metrics.py                        # Evaluation metrics
│               └── utils.py                          # Utility functions
│
└── train_out/
    ├── training_history.json                         # Training loss/metrics per epoch
    ├── validation_evaluation.png                     # Validation Dice/IoU distributions
    ├── baseline_comparison.png                       # U-Net vs Otsu visual comparison
    ├── metrics_comparison.png                        # Quantitative comparison charts
    ├── test_set_evaluation.png                       # Test set results (Val vs Test)
    ├── test_samples_prediction.png                   # Sample predictions from test set
    ├── qualitative_success_cases.png                 # Top 6 performing predictions
    ├── qualitative_failure_cases.png                 # Bottom 6 predictions with error maps
    ├── qualitative_failure_analysis.png              # Detailed analysis of worst case
    └── qualitative_overlay_gallery.png               # 4×4 gallery with overlays
```

## Installation & Setup

### 1. Clone or Download Repository
```bash
git clone https://github.com/efe-cKia00/deeplearning-skin-lesion-segmentation.git
```
___
### 2. Install Dependencies

**Using Conda (Recommended):**
```bash
conda env create -f environment.yml
conda activate dl-sls
```
**Using Google Colab (VS Code Installation Steps)**
1. Download/install the Google Colab Visual Studio Code extension
2. Open ***unet_rn34_pipeline.ipynb*** and connect to a Kernel ***(Select Kernel > Select Another Kernel > Colab > New Colab Server)***
3. Run the first cell, follow the pop-up instructions to connect your Google Drive, and confirm output:
```
Mounted at /content/drive
Google Drive has been mounted successfully
```
4. Upload the following files to the home directory **(My Drive)** of your Google Drive (Organize accordingly):
```
deeplearning-skin-lession-segmentation/
├── environment.yml                                   # Conda environment specification
├── requirements.txt                                  # Python requirements for Google Colab ONLY
└── data/
    ├── train/
    │   ├── images/                                   # 1,594 training images
    │   └── masks/                                    # 1,594 training masks
    ├── val/
    │   ├── images/                                   # 100 validation images
    │   └── masks/                                    # 100 validation masks
    ├── test/
    │   ├── images/                                   # 1,000 test images
    │   └── masks/                                    # 1,000 test masks
    └── models/
        └── notebooks/
            └── src/
                ├── dataset.py                        # SegmentationDatasets class
                ├── loss.py                           # DiceLoss & BCEWithDiceLoss
                ├── metrics.py                        # Evaluation metrics
                └── utils.py                          # Utility functions
```
5. On the Activity Bar in VS Code, click Colab. Under CONTENTS, click the Open Terminal button right next to the remote device's name. Run the following command:
```
pip install -r ./drive/MyDrive/deeplearning-skin-lession-segmentation/requirements.txt
```
**Using python manually (ONLY if Google Colab or a virtual environment is not being utilized):**
```bash
pip install torch torchvision pytorch-lightning
pip install segmentation-models-pytorch albumentations
pip install opencv-python pillow matplotlib scikit-image
pip install numpy pandas scipy
pip install jupyter jupyterlab
```
___
### 3. Prepare Dataset
**NOTE: If you are using Google Colab, then you may delete the datasets from your local machine. The model would utilize the dataset on your Google Drive**
1. Download ISIC 2018 dataset from [ISIC-Archive](https://challenge.isic-archive.com/data/#2018)
2. Organize into directory structure (as per the project structure):
   ```
   data/
   ├── train/images/ & masks/
   ├── val/images/ & masks/
   └── test/images/ & masks/
   ```
3. Data split is already configured in the notebooks

## Usage

### Running the Main Pipeline

**Execute cells sequentially:**

1. **Cell 1-5**: Setup and imports
2. **Cell 6-7**: Data exploration and sanity check
3. **Cell 8-13**: Model configuration and training
4. **Cell 14-17**: Training visualization
5. **Cell 18-26**: Baseline comparison (Otsu's method)
6. **Cell 27-28**: Validation evaluation
7. **Cell 29-30**: Test set evaluation (final results)
8. **Cell 31**: Qualitative analysis (masks, overlays, failure cases)

### Running Individual Components

**Validate Installation:**
```python
import torch
import segmentation_models_pytorch as smp

# Check CUDA availability
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Load pretrained U-Net
model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Load Trained Model:**
```python
import torch
import segmentation_models_pytorch as smp

# Create model
model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1)

# Load weights
checkpoint_path = 'data/models/checkpoints/best_model.pth'
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Inference on single image
image_tensor = torch.randn(1, 3, 256, 256)  # Batch of 1, RGB, 256×256
with torch.no_grad():
    prediction = torch.sigmoid(model(image_tensor))
    binary_mask = (prediction > 0.5).float()
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Loss Function | BCEWithDiceLoss (α=0.5) |
| Batch Size | 32 |
| Epochs | 50 (early stopping) |
| Metrics | Dice, IoU |
| Device | GPU (Tesla T4) |

## Results

### Quantitative Performance

| Method | Dataset | Dice | IoU |
|--------|---------|------|-----|
| Otsu (Baseline) | Validation | 0.1771 | 0.1122 |
| **U-Net** | **Validation** | **0.9070** | **0.8300** |
| Otsu (Baseline) | Test | 0.1771 | 0.1177 |
| **U-Net** | **Test** | **0.8647** | **0.7787** |

### Performance Distribution (Test Set)

- **Excellent (≥0.90):** 591 images (59.1%)
- **Good (0.80-0.90):** 273 images (27.3%)
- **Fair (0.70-0.80):** 77 images (7.7%)
- **Poor (<0.70):** 59 images (5.9%)

**Mean Dice: 0.8788 | Median: 0.9176 | Std Dev: 0.1276**

### Model Improvement

- **Dice Improvement:** 0.6876 absolute (+389% relative)
- **IoU Improvement:** 0.661 absolute (+561% relative)
- **Generalization:** 4-7% validation-to-test degradation (excellent)

## Key Insights

### Success Cases (Dice >0.99)
- Clear lesion boundaries with high contrast
- Medium-sized lesions with uniform illumination
- Minimal artifacts and well-defined morphology

### Failure Cases (Dice <0.15)
- Very small lesions (model struggles with fine details)
- Ambiguous boundaries with gradual transitions
- Complex morphology with intricate borders
- Potential annotation ambiguity in ground truth

**Model bias:** Conservative segmentation with tendency toward false negatives on difficult cases.

## Output Visualizations

All visualizations are saved to the outputs directory:

1. **Validation & Test Metrics**: Dice/IoU comparisons across methods and datasets
2. **Success Cases**: Top 6 predictions with overlay visualization
3. **Failure Cases**: Bottom 6 predictions with error maps (FP=Red, FN=Blue, Correct=Green)
4. **Overlay Gallery**: 4×4 grid showing performance across difficulty spectrum
5. **Confidence Heatmaps**: Model confidence maps for failure analysis

## Computational Requirements

- **GPU:** NVIDIA A100-SXM4-40GB or equivalent
- **Training Time:** 2-3 hours on GPU
- **Inference Time:** <100ms per image on GPU
- **RAM:** 16GB (for batch loading and processing)
- **Disk Space:** ~5GB (for dataset) + ~100MB (model weights)

## Citation

If you reference this project, please cite:

```bibtex
@misc{efeao2026dlsls,
  title={U-Net for Skin Lesion Segmentation on ISIC 2018},
  author={Efe Awo-Osagie},
  year={2026},
  note={deeplearning-skin-lession-segmentation}
}

@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}

@article{codella2018skin,
  title={Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (ISIC)},
  author={Codella, Noel and Rotemberg, Veronica and Tschandl, Philipp and others},
  journal={arXiv preprint arXiv:1902.03368},
  year={2018}
}
```

## Author(s)
[Efe Awo-Osagie](https://www.linkedin.com/in/efe-awo/)

## License

See [LICENSE.txt](LICENSE.txt) for project license details.

## Dataset Attribution

See [ATTRIBUTION.txt](ATTRIBUTION.txt) for ISIC dataset attribution and terms of use.

## Troubleshooting

### CUDA/GPU Issues
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# If GPU not available, CPU fallback is automatic
# (slower but functional for inference)
```

### Out of Memory
- Reduce batch size in notebook (e.g., 32 → 16)
- Use gradient accumulation for effective larger batches
- Reduce image resolution if necessary

### Missing Dependencies
```bash
# Reinstall environment (IF USING CONDA)

conda remove -n dl-sls --all
conda env create -f environment.yml
conda activate dl-sls
```

### Google Drive Path Issues
Edit the `base_dir` and `out_dir` variable in the notebook *cell 3* to point to your accurate path location:
```python
base_dir = '/path/to/your/google-drive'  # Update this path to reflect your Google Drive
```

## Contact & Support

For questions or issues, refer to:
- Main notebook: `data/models/notebooks/unet_rn34_pipeline.ipynb`
- Final report: `FinalProject_Report.docx`
- Source code: `data/models/notebooks/src/`

## Future Work

- Model ensemble methods for improved robustness
- Uncertainty quantification for clinical decision support
- Attention mechanisms to focus on challenging regions
- Multi-task learning (diagnosis + segmentation)
- External dataset validation for generalization assessment
- Real-time inference optimization (TensorRT, ONNX)

---

**Project Status:** ✅ Complete  
**Last Updated:** May 10, 2026  
**Target Achievement:** ✅ Dice ≥ 0.83 ACHIEVED (0.8647)