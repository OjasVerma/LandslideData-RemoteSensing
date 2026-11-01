
LANDSLIDE4SENSE PROJECT
Final Research and Reproducibility Archive

Created: 2026-11-01 15:44:11

PROJECT
Contribution of RGB, NIR and SWIR Bands to Landslide Mapping
Sentinel-2 Spectral Band Ablation and Landslide Classification

DATASET
Landslide4Sense

Selected spectral bands:
B2, B3, B4, B8, B11, B12

RANDOM FOREST EXPERIMENTS
M1: B2, B3, B4
M2: B2, B3, B4, B8
M3: B2, B3, B4, B8, B11
M4: B2, B3, B4, B8, B11, B12

DEEP LEARNING EXTENSION
U-Net using all six selected bands.

EVALUATION
Independent test set:
800 patches

Metrics:
Precision, Recall, F1-score and IoU.

CONTENTS
Models:
Trained RF models and U-Net checkpoint.

Predictions:
Saved predictions and ground-truth masks.

Figures:
Prediction maps, spatial error maps, confusion matrices,
metric comparison and any additional completed analysis.

Results:
CSV files containing numerical evaluation results.

Data_Information:
Preprocessing and dataset information, when available.

Notebook:
Complete executable Colab notebook, if supplied.

IMPORTANT
The original Landslide4Sense dataset is not included.
The normalized image and mask dataset is also not included.
Download the original dataset separately from:
https://zenodo.org/records/10463239

Missing files are listed in the packaging output.
Check that all required analysis outputs have been generated
before treating this archive as the final research submission.
