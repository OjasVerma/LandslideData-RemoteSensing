
from pathlib import Path
import re
import json
import gc

import h5py
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. PATHS
# ============================================================

DATA_ROOT = Path(r"C:\Users\Ojas\Desktop\RS_Project\Data_Normalized")

VALID_IMG = DATA_ROOT / "ValidData" / "img"
VALID_MASK = DATA_ROOT / "ValidData" / "mask"

OUTPUT_DIR = DATA_ROOT.parent / "RF_Tuning_M2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PATH = (
    DATA_ROOT.parent
    / "Ablation_Results"
    / "training_sample.npz"
)

# If you saved training_sample.npz elsewhere,
# change SAMPLE_PATH accordingly.


# ============================================================
# 2. EXPERIMENT SETTINGS
# ============================================================

# M2 = B2, B3, B4, B8
M2_INDICES = [0, 1, 2, 3]

RANDOM_SEED = 42

THRESHOLDS = np.arange(
    0.05, 0.951, 0.025
)

# Six configurations
CONFIGURATIONS = {

    "RF_1_baseline": {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
    },

    "RF_2_more_trees": {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
    },

    "RF_3_deeper": {
        "n_estimators": 300,
        "max_depth": 25,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
    },

    "RF_4_deep": {
        "n_estimators": 300,
        "max_depth": 35,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
    },

    "RF_5_larger_leaf": {
        "n_estimators": 300,
        "max_depth": 25,
        "min_samples_leaf": 5,
        "max_features": "sqrt"
    },

    "RF_6_half_features": {
        "n_estimators": 300,
        "max_depth": 25,
        "min_samples_leaf": 2,
        "max_features": 0.5
    }
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_file_id(filename):

    match = re.search(r"(\d+)\.h5$", filename)

    return int(match.group(1)) if match else None


def read_h5(filepath, key):

    with h5py.File(filepath, "r") as f:

        if key in f:
            return f[key][:]

        datasets = [
            name for name in f.keys()
            if isinstance(f[name], h5py.Dataset)
        ]

        if not datasets:
            raise ValueError(
                f"No dataset found: {filepath}"
            )

        return f[datasets[0]][:]


def get_file_pairs(image_dir, mask_dir):

    images = {
        get_file_id(f.name): f
        for f in image_dir.glob("*.h5")
        if get_file_id(f.name) is not None
    }

    masks = {
        get_file_id(f.name): f
        for f in mask_dir.glob("*.h5")
        if get_file_id(f.name) is not None
    }

    common_ids = sorted(
        set(images) & set(masks)
    )

    if not common_ids:
        raise RuntimeError(
            "No matching validation image-mask pairs."
        )

    return images, masks, common_ids


def calculate_metrics(y_true, y_pred):

    tp = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    fp = np.sum(
        (y_true == 0) & (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1) & (y_pred == 0)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou)
    }


# ============================================================
# 4. LOAD FIXED TRAINING SAMPLE
# ============================================================

print("=" * 65)
print("LOADING FIXED TRAINING SAMPLE")
print("=" * 65)

if not SAMPLE_PATH.exists():
    raise FileNotFoundError(
        f"Training sample not found: {SAMPLE_PATH}"
    )

data = np.load(SAMPLE_PATH)

X_train_all = data["X_train"]
y_train = data["y_train"]

del data

if X_train_all.ndim != 2 or X_train_all.shape[1] != 6:
    raise ValueError(
        f"Expected X_train shape (N, 6), got {X_train_all.shape}"
    )

if len(X_train_all) != len(y_train):
    raise ValueError(
        "Training feature and label counts do not match."
    )

if not np.all(np.isfinite(X_train_all)):
    raise ValueError(
        "Non-finite training features found."
    )

print("X_train:", X_train_all.shape)
print("y_train:", y_train.shape)

print("Positive pixels:", np.sum(y_train == 1))
print("Negative pixels:", np.sum(y_train == 0))

# Select M2 features
X_train = X_train_all[:, M2_INDICES]

print("M2 training features:", X_train.shape)

del X_train_all
gc.collect()


# ============================================================
# 5. LOAD VALIDATION DATA
# ============================================================

print("\n" + "=" * 65)
print("LOADING VALIDATION DATA")
print("=" * 65)

image_files, mask_files, image_ids = get_file_pairs(
    VALID_IMG, VALID_MASK
)

print("Validation image-mask pairs:", len(image_ids))

X_valid_parts = []
y_valid_parts = []

for number, image_id in enumerate(
    image_ids, start=1
):

    image = read_h5(
        image_files[image_id], "img"
    )

    mask = read_h5(
        mask_files[image_id], "mask"
    )

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Shape mismatch in image {image_id}"
        )

    if image.shape[-1] != 6:
        raise ValueError(
            f"Expected 6 channels in image {image_id}"
        )

    if not np.all(np.isfinite(image)):
        raise ValueError(
            f"Non-finite values in image {image_id}"
        )

    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError(
            f"Invalid mask values in image {image_id}"
        )

    # Extract only M2 bands
    features = image.reshape(-1, 6)[:, M2_INDICES]

    labels = mask.reshape(-1).astype(np.uint8)

    X_valid_parts.append(features)
    y_valid_parts.append(labels)

    if number % 50 == 0:
        print(
            f"Loaded {number}/{len(image_ids)} images"
        )

X_valid = np.concatenate(
    X_valid_parts, axis=0
)

y_valid = np.concatenate(
    y_valid_parts
)

del X_valid_parts
del y_valid_parts
gc.collect()

print("\nValidation features:", X_valid.shape)
print("Validation labels:", y_valid.shape)
print("Validation landslide pixels:", np.sum(y_valid == 1))
print("Validation background pixels:", np.sum(y_valid == 0))


# ============================================================
# 6. TRAIN AND EVALUATE CONFIGURATIONS
# ============================================================

results = []

best_f1 = -1
best_model = None
best_config_name = None
best_threshold = None

for config_name, params in CONFIGURATIONS.items():

    print("\n" + "=" * 70)
    print("TRAINING:", config_name)
    print("=" * 70)

    print("Parameters:", params)

    model = RandomForestClassifier(
        **params,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    print("Generating validation probabilities...")

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    # --------------------------------------------------------
    # Find best threshold using validation F1
    # --------------------------------------------------------

    config_best_f1 = -1
    config_best_threshold = None
    config_best_metrics = None

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(np.uint8)

        metrics = calculate_metrics(
            y_valid,
            predictions
        )

        if metrics["f1"] > config_best_f1:

            config_best_f1 = metrics["f1"]
            config_best_threshold = float(threshold)
            config_best_metrics = metrics

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    result = {
        "configuration": config_name,
        **params,
        "threshold": config_best_threshold,
        **config_best_metrics
    }

    results.append(result)

    print("\nBEST VALIDATION RESULT")

    for key, value in result.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        OUTPUT_DIR /
        f"{config_name}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print("Saved:", model_path)

    # --------------------------------------------------------
    # Track best model
    # --------------------------------------------------------

    if config_best_f1 > best_f1:

        best_f1 = config_best_f1
        best_model = model
        best_config_name = config_name
        best_threshold = config_best_threshold

    del probabilities
    del model
    gc.collect()


# ============================================================
# 7. SAVE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1",
    ascending=False
)

csv_path = (
    OUTPUT_DIR /
    "M2_hyperparameter_comparison.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

print("\n" + "=" * 70)
print("HYPERPARAMETER COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# ============================================================
# 8. SAVE BEST CONFIGURATION
# ============================================================

best_row = results_df.iloc[0].to_dict()

best_metadata = {
    "model": best_config_name,
    "bands": ["B2", "B3", "B4", "B8"],
    "threshold": float(best_threshold),
    "validation_metrics": {
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
        "iou": float(best_row["iou"])
    },
    "parameters": {
        key: (
            None if pd.isna(value) else value
        )
        for key, value in best_row.items()
        if key not in [
            "configuration",
            "threshold",
            "precision",
            "recall",
            "f1",
            "iou"
        ]
    }
}

with open(
    OUTPUT_DIR / "best_M2_configuration.json",
    "w"
) as f:

    json.dump(
        best_metadata,
        f,
        indent=4
    )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("M2 TUNING COMPLETE")
print("=" * 70)

print("Best configuration:", best_config_name)
print("Best threshold:", best_threshold)
print("Best validation F1:", best_f1)

print("\nComparison CSV:", csv_path)

print("\nAll models saved in:")
print(OUTPUT_DIR)
