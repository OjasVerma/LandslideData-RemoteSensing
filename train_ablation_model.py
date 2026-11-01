
from pathlib import Path
import re
import json

import h5py
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"C:\Users\Ojas\Desktop\RS_Project\Data_Normalized")

TRAIN_IMG = DATA_ROOT / "TrainData" / "img"
TRAIN_MASK = DATA_ROOT / "TrainData" / "mask"

VALID_IMG = DATA_ROOT / "ValidData" / "img"
VALID_MASK = DATA_ROOT / "ValidData" / "mask"

OUTPUT_DIR = DATA_ROOT.parent / "Ablation_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Normalized channel order:
# [B2, B3, B4, B8, B11, B12]

BAND_POSITIONS = {
    "B2": 0,
    "B3": 1,
    "B4": 2,
    "B8": 3,
    "B11": 4,
    "B12": 5
}

MODELS = {
    "M1_RGB": ["B2", "B3", "B4"],

    "M2_RGB_NIR": [
        "B2", "B3", "B4", "B8"
    ],

    "M3_RGB_NIR_SWIR1": [
        "B2", "B3", "B4", "B8", "B11"
    ],

    "M4_ALL_SIX": [
        "B2", "B3", "B4", "B8", "B11", "B12"
    ]
}

# Training sample sizes
POSITIVE_SAMPLES = 100_000
NEGATIVE_SAMPLES = 300_000

# Maximum pixels sampled from each image
POSITIVE_PER_IMAGE = 100
NEGATIVE_PER_IMAGE = 300

RANDOM_SEED = 42

# Random Forest settings
RF_SETTINGS = {
    "n_estimators": 300,
    "max_depth": 25,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# Validation threshold search
THRESHOLDS = np.arange(
    0.05, 0.951, 0.025
)

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def get_file_id(filename):

    match = re.search(r"(\d+)\.h5$", filename)

    return int(match.group(1)) if match else None


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
            f"No matching pairs in {image_dir}"
        )

    return images, masks, common_ids


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
                f"No dataset found in {filepath}"
            )

        return f[datasets[0]][:]


def choose_rows(features, labels, class_value, maximum):

    indices = np.flatnonzero(
        labels == class_value
    )

    if len(indices) == 0:
        return np.empty(
            (0, 6), dtype=np.float32
        )

    if len(indices) > maximum:

        indices = rng.choice(
            indices,
            size=maximum,
            replace=False
        )

    return features[indices]


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

    precision = tp / (tp + fp) if tp + fp else 0

    recall = tp / (tp + fn) if tp + fn else 0

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
# 3. CREATE A FIXED TRAINING PIXEL SAMPLE
# ============================================================

def build_training_sample():

    print("\nBUILDING TRAINING SAMPLE")
    print("=" * 65)

    image_files, mask_files, image_ids = get_file_pairs(
        TRAIN_IMG, TRAIN_MASK
    )

    print("Training image-mask pairs:", len(image_ids))

    positive_parts = []
    negative_parts = []

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
                f"Shape mismatch: image {image_id}"
            )

        if image.shape[-1] != 6:
            raise ValueError(
                f"Expected 6 channels, got {image.shape}"
            )

        if not np.all(np.isfinite(image)):
            raise ValueError(
                f"Non-finite image values: {image_id}"
            )

        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError(
                f"Invalid mask labels: {image_id}"
            )

        features = image.reshape(-1, 6)
        labels = mask.reshape(-1)

        positive = choose_rows(
            features,
            labels,
            class_value=1,
            maximum=POSITIVE_PER_IMAGE
        )

        negative = choose_rows(
            features,
            labels,
            class_value=0,
            maximum=NEGATIVE_PER_IMAGE
        )

        positive_parts.append(positive)
        negative_parts.append(negative)

        if number % 500 == 0:
            print(
                f"Processed {number}/{len(image_ids)} images"
            )

    positive_pool = np.concatenate(
        positive_parts, axis=0
    )

    negative_pool = np.concatenate(
        negative_parts, axis=0
    )

    print("\nCandidate positive pixels:", len(positive_pool))
    print("Candidate negative pixels:", len(negative_pool))

    if len(positive_pool) < POSITIVE_SAMPLES:
        raise RuntimeError(
            "Not enough positive pixels for requested sample."
        )

    if len(negative_pool) < NEGATIVE_SAMPLES:
        raise RuntimeError(
            "Not enough negative pixels for requested sample."
        )

    # Select final fixed sample
    positive_indices = rng.choice(
        len(positive_pool),
        size=POSITIVE_SAMPLES,
        replace=False
    )

    negative_indices = rng.choice(
        len(negative_pool),
        size=NEGATIVE_SAMPLES,
        replace=False
    )

    X_positive = positive_pool[positive_indices]
    X_negative = negative_pool[negative_indices]

    X = np.concatenate(
        [X_positive, X_negative],
        axis=0
    )

    y = np.concatenate([
        np.ones(POSITIVE_SAMPLES, dtype=np.uint8),
        np.zeros(NEGATIVE_SAMPLES, dtype=np.uint8)
    ])

    # Shuffle once. Every model receives this same sample.
    shuffle_indices = rng.permutation(len(y))

    X = X[shuffle_indices]
    y = y[shuffle_indices]

    print("\nFINAL TRAINING SAMPLE")
    print("Total pixels:", len(y))
    print("Landslide pixels:", np.sum(y == 1))
    print("Background pixels:", np.sum(y == 0))

    return X, y


# ============================================================
# 4. COLLECT VALIDATION PREDICTIONS
# ============================================================

def get_validation_predictions(model, selected_bands):

    print("\nCollecting validation predictions...")

    image_files, mask_files, image_ids = get_file_pairs(
        VALID_IMG, VALID_MASK
    )

    selected_indices = [
        BAND_POSITIONS[b]
        for b in selected_bands
    ]

    all_probabilities = []
    all_labels = []

    batch_size = 100_000

    for number, image_id in enumerate(
        image_ids, start=1
    ):

        image = read_h5(
            image_files[image_id], "img"
        )

        mask = read_h5(
            mask_files[image_id], "mask"
        )

        features = image.reshape(
            -1, 6
        )[:, selected_indices]

        labels = mask.reshape(-1).astype(
            np.uint8
        )

        # Predict in batches to limit memory use
        for start in range(
            0, len(labels), batch_size
        ):

            end = min(
                start + batch_size,
                len(labels)
            )

            probabilities = model.predict_proba(
                features[start:end]
            )[:, 1]

            all_probabilities.append(
                probabilities.astype(np.float32)
            )

            all_labels.append(
                labels[start:end]
            )

        if number % 50 == 0:
            print(
                f"Validation images: "
                f"{number}/{len(image_ids)}"
            )

    return (
        np.concatenate(all_probabilities),
        np.concatenate(all_labels)
    )


# ============================================================
# 5. SELECT THE BEST VALIDATION THRESHOLD
# ============================================================

def find_best_threshold(probabilities, y_true):

    best_threshold = 0.5
    best_f1 = -1
    best_metrics = None

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(np.uint8)

        metrics = calculate_metrics(
            y_true,
            predictions
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


# ============================================================
# 6. TRAIN FOUR RANDOM FOREST MODELS
# ============================================================

X_train, y_train = build_training_sample()

results = {}

np.savez_compressed(
    OUTPUT_DIR / "training_sample.npz",
    X_train=X_train,
    y_train=y_train
)

print("Training sample saved successfully.")


for model_name, selected_bands in MODELS.items():

    print("\n" + "=" * 70)
    print("TRAINING:", model_name)
    print("Bands:", selected_bands)
    print("=" * 70)

    selected_indices = [
        BAND_POSITIONS[b]
        for b in selected_bands
    ]

    X_model = X_train[:, selected_indices]

    model = RandomForestClassifier(
        **RF_SETTINGS
    )

    model.fit(
        X_model,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    model_path = (
        OUTPUT_DIR /
        f"{model_name}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print("Saved model:", model_path)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    probabilities, y_valid = get_validation_predictions(
        model,
        selected_bands
    )

    threshold, metrics = find_best_threshold(
        probabilities,
        y_valid
    )

    print("\nVALIDATION RESULTS")
    print("Selected threshold:", threshold)

    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.5f}")

    results[model_name] = {
        "bands": selected_bands,
        "threshold": threshold,
        "validation_metrics": metrics,
        "model_file": model_path.name,
        "rf_settings": RF_SETTINGS
    }

    # Release validation arrays before next model
    del probabilities
    del y_valid
    del model
    del X_model


# ============================================================
# 7. SAVE EXPERIMENT CONFIGURATION
# ============================================================

metadata_path = (
    OUTPUT_DIR /
    "ablation_metadata.json"
)

with open(metadata_path, "w") as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ALL FOUR MODELS TRAINED")
print("=" * 70)

for model_name, result in results.items():

    print("\n", model_name)

    print("Bands:", result["bands"])

    print(
        "Validation threshold:",
        result["threshold"]
    )

    print(
        "Validation F1:",
        round(
            result["validation_metrics"]["f1"],
            5
        )
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nTraining and validation complete.")
