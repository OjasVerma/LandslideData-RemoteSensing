
from pathlib import Path
import re
import json

import h5py
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Ojas\Desktop\RS_Project"
)

DATA_ROOT = PROJECT_ROOT / "Data_Normalized"

TEST_IMG_DIR = DATA_ROOT / "TestData" / "img"
TEST_MASK_DIR = DATA_ROOT / "TestData" / "mask"

MODEL_DIR = PROJECT_ROOT / "Ablation_Results"

OUTPUT_DIR = PROJECT_ROOT / "Final_Test_Results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FROZEN MODELS, BANDS AND VALIDATION THRESHOLDS
# ============================================================

BAND_NAMES = [
    "B2", "B3", "B4", "B8", "B11", "B12"
]

MODELS = {
    "M1_RGB": {
        "file": "M1_RGB.joblib",
        "bands": [0, 1, 2],
        "threshold": 0.825
    },

    "M2_RGB_NIR": {
        "file": "M2_RGB_NIR.joblib",
        "bands": [0, 1, 2, 3],
        "threshold": 0.900
    },

    "M3_RGB_NIR_SWIR1": {
        "file": "M3_RGB_NIR_SWIR1.joblib",
        "bands": [0, 1, 2, 3, 4],
        "threshold": 0.875
    },

    "M4_ALL_SIX": {
        "file": "M4_ALL_SIX.joblib",
        "bands": [0, 1, 2, 3, 4, 5],
        "threshold": 0.850
    }
}


# ============================================================
# 3. PAIR IMAGE AND MASK FILES USING THEIR NUMERIC IDs
# ============================================================

def extract_id(path):

    match = re.search(r"(\d+)$", path.stem)

    if match is None:
        raise ValueError(
            f"Could not extract numeric ID from {path.name}"
        )

    return match.group(1)


image_files = {
    extract_id(p): p
    for p in TEST_IMG_DIR.glob("*.h5")
}

mask_files = {
    extract_id(p): p
    for p in TEST_MASK_DIR.glob("*.h5")
}

common_ids = sorted(
    set(image_files) & set(mask_files),
    key=int
)

if len(common_ids) != 800:
    raise ValueError(
        f"Expected 800 test pairs, found {len(common_ids)}. "
        "Check image-mask pairing before evaluation."
    )

if set(image_files) != set(mask_files):
    raise ValueError(
        "Image-mask ID mismatch detected. "
        "Check the test directories."
    )

print("Test image-mask pairs:", len(common_ids))


# ============================================================
# 4. LOAD ALL FOUR TRAINED MODELS
# ============================================================

loaded_models = {}

for name, config in MODELS.items():

    model_path = MODEL_DIR / config["file"]

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    loaded_models[name] = joblib.load(model_path)

    print("Loaded:", name)


# ============================================================
# 5. METRIC CALCULATION
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = TP / (TP + FP) if TP + FP else 0.0

    recall = TP / (TP + FN) if TP + FN else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    iou = (
        TP / (TP + FP + FN)
        if TP + FP + FN else 0.0
    )

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou
    }


# ============================================================
# 6. INITIALIZE RESULT STORAGE
# ============================================================

total_counts = {
    name: {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0
    }
    for name in MODELS
}

per_image_results = []


# ============================================================
# 7. FINAL TEST EVALUATION
# ============================================================

for image_number, image_id in enumerate(common_ids, start=1):

    image_path = image_files[image_id]
    mask_path = mask_files[image_id]

    # Read normalized image and reference mask
    with h5py.File(image_path, "r") as f:
        image = f["img"][:]

    with h5py.File(mask_path, "r") as f:
        mask = f["mask"][:]

    if image.ndim != 3 or image.shape[-1] != 6:
        raise ValueError(
            f"Unexpected image shape in {image_path.name}: "
            f"{image.shape}"
        )

    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"Image-mask dimension mismatch: {image_path.name}"
        )

    if not np.all(np.isfinite(image)):
        raise ValueError(
            f"Non-finite values found in {image_path.name}"
        )

    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError(
            f"Unexpected mask labels in {mask_path.name}"
        )

    y_true = mask.astype(np.uint8).ravel()

    # Evaluate each model on the same test image
    for name, config in MODELS.items():

        model = loaded_models[name]

        selected_bands = config["bands"]
        threshold = config["threshold"]

        # Select the correct spectral bands
        X = image[:, :, selected_bands]

        # Flatten image into pixel feature vectors
        X = X.reshape(-1, len(selected_bands))

        # Locate the probability column corresponding to class 1
        positive_class_index = np.where(
            model.classes_ == 1
        )[0]

        if len(positive_class_index) != 1:
            raise ValueError(
                f"Model {name} does not contain class 1."
            )

        positive_class_index = positive_class_index[0]

        probabilities = model.predict_proba(X)[
            :, positive_class_index
        ]

        # Apply the frozen validation threshold
        y_pred = (probabilities >= threshold).astype(np.uint8)

        # Accumulate global confusion counts
        counts = calculate_metrics(y_true, y_pred)

        for key in ["TP", "TN", "FP", "FN"]:
            total_counts[name][key] += counts[key]

        # Save predicted map
        prediction_map = y_pred.reshape(mask.shape)

        map_dir = OUTPUT_DIR / "predicted_maps" / name
        map_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            map_dir / f"image_{image_id}_prediction.npy",
            prediction_map
        )

        # Per-image metrics for later error analysis
        per_image_results.append({
            "image_id": image_id,
            "model": name,
            "TP": counts["TP"],
            "TN": counts["TN"],
            "FP": counts["FP"],
            "FN": counts["FN"],
            "Precision": counts["Precision"],
            "Recall": counts["Recall"],
            "F1": counts["F1"],
            "IoU": counts["IoU"]
        })

    if image_number % 50 == 0:
        print(
            f"Evaluated {image_number}/{len(common_ids)} images"
        )


# ============================================================
# 8. FINAL AGGREGATED TEST METRICS
# ============================================================

final_results = []

for name, counts in total_counts.items():

    TP = counts["TP"]
    TN = counts["TN"]
    FP = counts["FP"]
    FN = counts["FN"]

    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    iou = (
        TP / (TP + FP + FN)
        if TP + FP + FN else 0.0
    )

    final_results.append({
        "Model": name,
        "Bands": ", ".join(
            BAND_NAMES[i] for i in MODELS[name]["bands"]
        ),
        "Threshold": MODELS[name]["threshold"],
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou
    })


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(final_results)

results_df.to_csv(
    OUTPUT_DIR / "final_test_metrics.csv",
    index=False
)

per_image_df = pd.DataFrame(per_image_results)

per_image_df.to_csv(
    OUTPUT_DIR / "per_image_metrics.csv",
    index=False
)

confusion_df = results_df[
    ["Model", "TP", "FP", "TN", "FN"]
]

confusion_df.to_csv(
    OUTPUT_DIR / "confusion_matrices.csv",
    index=False
)

# Save the exact thresholds used
with open(
    OUTPUT_DIR / "test_evaluation_metadata.json",
    "w"
) as f:

    json.dump(
        {
            name: {
                "bands": [
                    BAND_NAMES[i]
                    for i in config["bands"]
                ],
                "threshold": config["threshold"]
            }
            for name, config in MODELS.items()
        },
        f,
        indent=4
    )


# ============================================================
# 10. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

print(
    results_df[
        ["Model", "Threshold", "Precision", "Recall", "F1", "IoU"]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nConfusion Matrices:")
print(
    confusion_df.to_string(index=False)
)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nFINAL TEST EVALUATION COMPLETE")
