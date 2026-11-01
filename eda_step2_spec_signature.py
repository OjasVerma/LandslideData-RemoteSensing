
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

# Change this to your original Data directory
DATA_ROOT = Path(r"C:\Users\Ojas\Desktop\RS_Project\Data")

SPLIT = "TrainData"

IMAGE_DIR = DATA_ROOT / SPLIT / "img"
MASK_DIR = DATA_ROOT / SPLIT / "mask"

OUTPUT_DIR = DATA_ROOT.parent / "EDA"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Selected bands: zero-based channel indices
BAND_INDICES = {
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8": 7,
    "B11": 10,
    "B12": 11,
}

BAND_NAMES = list(BAND_INDICES.keys())
CHANNELS = list(BAND_INDICES.values())

# Number of pixels sampled per class per image
# Used only for approximate median and quartiles.
MAX_SAMPLES_PER_CLASS_PER_IMAGE = 200

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 2. HELPER FUNCTIONS
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
            raise ValueError(f"No dataset found in {filepath}")

        return f[datasets[0]][:]


def create_statistics_container():

    return {
        0: {
            "count": 0,
            "sum": np.zeros(len(BAND_NAMES), dtype=np.float64),
            "sum_sq": np.zeros(len(BAND_NAMES), dtype=np.float64),
        },
        1: {
            "count": 0,
            "sum": np.zeros(len(BAND_NAMES), dtype=np.float64),
            "sum_sq": np.zeros(len(BAND_NAMES), dtype=np.float64),
        }
    }


def update_statistics(container, class_label, pixels):

    pixels = pixels.astype(np.float64)

    container[class_label]["count"] += len(pixels)
    container[class_label]["sum"] += np.sum(pixels, axis=0)
    container[class_label]["sum_sq"] += np.sum(
        pixels ** 2, axis=0
    )


def calculate_mean_std(container, class_label):

    data = container[class_label]

    count = data["count"]

    if count == 0:
        return None, None

    mean = data["sum"] / count

    variance = data["sum_sq"] / count - mean ** 2
    variance = np.maximum(variance, 0)

    std = np.sqrt(variance)

    return mean, std


def sample_pixels(pixels, maximum):

    if len(pixels) <= maximum:
        return pixels

    indices = rng.choice(
        len(pixels),
        size=maximum,
        replace=False
    )

    return pixels[indices]


# ============================================================
# 3. MATCH IMAGE AND MASK FILES
# ============================================================

image_files = {
    get_file_id(f.name): f
    for f in IMAGE_DIR.glob("*.h5")
    if get_file_id(f.name) is not None
}

mask_files = {
    get_file_id(f.name): f
    for f in MASK_DIR.glob("*.h5")
    if get_file_id(f.name) is not None
}

common_ids = sorted(set(image_files) & set(mask_files))

if not common_ids:
    raise RuntimeError("No matching image-mask pairs found.")

print("Matched training images:", len(common_ids))


# ============================================================
# 4. INITIALIZE ACCUMULATORS
# ============================================================

# Exact statistics across all training pixels
train_stats = create_statistics_container()

# Sampled pixel values for approximate quantiles
sampled_values = {
    0: [],
    1: []
}

# Exact statistics for Sample 1
sample1_stats = create_statistics_container()


# ============================================================
# 5. PROCESS EVERY TRAINING IMAGE
# ============================================================

for image_number, sample_id in enumerate(common_ids, start=1):

    image = read_h5(image_files[sample_id], "img")
    mask = read_h5(mask_files[sample_id], "mask")

    if image.shape[:2] != mask.shape:
        raise ValueError(f"Dimension mismatch: image {sample_id}")

    # Extract only the six selected bands
    selected_image = image[:, :, CHANNELS]

    if not np.all(np.isfinite(selected_image)):
        raise ValueError(
            f"Non-finite spectral values in image {sample_id}"
        )

    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError(
            f"Non-binary mask in image {sample_id}"
        )

    # Separate the two pixel classes
    for class_label in [0, 1]:

        class_pixels = selected_image[
            mask == class_label
        ]

        # Update exact training statistics
        update_statistics(
            train_stats,
            class_label,
            class_pixels
        )

        # Keep a manageable random sample for quantiles
        selected_pixels = sample_pixels(
            class_pixels,
            MAX_SAMPLES_PER_CLASS_PER_IMAGE
        )

        sampled_values[class_label].append(
            selected_pixels.astype(np.float32)
        )

        # Also calculate exact statistics for Sample 1
        if sample_id == 1:

            update_statistics(
                sample1_stats,
                class_label,
                class_pixels
            )

    if image_number % 500 == 0:
        print(
            f"Processed {image_number}/{len(common_ids)} images"
        )


print("\nAll training images processed.")


# ============================================================
# 6. CALCULATE SAMPLE 1 AND TRAINING STATISTICS
# ============================================================

class_names = {
    0: "Non-landslide",
    1: "Landslide"
}

results = []

for class_label in [0, 1]:

    train_mean, train_std = calculate_mean_std(
        train_stats, class_label
    )

    sample_mean, sample_std = calculate_mean_std(
        sample1_stats, class_label
    )

    # Combine sampled pixels across all images
    sampled = np.concatenate(
        sampled_values[class_label],
        axis=0
    )

    print("\n" + "=" * 75)
    print(class_names[class_label].upper())
    print("=" * 75)

    print(
        "Training pixels:",
        train_stats[class_label]["count"]
    )

    print(
        "Sampled pixels for quantiles:",
        len(sampled)
    )

    for i, band_name in enumerate(BAND_NAMES):

        q25, median, q75 = np.percentile(
            sampled[:, i],
            [25, 50, 75]
        )

        print(
            f"{band_name:5s} | "
            f"Mean: {train_mean[i]:.4f} | "
            f"Std: {train_std[i]:.4f} | "
            f"Median*: {median:.4f} | "
            f"IQR*: {q25:.4f} - {q75:.4f}"
        )

        results.append({
            "Class": class_names[class_label],
            "Band": band_name,
            "Pixel_Count": train_stats[class_label]["count"],
            "Mean": train_mean[i],
            "Std": train_std[i],
            "Median_Sampled": median,
            "Q25_Sampled": q25,
            "Q75_Sampled": q75,
        })

    print("\nSample 1 mean:", np.round(sample_mean, 4))
    print("Sample 1 std: ", np.round(sample_std, 4))


# ============================================================
# 7. CALCULATE CLASS DIFFERENCES
# ============================================================

mean_background, std_background = calculate_mean_std(
    train_stats, 0
)

mean_landslide, std_landslide = calculate_mean_std(
    train_stats, 1
)

print("\n" + "=" * 75)
print("LANDSLIDE MINUS NON-LANDSLIDE MEAN DIFFERENCE")
print("=" * 75)

for i, band_name in enumerate(BAND_NAMES):

    difference = mean_landslide[i] - mean_background[i]

    print(
        f"{band_name:5s} | "
        f"Background: {mean_background[i]:.4f} | "
        f"Landslide: {mean_landslide[i]:.4f} | "
        f"Difference: {difference:+.4f}"
    )


# ============================================================
# 8. SAVE STATISTICS AS CSV
# ============================================================

df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "spectral_signature_statistics.csv"

df.to_csv(csv_path, index=False)

print("\nStatistics saved:", csv_path)


# ============================================================
# 9. PLOT SPECTRAL SIGNATURES
# ============================================================

sample1_mean_0, sample1_std_0 = calculate_mean_std(
    sample1_stats, 0
)

sample1_mean_1, sample1_std_1 = calculate_mean_std(
    sample1_stats, 1
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

x = np.arange(len(BAND_NAMES))


# ------------------------------------------------------------
# LEFT: SAMPLE 1
# ------------------------------------------------------------

axes[0].errorbar(
    x,
    sample1_mean_0,
    yerr=sample1_std_0,
    marker="o",
    capsize=4,
    label="Non-landslide"
)

axes[0].errorbar(
    x,
    sample1_mean_1,
    yerr=sample1_std_1,
    marker="o",
    capsize=4,
    label="Landslide"
)

axes[0].set_title("Sample 1")
axes[0].set_xticks(x)
axes[0].set_xticklabels(BAND_NAMES)
axes[0].set_xlabel("Spectral bands")
axes[0].set_ylabel("Original dataset value")
axes[0].legend()
axes[0].grid(alpha=0.3)


# ------------------------------------------------------------
# RIGHT: ALL TRAINING IMAGES
# ------------------------------------------------------------

axes[1].errorbar(
    x,
    mean_background,
    yerr=std_background,
    marker="o",
    capsize=4,
    label="Non-landslide"
)

axes[1].errorbar(
    x,
    mean_landslide,
    yerr=std_landslide,
    marker="o",
    capsize=4,
    label="Landslide"
)

axes[1].set_title("Entire Training Dataset")
axes[1].set_xticks(x)
axes[1].set_xticklabels(BAND_NAMES)
axes[1].set_xlabel("Spectral bands")
axes[1].set_ylabel("Original dataset value")
axes[1].legend()
axes[1].grid(alpha=0.3)


fig.suptitle(
    "Landslide vs Non-landslide Spectral Signatures",
    fontsize=15
)

plt.tight_layout()

plot_path = OUTPUT_DIR / "spectral_signatures.png"

plt.savefig(
    plot_path,
    dpi=250,
    bbox_inches="tight"
)

plt.show()


print("\nEDA STEP 2 COMPLETE")
print("Plot:", plot_path)
print("CSV: ", csv_path)
