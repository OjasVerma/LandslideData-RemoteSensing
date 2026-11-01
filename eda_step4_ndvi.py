
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

OUTPUT_DIR = DATA_ROOT.parent / "EDA" / "Step4_NDVI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Original 14-channel image, zero-based indices:
RED_INDEX = 3       # B4
NIR_INDEX = 7       # B8

MAX_SAMPLES_PER_CLASS_PER_IMAGE = 300

EPSILON = 1e-8

rng = np.random.default_rng(42)


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
            raise ValueError(
                f"No dataset found in {filepath}"
            )

        return f[datasets[0]][:]


def sample_values(values, maximum):

    if len(values) <= maximum:
        return values

    indices = rng.choice(
        len(values),
        size=maximum,
        replace=False
    )

    return values[indices]


def save_plot(filename):

    path = OUTPUT_DIR / filename

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )

    print("Saved:", path)


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

common_ids = sorted(
    set(image_files) & set(mask_files)
)

if not common_ids:
    raise RuntimeError(
        "No matching image-mask pairs found."
    )

print("Matching image-mask pairs:", len(common_ids))


# ============================================================
# 4. INITIALIZE STORAGE
# ============================================================

# Sampled NDVI values for plotting
sampled_ndvi = {
    0: [],
    1: []
}

# Per-image statistics
image_records = []

# Overall valid NDVI range
global_min = np.inf
global_max = -np.inf

invalid_ndvi_pixels = 0


# ============================================================
# 5. PROCESS ALL TRAINING IMAGES
# ============================================================

for number, sample_id in enumerate(
    common_ids, start=1
):

    image = read_h5(
        image_files[sample_id], "img"
    )

    mask = read_h5(
        mask_files[sample_id], "mask"
    )

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Shape mismatch in image {sample_id}"
        )

    red = image[:, :, RED_INDEX].astype(
        np.float32
    )

    nir = image[:, :, NIR_INDEX].astype(
        np.float32
    )

    if not np.all(np.isfinite(red)):
        raise ValueError(
            f"Invalid B4 values in image {sample_id}"
        )

    if not np.all(np.isfinite(nir)):
        raise ValueError(
            f"Invalid B8 values in image {sample_id}"
        )

    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError(
            f"Non-binary mask in image {sample_id}"
        )

    # --------------------------------------------------------
    # Calculate NDVI
    # --------------------------------------------------------

    denominator = nir + red

    valid = (
        np.isfinite(denominator)
        & (np.abs(denominator) > EPSILON)
    )

    ndvi = np.full(
        red.shape,
        np.nan,
        dtype=np.float32
    )

    ndvi[valid] = (
        (nir[valid] - red[valid])
        / denominator[valid]
    )

    invalid_ndvi_pixels += int(
        np.sum(~valid)
    )

    valid_values = ndvi[valid]

    if len(valid_values) > 0:

        global_min = min(
            global_min,
            float(np.min(valid_values))
        )

        global_max = max(
            global_max,
            float(np.max(valid_values))
        )

    # --------------------------------------------------------
    # Calculate class-wise statistics
    # --------------------------------------------------------

    record = {
        "Image_ID": sample_id
    }

    for class_label in [0, 1]:

        class_name = (
            "Background"
            if class_label == 0
            else "Landslide"
        )

        class_pixels = (
            (mask == class_label) & valid
        )

        values = ndvi[class_pixels]

        record[
            f"{class_name}_Pixels"
        ] = len(values)

        if len(values) > 0:

            record[
                f"{class_name}_NDVI_Mean"
            ] = float(np.mean(values))

            record[
                f"{class_name}_NDVI_Median"
            ] = float(np.median(values))

            record[
                f"{class_name}_NDVI_Std"
            ] = float(np.std(values))

        else:

            record[
                f"{class_name}_NDVI_Mean"
            ] = np.nan

            record[
                f"{class_name}_NDVI_Median"
            ] = np.nan

            record[
                f"{class_name}_NDVI_Std"
            ] = np.nan

        # Random sample for plots
        sampled = sample_values(
            values,
            MAX_SAMPLES_PER_CLASS_PER_IMAGE
        )

        sampled_ndvi[class_label].append(
            sampled
        )

    image_records.append(record)

    if number % 500 == 0:

        print(
            f"Processed {number}/{len(common_ids)} images"
        )


print("\nProcessing complete.")

print("Invalid NDVI pixels:", invalid_ndvi_pixels)

print(
    f"Overall NDVI range: "
    f"{global_min:.5f} to {global_max:.5f}"
)


# ============================================================
# 6. COMBINE SAMPLES
# ============================================================

background_ndvi = np.concatenate(
    sampled_ndvi[0]
)

landslide_ndvi = np.concatenate(
    sampled_ndvi[1]
)

print("\nSampled background NDVI:", len(background_ndvi))
print("Sampled landslide NDVI:", len(landslide_ndvi))


# ============================================================
# 7. OVERALL CLASS STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL NDVI STATISTICS - SAMPLED PIXELS")
print("=" * 70)

for name, values in [
    ("Background", background_ndvi),
    ("Landslide", landslide_ndvi)
]:

    print(f"\n{name}")

    print("Mean:", round(float(np.mean(values)), 5))
    print("Median:", round(float(np.median(values)), 5))
    print("Std:", round(float(np.std(values)), 5))
    print("Min:", round(float(np.min(values)), 5))
    print("Max:", round(float(np.max(values)), 5))


# ============================================================
# 8. CREATE PER-IMAGE DATAFRAME
# ============================================================

df_images = pd.DataFrame(image_records)

valid_images = (
    df_images["Background_NDVI_Mean"].notna()
    & df_images["Landslide_NDVI_Mean"].notna()
)

df_images["NDVI_Mean_Difference"] = (
    df_images["Landslide_NDVI_Mean"]
    - df_images["Background_NDVI_Mean"]
)

df_images["NDVI_Median_Difference"] = (
    df_images["Landslide_NDVI_Median"]
    - df_images["Background_NDVI_Median"]
)


# ============================================================
# 9. PER-IMAGE DIFFERENCE SUMMARY
# ============================================================

mean_diff = df_images.loc[
    valid_images,
    "NDVI_Mean_Difference"
]

median_diff = df_images.loc[
    valid_images,
    "NDVI_Median_Difference"
]

print("\n" + "=" * 70)
print("WITHIN-IMAGE NDVI DIFFERENCE")
print("=" * 70)

print(
    "Mean difference:",
    round(float(mean_diff.mean()), 5)
)

print(
    "Median difference:",
    round(float(mean_diff.median()), 5)
)

print(
    "Images with lower landslide NDVI:",
    round(float((mean_diff < 0).mean() * 100), 2),
    "%"
)

print(
    "Images with higher landslide NDVI:",
    round(float((mean_diff > 0).mean() * 100), 2),
    "%"
)


# ============================================================
# 10. PLOT A: NDVI HISTOGRAM
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    background_ndvi,
    bins=70,
    range=(-1, 1),
    density=True,
    alpha=0.55,
    label="Background"
)

plt.hist(
    landslide_ndvi,
    bins=70,
    range=(-1, 1),
    density=True,
    alpha=0.55,
    label="Landslide"
)

plt.xlabel("NDVI")
plt.ylabel("Density")

plt.title(
    "NDVI Distribution: Landslide vs Background"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

save_plot("A_NDVI_histogram.png")
plt.show()


# ============================================================
# 11. PLOT B: NDVI BOXPLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.boxplot(
    [
        background_ndvi,
        landslide_ndvi
    ],
    tick_labels=[
        "Background",
        "Landslide"
    ],
    showfliers=False
)

plt.ylabel("NDVI")

plt.title(
    "NDVI Boxplot: Landslide vs Background"
)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

save_plot("B_NDVI_boxplot.png")
plt.show()


# ============================================================
# 12. PLOT C: WITHIN-IMAGE NDVI DIFFERENCES
# ============================================================

plt.figure(figsize=(9, 6))

plt.boxplot(
    [
        mean_diff.dropna().values,
        median_diff.dropna().values
    ],
    tick_labels=[
        "Mean difference",
        "Median difference"
    ],
    showfliers=False
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.ylabel(
    "Landslide NDVI - Background NDVI"
)

plt.title(
    "Within-Image NDVI Difference"
)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

save_plot("C_NDVI_within_image_difference.png")
plt.show()


# ============================================================
# 13. SAVE TABLES
# ============================================================

df_images.to_csv(
    OUTPUT_DIR / "per_image_ndvi_statistics.csv",
    index=False
)

summary = pd.DataFrame({
    "Class": ["Background", "Landslide"],
    "Sampled_Mean": [
        np.mean(background_ndvi),
        np.mean(landslide_ndvi)
    ],
    "Sampled_Median": [
        np.median(background_ndvi),
        np.median(landslide_ndvi)
    ],
    "Sampled_Std": [
        np.std(background_ndvi),
        np.std(landslide_ndvi)
    ]
})

summary.to_csv(
    OUTPUT_DIR / "ndvi_class_summary.csv",
    index=False
)


# ============================================================
# 14. FINISH
# ============================================================

print("\n" + "=" * 70)
print("EDA STEP 4 COMPLETE")
print("=" * 70)

print("\nGenerated figures:")
print("A_NDVI_histogram.png")
print("B_NDVI_boxplot.png")
print("C_NDVI_within_image_difference.png")

print("\nGenerated tables:")
print("per_image_ndvi_statistics.csv")
print("ndvi_class_summary.csv")

print("\nOutput directory:")
print(OUTPUT_DIR)
