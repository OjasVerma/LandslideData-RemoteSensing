
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

OUTPUT_DIR = DATA_ROOT.parent / "EDA" / "Step3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Six selected bands (zero-based indices)
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

# Maximum pixels sampled per class per image
# Used for plotting only, not for calculating per-image means.
MAX_SAMPLES_PER_CLASS = 300

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
            raise ValueError(f"No dataset found: {filepath}")

        return f[datasets[0]][:]


def sample_pixels(pixels, maximum):

    if len(pixels) <= maximum:
        return pixels

    indices = rng.choice(
        len(pixels),
        size=maximum,
        replace=False
    )

    return pixels[indices]


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

common_ids = sorted(set(image_files) & set(mask_files))

if not common_ids:
    raise RuntimeError("No matching image-mask pairs found.")

print("Matched image-mask pairs:", len(common_ids))


# ============================================================
# 4. INITIALIZE STORAGE
# ============================================================

# Sampled pixel values for plots
# Class 0 = background
# Class 1 = landslide

sampled_pixels = {
    0: [],
    1: []
}

# Per-image class means
image_records = []

# Total pixel counts
total_pixels = {
    0: 0,
    1: 0
}


# ============================================================
# 5. PROCESS ALL TRAINING IMAGES
# ============================================================

for number, sample_id in enumerate(common_ids, start=1):

    image = read_h5(
        image_files[sample_id], "img"
    )

    mask = read_h5(
        mask_files[sample_id], "mask"
    )

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Dimension mismatch in image {sample_id}"
        )

    selected_image = image[:, :, CHANNELS]

    if not np.all(np.isfinite(selected_image)):
        raise ValueError(
            f"Non-finite values in image {sample_id}"
        )

    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError(
            f"Non-binary mask in image {sample_id}"
        )

    record = {
        "Image_ID": sample_id,
        "Landslide_Pixels": int(np.sum(mask == 1)),
        "Background_Pixels": int(np.sum(mask == 0)),
    }

    # --------------------------------------------------------
    # Separate classes
    # --------------------------------------------------------

    for class_label in [0, 1]:

        pixels = selected_image[
            mask == class_label
        ].astype(np.float32)

        total_pixels[class_label] += len(pixels)

        # Exact class mean within this image
        if len(pixels) > 0:

            class_mean = np.mean(
                pixels,
                axis=0
            )

        else:

            class_mean = np.full(
                len(BAND_NAMES),
                np.nan
            )

        # Store per-image class means
        prefix = (
            "Background" if class_label == 0
            else "Landslide"
        )

        for i, band in enumerate(BAND_NAMES):

            record[f"{prefix}_{band}"] = class_mean[i]

        # Store a random sample for plots
        sampled = sample_pixels(
            pixels,
            MAX_SAMPLES_PER_CLASS
        )

        sampled_pixels[class_label].append(sampled)

    image_records.append(record)

    if number % 500 == 0:

        print(
            f"Processed {number}/{len(common_ids)} images"
        )


print("\nProcessing complete.")

print("Total background pixels:", total_pixels[0])
print("Total landslide pixels:", total_pixels[1])


# ============================================================
# 6. COMBINE SAMPLED PIXELS
# ============================================================

background = np.concatenate(
    sampled_pixels[0],
    axis=0
)

landslide = np.concatenate(
    sampled_pixels[1],
    axis=0
)

print("\nSampled background pixels:", len(background))
print("Sampled landslide pixels:", len(landslide))


# ============================================================
# 7. CREATE DATAFRAME
# ============================================================

df_background = pd.DataFrame(
    background,
    columns=BAND_NAMES
)

df_background["Class"] = "Background"


df_landslide = pd.DataFrame(
    landslide,
    columns=BAND_NAMES
)

df_landslide["Class"] = "Landslide"


df_pixels = pd.concat(
    [df_background, df_landslide],
    ignore_index=True
)


df_images = pd.DataFrame(image_records)


# ============================================================
# 8. CALCULATE PER-IMAGE DIFFERENCES
# ============================================================

difference_records = []

for band in BAND_NAMES:

    background_col = f"Background_{band}"
    landslide_col = f"Landslide_{band}"

    valid = (
        df_images[background_col].notna()
        & df_images[landslide_col].notna()
    )

    differences = (
        df_images.loc[valid, landslide_col]
        - df_images.loc[valid, background_col]
    )

    for difference in differences:

        difference_records.append({
            "Band": band,
            "Difference": difference
        })


df_differences = pd.DataFrame(difference_records)


print("\nPER-IMAGE DIFFERENCE SUMMARY")
print("-" * 70)

for band in BAND_NAMES:

    values = df_differences.loc[
        df_differences["Band"] == band,
        "Difference"
    ]

    print(
        f"{band:5s} | "
        f"Mean difference: {values.mean():+.4f} | "
        f"Median difference: {values.median():+.4f} | "
        f"Positive images: {(values > 0).mean() * 100:.1f}%"
    )


# ============================================================
# 9. PLOT A: BOXPLOTS
# ============================================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 9)
)

for ax, band in zip(axes.flat, BAND_NAMES):

    values = [
        df_background[band].values,
        df_landslide[band].values
    ]

    ax.boxplot(
        values,
        tick_labels=["Background", "Landslide"],
        showfliers=False
    )

    ax.set_title(band)
    ax.set_ylabel("Original dataset value")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Spectral Distribution: Landslide vs Background",
    fontsize=15
)

plt.tight_layout()

save_plot("A_boxplots.png")
plt.show()


# ============================================================
# 10. PLOT B: OVERLAPPING HISTOGRAMS
# ============================================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 9)
)

for ax, band in zip(axes.flat, BAND_NAMES):

    bg_values = df_background[band].values
    ls_values = df_landslide[band].values

    # Common display range based on pooled percentiles
    pooled = np.concatenate([
        bg_values,
        ls_values
    ])

    low, high = np.percentile(
        pooled,
        [1, 99]
    )

    ax.hist(
        bg_values,
        bins=60,
        range=(low, high),
        density=True,
        alpha=0.55,
        label="Background"
    )

    ax.hist(
        ls_values,
        bins=60,
        range=(low, high),
        density=True,
        alpha=0.55,
        label="Landslide"
    )

    ax.set_title(band)
    ax.set_xlabel("Original dataset value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

fig.suptitle(
    "Spectral Histograms: Class Overlap",
    fontsize=15
)

plt.tight_layout()

save_plot("B_histograms.png")
plt.show()


# ============================================================
# 11. PLOT C: TWO-BAND SCATTER PLOTS
# ============================================================

# Select at most 10,000 pixels from each class
# for readable scatter plots.

def scatter_sample(data, maximum=10000):

    if len(data) <= maximum:
        return data

    indices = rng.choice(
        len(data),
        size=maximum,
        replace=False
    )

    return data[indices]


bg_scatter = scatter_sample(background)
ls_scatter = scatter_sample(landslide)


fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6)
)


# ------------------------------------------------------------
# B4 vs B12
# ------------------------------------------------------------

axes[0].scatter(
    bg_scatter[:, 2],
    bg_scatter[:, 5],
    s=5,
    alpha=0.25,
    label="Background"
)

axes[0].scatter(
    ls_scatter[:, 2],
    ls_scatter[:, 5],
    s=5,
    alpha=0.25,
    label="Landslide"
)

axes[0].set_xlabel("B4 (Red)")
axes[0].set_ylabel("B12 (SWIR2)")
axes[0].set_title("B4 vs B12")
axes[0].legend()
axes[0].grid(alpha=0.2)


# ------------------------------------------------------------
# B4 vs B8
# ------------------------------------------------------------

axes[1].scatter(
    bg_scatter[:, 2],
    bg_scatter[:, 3],
    s=5,
    alpha=0.25,
    label="Background"
)

axes[1].scatter(
    ls_scatter[:, 2],
    ls_scatter[:, 3],
    s=5,
    alpha=0.25,
    label="Landslide"
)

axes[1].set_xlabel("B4 (Red)")
axes[1].set_ylabel("B8 (NIR)")
axes[1].set_title("B4 vs B8")
axes[1].legend()
axes[1].grid(alpha=0.2)


fig.suptitle(
    "Two-Band Spectral Separability",
    fontsize=15
)

plt.tight_layout()

save_plot("C_scatterplots.png")
plt.show()


# ============================================================
# 12. PLOT D: PER-IMAGE DIFFERENCE BOXPLOTS
# ============================================================

difference_data = [
    df_differences.loc[
        df_differences["Band"] == band,
        "Difference"
    ].values
    for band in BAND_NAMES
]

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.boxplot(
    difference_data,
    tick_labels=BAND_NAMES,
    showfliers=False
)

# Zero reference line
ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_title(
    "Within-Image Spectral Mean Difference"
)

ax.set_xlabel("Spectral band")
ax.set_ylabel(
    "Landslide mean - Background mean"
)

ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

save_plot("D_per_image_differences.png")
plt.show()


# ============================================================
# 13. SAVE TABLES
# ============================================================

df_images.to_csv(
    OUTPUT_DIR / "per_image_spectral_means.csv",
    index=False
)

df_differences.to_csv(
    OUTPUT_DIR / "per_image_spectral_differences.csv",
    index=False
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA STEP 3 COMPLETE")
print("=" * 70)

print("\nGenerated figures:")
print("A_boxplots.png")
print("B_histograms.png")
print("C_scatterplots.png")
print("D_per_image_differences.png")

print("\nGenerated tables:")
print("per_image_spectral_means.csv")
print("per_image_spectral_differences.csv")

print("\nOutput directory:")
print(OUTPUT_DIR)
