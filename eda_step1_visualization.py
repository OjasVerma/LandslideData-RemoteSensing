
from pathlib import Path
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

# CHANGE THIS to the location of your original Data directory
DATA_ROOT = Path(r"C:\Users\Ojas\Desktop\RS_Project\Data")

SPLIT = "TrainData"

IMAGE_DIR = DATA_ROOT / SPLIT / "img"
MASK_DIR = DATA_ROOT / SPLIT / "mask"

# Select a specific numeric ID, e.g. 25.
# Keep as None to automatically select the first image
# containing at least one landslide pixel.
SAMPLE_ID = None

# Six selected bands, using zero-based indices
BAND_INDICES = {
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8": 7,
    "B11": 10,
    "B12": 11,
}

OUTPUT_DIR = DATA_ROOT.parent / "EDA"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def get_file_id(filename):
    """
    Extract the trailing numeric ID from a filename.

    Examples:
        image_25.h5 -> 25
        mask_25.h5  -> 25
    """
    match = re.search(r"(\d+)\.h5$", filename)

    if match:
        return int(match.group(1))

    return None


def read_h5_dataset(filepath, preferred_key):
    """Read a dataset from an HDF5 file."""

    with h5py.File(filepath, "r") as f:

        if preferred_key in f:
            return f[preferred_key][:]

        # Fallback: use the first available dataset
        dataset_names = [
            key for key in f.keys()
            if isinstance(f[key], h5py.Dataset)
        ]

        if not dataset_names:
            raise ValueError(f"No dataset found in {filepath}")

        return f[dataset_names[0]][:]


def percentile_stretch(band):
    """
    Stretch a band between its 2nd and 98th percentiles
    for visualization.

    This changes display contrast only.
    It does not modify the original data.
    """

    low, high = np.percentile(band, [2, 98])

    if high <= low:
        return np.zeros_like(band, dtype=np.float32)

    stretched = (band - low) / (high - low)

    return np.clip(stretched, 0, 1)


def make_composite(red, green, blue):
    """Create an RGB composite using individual band stretches."""

    return np.dstack([
        percentile_stretch(red),
        percentile_stretch(green),
        percentile_stretch(blue)
    ])


def overlay_mask(ax, mask):
    """Draw landslide boundaries over an image."""

    if np.any(mask == 1):
        ax.contour(
            mask,
            levels=[0.5],
            colors="red",
            linewidths=0.8
        )


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

print("Matching image-mask pairs:", len(common_ids))


# ============================================================
# 4. SELECT A SAMPLE
# ============================================================

selected_id = None
image = None
mask = None

candidate_ids = (
    [SAMPLE_ID] if SAMPLE_ID is not None else common_ids
)

for sample_id in candidate_ids:

    if sample_id not in image_files or sample_id not in mask_files:
        raise ValueError(f"No matching image-mask pair for ID {sample_id}")

    current_image = read_h5_dataset(
        image_files[sample_id], "img"
    )

    current_mask = read_h5_dataset(
        mask_files[sample_id], "mask"
    )

    if current_image.ndim != 3 or current_image.shape[-1] != 14:
        raise ValueError(
            f"Unexpected image shape: {current_image.shape}"
        )

    if current_mask.ndim != 2:
        raise ValueError(
            f"Unexpected mask shape: {current_mask.shape}"
        )

    if current_image.shape[:2] != current_mask.shape:
        raise ValueError("Image and mask dimensions do not match.")

    if SAMPLE_ID is not None or np.any(current_mask == 1):

        selected_id = sample_id
        image = current_image
        mask = current_mask
        break

if image is None:
    raise RuntimeError("No suitable sample found.")

print("\nSelected sample ID:", selected_id)
print("Original image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask values:", np.unique(mask))


# ============================================================
# 5. EXTRACT THE SIX SELECTED BANDS
# ============================================================

bands = {}

for name, index in BAND_INDICES.items():
    bands[name] = image[:, :, index]

print("\nSelected bands:", list(bands.keys()))


# ============================================================
# 6. PRINT BASIC STATISTICS
# ============================================================

print("\nSPECTRAL BAND STATISTICS")
print("-" * 65)

for name, band in bands.items():

    print(
        f"{name:5s} | "
        f"Min: {np.min(band):.4f} | "
        f"Max: {np.max(band):.4f} | "
        f"Mean: {np.mean(band):.4f} | "
        f"Std: {np.std(band):.4f}"
    )

positive_pixels = np.sum(mask == 1)
total_pixels = mask.size

print("\nMASK STATISTICS")
print("-" * 40)
print("Total pixels:", total_pixels)
print("Landslide pixels:", positive_pixels)
print("Background pixels:", total_pixels - positive_pixels)
print("Landslide percentage:", round(
    100 * positive_pixels / total_pixels, 3
), "%")


# ============================================================
# 7. CREATE THREE SPECTRAL COMPOSITES
# ============================================================

# Natural-colour RGB:
# R = B4, G = B3, B = B2

rgb = make_composite(
    bands["B4"],
    bands["B3"],
    bands["B2"]
)

# NIR false-colour:
# R = B8, G = B4, B = B3

nir_composite = make_composite(
    bands["B8"],
    bands["B4"],
    bands["B3"]
)

# SWIR false-colour:
# R = B12, G = B11, B = B4

swir_composite = make_composite(
    bands["B12"],
    bands["B11"],
    bands["B4"]
)


# ============================================================
# 8. DISPLAY COMPOSITES WITH MASK OVERLAY
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

composites = [
    (rgb, "RGB: B4-B3-B2"),
    (nir_composite, "NIR False Colour: B8-B4-B3"),
    (swir_composite, "SWIR False Colour: B12-B11-B4"),
]

for ax, (composite, title) in zip(axes, composites):

    ax.imshow(composite)
    overlay_mask(ax, mask)

    ax.set_title(title)
    ax.axis("off")

fig.suptitle(
    f"Landslide4Sense Sample {selected_id} - Spectral Composites",
    fontsize=15
)

plt.tight_layout()

composite_path = OUTPUT_DIR / f"sample_{selected_id}_composites.png"

plt.savefig(composite_path, dpi=200, bbox_inches="tight")
plt.show()


# ============================================================
# 9. DISPLAY ALL SIX INDIVIDUAL BANDS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(13, 8))

for ax, (name, band) in zip(axes.flat, bands.items()):

    ax.imshow(
        percentile_stretch(band),
        cmap="gray"
    )

    ax.set_title(name)
    ax.axis("off")

fig.suptitle(
    f"Individual Spectral Bands - Sample {selected_id}",
    fontsize=15
)

plt.tight_layout()

bands_path = OUTPUT_DIR / f"sample_{selected_id}_individual_bands.png"

plt.savefig(bands_path, dpi=200, bbox_inches="tight")
plt.show()


print("\nEDA Step 1 complete.")
print("Saved composite figure:", composite_path)
print("Saved individual-band figure:", bands_path)
