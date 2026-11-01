
from pathlib import Path
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

DATA_ROOT = Path(r"C:\Users\Ojas\Desktop\RS_Project\Data")

SPLIT = "TrainData"

IMAGE_DIR = DATA_ROOT / SPLIT / "img"
MASK_DIR = DATA_ROOT / SPLIT / "mask"

OUTPUT_DIR = DATA_ROOT.parent / "EDA" / "Step5_Visual_Check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose the image ID you want to inspect
SAMPLE_ID = 1

# Original 14-channel image, zero-based indices
BAND_INDEX = {
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8": 7,
    "B11": 10,
    "B12": 11,
}


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


def stretch_band(band, lower=2, upper=98):

    """
    Percentile stretch for visualization only.
    Does not modify the original data.
    """

    low, high = np.percentile(
        band,
        [lower, upper]
    )

    if high <= low:
        return np.zeros_like(band, dtype=np.float32)

    stretched = (
        (band - low) / (high - low)
    )

    return np.clip(stretched, 0, 1)


def make_rgb(red, green, blue):

    """
    Independently stretch each channel
    and combine into an RGB image.
    """

    return np.dstack([
        stretch_band(red),
        stretch_band(green),
        stretch_band(blue)
    ])


# ============================================================
# 3. FIND THE IMAGE AND MASK
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

if SAMPLE_ID not in image_files:
    raise FileNotFoundError(
        f"Image ID {SAMPLE_ID} not found."
    )

if SAMPLE_ID not in mask_files:
    raise FileNotFoundError(
        f"Mask ID {SAMPLE_ID} not found."
    )


# ============================================================
# 4. LOAD ORIGINAL IMAGE AND MASK
# ============================================================

image = read_h5(
    image_files[SAMPLE_ID], "img"
)

mask = read_h5(
    mask_files[SAMPLE_ID], "mask"
)

print("Image ID:", SAMPLE_ID)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask values:", np.unique(mask))


# ============================================================
# 5. EXTRACT BANDS
# ============================================================

B2 = image[:, :, BAND_INDEX["B2"]]
B3 = image[:, :, BAND_INDEX["B3"]]
B4 = image[:, :, BAND_INDEX["B4"]]

B8 = image[:, :, BAND_INDEX["B8"]]

B11 = image[:, :, BAND_INDEX["B11"]]
B12 = image[:, :, BAND_INDEX["B12"]]


# ============================================================
# 6. CREATE COMPOSITES
# ============================================================

# True-colour RGB:
# Red = B4, Green = B3, Blue = B2

rgb = make_rgb(
    B4, B3, B2
)


# NIR false colour:
# Red = B8, Green = B4, Blue = B3

nir_fcc = make_rgb(
    B8, B4, B3
)


# SWIR false colour:
# Red = B12, Green = B11, Blue = B4

swir_fcc = make_rgb(
    B12, B11, B4
)


# ============================================================
# 7. PLOT ALL SIX PANELS
# ============================================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 10)
)


# ------------------------------------------------------------
# Panel 1: B4 Red
# ------------------------------------------------------------

axes[0, 0].imshow(
    stretch_band(B4),
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 0].set_title("B4 - Red (Grayscale)")


# ------------------------------------------------------------
# Panel 2: B8 NIR
# ------------------------------------------------------------

axes[0, 1].imshow(
    stretch_band(B8),
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 1].set_title("B8 - NIR (Grayscale)")


# ------------------------------------------------------------
# Panel 3: True-colour RGB
# ------------------------------------------------------------

axes[0, 2].imshow(rgb)

axes[0, 2].set_title(
    "True Colour RGB (B4, B3, B2)"
)


# ------------------------------------------------------------
# Panel 4: NIR false colour
# ------------------------------------------------------------

axes[1, 0].imshow(nir_fcc)

axes[1, 0].set_title(
    "NIR False Colour (B8, B4, B3)"
)


# ------------------------------------------------------------
# Panel 5: SWIR false colour
# ------------------------------------------------------------

axes[1, 1].imshow(swir_fcc)

axes[1, 1].set_title(
    "SWIR False Colour (B12, B11, B4)"
)


# ------------------------------------------------------------
# Panel 6: RGB + Landslide mask
# ------------------------------------------------------------

axes[1, 2].imshow(rgb)

# Transparent red overlay on landslide pixels
overlay = np.zeros(
    (*mask.shape, 4),
    dtype=np.float32
)

overlay[mask == 1] = [
    1, 0, 0, 0.65
]

axes[1, 2].imshow(overlay)

axes[1, 2].set_title(
    "RGB + Landslide Mask Overlay"
)


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

for ax in axes.flat:

    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    f"Sentinel-2 Spectral Band Visual Inspection "
    f"- Image {SAMPLE_ID}",
    fontsize=15
)

plt.tight_layout()

output_path = (
    OUTPUT_DIR /
    f"visual_check_image_{SAMPLE_ID}.png"
)

plt.savefig(
    output_path,
    dpi=250,
    bbox_inches="tight"
)

plt.show()

print("\nSaved visualization:", output_path)


# ============================================================
# 8. PRINT BAND VALUE RANGES
# ============================================================

print("\nBAND VALUE CHECK")
print("-" * 55)

for name, band in [
    ("B2", B2),
    ("B3", B3),
    ("B4", B4),
    ("B8", B8),
    ("B11", B11),
    ("B12", B12)
]:

    print(
        f"{name:5s} | "
        f"Min: {band.min():.4f} | "
        f"Max: {band.max():.4f} | "
        f"Mean: {band.mean():.4f}"
    )

print("\nVISUAL BAND CHECK COMPLETE")
