from pathlib import Path
import h5py
import numpy as np

MASK_DIR = Path(r"C:\Users\Ojas\Desktop\RS_Project\Data_Normalized\ValidData\mask")

landslide_images = 0
non_landslide_images = 0
total_landslide_pixels = 0
total_non_landslide_pixels = 0

for file in sorted(MASK_DIR.glob("*.h5")):

    with h5py.File(file, "r") as f:
        mask = f["mask"][:]

    values, counts = np.unique(mask, return_counts=True)

    total_landslide_pixels += np.sum(mask == 1)
    total_non_landslide_pixels += np.sum(mask == 0)

    if np.any(mask == 1):
        landslide_images += 1
    else:
        non_landslide_images += 1

print("Total validation masks:", len(list(MASK_DIR.glob("*.h5"))))
print("Masks containing landslides:", landslide_images)
print("Masks containing only background:", non_landslide_images)

print("Total landslide pixels:", total_landslide_pixels)
print("Total non-landslide pixels:", total_non_landslide_pixels)