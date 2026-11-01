from pathlib import Path
import h5py
import numpy as np

ROOT = Path(r"C:\Users\Ojas\Desktop\RS_Project\Data_Normalized")

for split in ["TrainData", "ValidData", "TestData"]:

    mask_dir = ROOT / split / "mask"

    landslide_pixels = 0
    total_pixels = 0
    positive_masks = 0

    for file in mask_dir.glob("*.h5"):

        with h5py.File(file, "r") as f:
            mask = f["mask"][:]

        landslide_pixels += np.count_nonzero(mask == 1)
        total_pixels += mask.size

        if np.any(mask == 1):
            positive_masks += 1

    negative_pixels = total_pixels - landslide_pixels

    print(f"\n{split}")
    print("Total masks:", len(list(mask_dir.glob("*.h5"))))
    print("Positive masks:", positive_masks)
    print("Negative-only masks:",
          len(list(mask_dir.glob("*.h5"))) - positive_masks)
    print("Landslide pixels:", landslide_pixels)
    print("Non-landslide pixels:", negative_pixels)
    print("Landslide percentage:",
          round(100 * landslide_pixels / total_pixels, 3), "%")