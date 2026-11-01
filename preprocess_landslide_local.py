"""
Landslide4Sense local preprocessing pipeline
--------------------------------------------
1. Audits image/mask HDF5 files in TrainData, ValidData and TestData.
2. Checks empty files/datasets, NaN/Inf, image dimensions, band count,
   binary mask values, and image-mask spatial correspondence.
3. Selects B2, B3, B4, B8, B11, B12 from the 14-channel image.
4. Computes per-band Z-score statistics using valid TRAIN images only.
5. Applies those same statistics to all splits.
6. Writes a new output directory; the source dataset is never modified.

Expected input layout:
Data/
    TrainData/img, TrainData/mask
    ValidData/img, ValidData/mask
    TestData/img, TestData/mask

Expected HDF5 dataset keys: "img" for images and "mask" for masks.
If those keys are absent, the script uses the first HDF5 dataset it finds.
"""

from pathlib import Path
import argparse
import csv
import json
import re
import sys

import h5py
import numpy as np


SPLITS = ("TrainData", "ValidData", "TestData")
BAND_NAMES = ("B2", "B3", "B4", "B8", "B11", "B12")

# Zero-based channel indices for B2, B3, B4, B8, B11, B12
BAND_INDICES = (1, 2, 3, 7, 10, 11)
EXPECTED_BANDS = 14


def get_sample_id(path: Path) -> str:
    """Extract trailing digits, so image_12.h5 pairs with mask_12.h5."""
    match = re.search(r"(\d+)$", path.stem)
    return match.group(1) if match else path.stem


def id_sort_key(sample_id: str):
    """Sort numeric IDs numerically and other IDs lexically without type errors."""
    if sample_id.isdigit():
        return (0, int(sample_id))
    return (1, sample_id)


def find_hdf5_dataset(h5_file: h5py.File, preferred_key: str):
    """Return the requested dataset, or the first dataset in the HDF5 file."""
    if preferred_key in h5_file and isinstance(h5_file[preferred_key], h5py.Dataset):
        return h5_file[preferred_key]

    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    h5_file.visititems(visitor)

    if not datasets:
        raise ValueError("No HDF5 dataset found in file")

    return h5_file[datasets[0]]


def read_h5(path: Path, preferred_key: str) -> np.ndarray:
    """Read an HDF5 array safely while the file is open."""
    with h5py.File(path, "r") as h5_file:
        dataset = find_hdf5_dataset(h5_file, preferred_key)
        if dataset.size == 0:
            raise ValueError("HDF5 dataset is empty")
        return dataset[()]


def validate_image(path: Path):
    try:
        image = read_h5(path, "img")

        if image.ndim != 3:
            return False, f"Expected 3D image (H,W,C), got shape {image.shape}", None

        if image.shape[-1] != EXPECTED_BANDS:
            return False, f"Expected {EXPECTED_BANDS} bands, got shape {image.shape}", None

        if image.shape[0] == 0 or image.shape[1] == 0:
            return False, f"Empty spatial dimension: {image.shape}", None

        if not np.issubdtype(image.dtype, np.number):
            return False, f"Non-numeric image dtype: {image.dtype}", None

        if not np.isfinite(image).all():
            return False, "NaN or Inf found in image", None

        return True, "OK", image.shape

    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def validate_mask(path: Path):
    try:
        mask = read_h5(path, "mask")

        # Some HDF5 masks may be stored as (H,W,1); accept and squeeze that case.
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)

        if mask.ndim != 2:
            return False, f"Expected 2D mask (H,W), got shape {mask.shape}", None

        if mask.shape[0] == 0 or mask.shape[1] == 0:
            return False, f"Empty spatial dimension: {mask.shape}", None

        if not np.issubdtype(mask.dtype, np.number):
            return False, f"Non-numeric mask dtype: {mask.dtype}", None

        if not np.isfinite(mask).all():
            return False, "NaN or Inf found in mask", None

        values = np.unique(mask)
        if not np.isin(values, (0, 1)).all():
            return False, f"Unexpected mask values: {values[:20]}", None

        return True, "OK", mask.shape

    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def discover_pairs(data_root: Path, audit_rows: list):
    """Find image/mask pairs by sample ID and audit every discovered ID."""
    valid_pairs = {}

    for split in SPLITS:
        image_dir = data_root / split / "img"
        mask_dir = data_root / split / "mask"

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

        image_files = sorted(image_dir.glob("*.h5"))
        mask_files = sorted(mask_dir.glob("*.h5"))

        images = {get_sample_id(path): path for path in image_files}
        masks = {get_sample_id(path): path for path in mask_files}

        # Detect duplicate IDs rather than silently overwriting one file.
        if len(images) != len(image_files):
            raise ValueError(f"Duplicate image sample IDs found in {image_dir}")
        if len(masks) != len(mask_files):
            raise ValueError(f"Duplicate mask sample IDs found in {mask_dir}")

        valid_pairs[split] = []

        for sample_id in sorted(set(images) | set(masks), key=id_sort_key):
            image_path = images.get(sample_id)
            mask_path = masks.get(sample_id)

            if image_path is None:
                image_status, image_shape = "Missing image", ""
                image_ok = False
            else:
                image_ok, image_status, image_shape = validate_image(image_path)

            if mask_path is None:
                mask_status, mask_shape = "Missing mask", ""
                mask_ok = False
            else:
                mask_ok, mask_status, mask_shape = validate_mask(mask_path)

            pair_ok = image_ok and mask_ok

            if pair_ok and image_shape[:2] != mask_shape:
                pair_ok = False
                mask_status = (
                    f"Image-mask dimension mismatch: "
                    f"image {image_shape[:2]} vs mask {mask_shape}"
                )

            audit_rows.append({
                "split": split,
                "sample_id": sample_id,
                "image_file": image_path.name if image_path else "",
                "mask_file": mask_path.name if mask_path else "",
                "image_status": image_status,
                "mask_status": mask_status,
                "valid_pair": pair_ok,
            })

            if pair_ok:
                valid_pairs[split].append((sample_id, image_path, mask_path))

    return valid_pairs


def calculate_training_statistics(train_pairs):
    """Compute per-band population mean/std over all training pixels."""
    band_sum = np.zeros(len(BAND_INDICES), dtype=np.float64)
    band_sq_sum = np.zeros(len(BAND_INDICES), dtype=np.float64)
    pixel_count = 0

    for _, image_path, _ in train_pairs:
        image = read_h5(image_path, "img")
        selected = image[..., BAND_INDICES].astype(np.float64, copy=False)

        band_sum += selected.sum(axis=(0, 1), dtype=np.float64)
        band_sq_sum += np.square(selected).sum(axis=(0, 1), dtype=np.float64)
        pixel_count += selected.shape[0] * selected.shape[1]

    if pixel_count == 0:
        raise RuntimeError("No valid training image pairs found; cannot normalize.")

    means = band_sum / pixel_count
    variances = band_sq_sum / pixel_count - np.square(means)
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances)

    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise RuntimeError("Non-finite normalization statistics were calculated.")

    zero_std_bands = [BAND_NAMES[i] for i, value in enumerate(stds) if value == 0]
    if zero_std_bands:
        raise RuntimeError(
            "Zero standard deviation in band(s): "
            + ", ".join(zero_std_bands)
            + ". Cannot perform Z-score normalization."
        )

    return means, stds, pixel_count


def write_h5(path: Path, key: str, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            key,
            data=array,
            compression="gzip",
            compression_opts=4,
        )


def process_splits(valid_pairs, output_root: Path, means, stds):
    """Write six-band normalized images and unchanged binary masks."""
    for split in SPLITS:
        print(f"\nProcessing {split}: {len(valid_pairs[split])} valid pairs")

        image_output_dir = output_root / split / "img"
        mask_output_dir = output_root / split / "mask"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        for sample_id, image_path, mask_path in valid_pairs[split]:
            image = read_h5(image_path, "img")
            mask = read_h5(mask_path, "mask")

            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)

            selected = image[..., BAND_INDICES].astype(np.float32)
            normalized = (
                selected - means.astype(np.float32)
            ) / stds.astype(np.float32)

            if not np.isfinite(normalized).all():
                raise RuntimeError(
                    f"Non-finite normalized values generated for {image_path}"
                )

            write_h5(image_output_dir / image_path.name, "img", normalized)
            # Masks are labels, not spectral data: do not normalize them.
            write_h5(mask_output_dir / mask_path.name, "mask", mask.astype(np.uint8))

        print(f"Finished {split}")


def verify_output(output_root: Path):
    print("\nOUTPUT VERIFICATION")
    for split in SPLITS:
        image_files = sorted((output_root / split / "img").glob("*.h5"))
        mask_files = sorted((output_root / split / "mask").glob("*.h5"))

        print(f"\n{split}")
        print(f"  Images: {len(image_files)}")
        print(f"  Masks:  {len(mask_files)}")

        if len(image_files) != len(mask_files):
            print("  WARNING: image and mask counts differ.")

        if image_files:
            sample_image = read_h5(image_files[0], "img")
            print(f"  Sample image shape: {sample_image.shape}")
            print(f"  Sample image dtype: {sample_image.dtype}")
            print(f"  Sample image finite: {np.isfinite(sample_image).all()}")

        if mask_files:
            sample_mask = read_h5(mask_files[0], "mask")
            print(f"  Sample mask shape: {sample_mask.shape}")
            print(f"  Sample mask values: {np.unique(sample_mask)}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit and normalize Landslide4Sense HDF5 data locally."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help=r'Path to source Data folder, e.g. "D:\LandslideProject\Data"',
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output folder. Default: sibling folder named Data_Normalized.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else data_root.parent / "Data_Normalized"
    )

    # Avoid accidentally writing output inside the source dataset.
    if output_root == data_root or data_root in output_root.parents:
        raise ValueError(
            "Output directory must be outside the source Data directory."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    audit_dir = output_root / "audit_output"
    audit_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source dataset: {data_root}")
    print(f"Output dataset: {output_root}")

    audit_rows = []
    valid_pairs = discover_pairs(data_root, audit_rows)

    audit_csv = audit_dir / "initial_audit.csv"
    fieldnames = [
        "split", "sample_id", "image_file", "mask_file",
        "image_status", "mask_status", "valid_pair",
    ]
    with audit_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    print("\nAUDIT SUMMARY")
    for split in SPLITS:
        total = sum(1 for row in audit_rows if row["split"] == split)
        valid = len(valid_pairs[split])
        invalid = total - valid
        print(f"{split}: {valid} valid pairs / {invalid} invalid or unmatched / {total} IDs")

    invalid_rows = [row for row in audit_rows if not row["valid_pair"]]
    if invalid_rows:
        print(f"\nFound {len(invalid_rows)} invalid/unmatched sample(s).")
        print(f"Details saved to: {audit_csv}")
        print("Invalid samples will be excluded from the normalized output.")

    if not valid_pairs["TrainData"]:
        raise RuntimeError("No valid training pairs found. See the audit CSV.")

    means, stds, pixel_count = calculate_training_statistics(
        valid_pairs["TrainData"]
    )

    stats = {
        "bands": list(BAND_NAMES),
        "zero_based_channel_indices": list(BAND_INDICES),
        "normalization": "Z-score",
        "statistics_source": "valid TrainData images only",
        "training_pixel_count": int(pixel_count),
        "means": means.tolist(),
        "standard_deviations": stds.tolist(),
    }

    stats_path = output_root / "normalization_stats.json"
    with stats_path.open("w", encoding="utf-8") as json_file:
        json.dump(stats, json_file, indent=4)

    print("\nNORMALIZATION STATISTICS")
    for band, mean, std in zip(BAND_NAMES, means, stds):
        print(f"{band}: mean={mean:.8g}, std={std:.8g}")

    process_splits(valid_pairs, output_root, means, stds)
    verify_output(output_root)

    print("\nPREPROCESSING COMPLETE")
    print(f"Audit CSV: {audit_csv}")
    print(f"Statistics: {stats_path}")
    print(f"Normalized dataset: {output_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
