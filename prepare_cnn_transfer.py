
from pathlib import Path
import zipfile

PROJECT = Path(r"C:\Users\Ojas\Desktop\RS_Project")
BASE = PROJECT / "Data_Normalized"

OUTPUT = PROJECT / "CNN_TestData.zip"

splits = ["TestData"]

# Create a ZIP containing only training and validation data
total_files = 0

with zipfile.ZipFile(
    OUTPUT,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=1
) as archive:

    for split in splits:
        split_dir = BASE / split

        if not split_dir.exists():
            raise FileNotFoundError(split_dir)

        files = sorted(split_dir.rglob("*.h5"))

        print(f"\n{split}: {len(files)} HDF5 files")

        for file in files:
            # Preserve directory structure inside ZIP
            relative_path = file.relative_to(BASE)
            archive.write(file, arcname=str(relative_path))
            total_files += 1

print("\nTransfer package created!")
print("Total files:", total_files)
print("Archive:", OUTPUT)
print(f"Archive size: {OUTPUT.stat().st_size / (1024**3):.2f} GB")
