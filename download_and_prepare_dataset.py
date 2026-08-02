"""
download_and_prepare_dataset.py
----------------------------------
Downloads the three source datasets from Kaggle, unzips them, and
reorganizes their folders into the "Plant___Disease" naming convention
this project expects, under raw_data/.

Requires:
    pip install kaggle
    A valid kaggle.json API token at ~/.kaggle/kaggle.json
    (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)

After running this, raw_data/ will contain folders like:
    raw_data/Mango___Anthracnose/
    raw_data/Mango___Healthy/
    raw_data/Coconut___Bud_Root_Dropping/
    ...

Run split_dataset.py next to turn raw_data/ into dataset/train|val|test.

NOTE: Dataset class-folder names on Kaggle change over time and vary in
casing/spacing. This script prints what it finds after each download so
you can verify the RENAME_MAP below still matches — adjust it if a
dataset's internal folder names differ from what's listed here.
"""

import os
import shutil
import subprocess
import sys
import zipfile

RAW_DIR = "raw_data"
DOWNLOAD_DIR = "kaggle_downloads"

DATASETS = [
    {
        "slug": "aryashah2k/mango-leaf-disease-dataset",
        "plant": "Mango",
    },
    {
        "slug": "nagabushan/coconut-tree-disease-dataset",
        "plant": "Coconut",
    },
    # Confirmed via `kaggle datasets list -s "arecanut"` — this is the one
    # actually labeled as a disease dataset (not just general arecanut photos).
    # NOTE: unlike the other two, this dataset's folders are named "0", "1", "2"...
    # (not disease names). The real disease name is embedded in each filename
    # instead, e.g. "bud borer_original_110.jpg_<uuid>.jpg". reorganize()
    # below detects this dataset by slug and parses labels from filenames.
    {
        "slug": "sumanthsadiga/arecanut-plant-disease-dataset",
        "plant": "Areca_Nut",
        "label_from_filename": True,
    },
]


def ensure_kaggle_installed():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing the kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])


def download_dataset(slug: str, dest_zip_dir: str):
    os.makedirs(dest_zip_dir, exist_ok=True)
    print(f"\nDownloading {slug} ...")
    subprocess.check_call(
        ["kaggle", "datasets", "download", "-d", slug, "-p", dest_zip_dir]
    )


def unzip_all(zip_dir: str, extract_to: str):
    os.makedirs(extract_to, exist_ok=True)
    for fname in os.listdir(zip_dir):
        if fname.endswith(".zip"):
            path = os.path.join(zip_dir, fname)
            print(f"Unzipping {fname} ...")
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(extract_to)


def find_class_folders(root: str):
    """Recursively find leaf folders that directly contain image files."""
    class_folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        image_files = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if image_files and not dirnames:
            class_folders.append(dirpath)
    return class_folders


def reorganize_by_filename(plant: str, extracted_root: str):
    """
    For datasets like sumanthsadiga/arecanut-plant-disease-dataset, where
    folders are just numeric IDs ("0", "1"...) but the real label is
    embedded in each filename, e.g.:
        "bud borer_original_110.jpg_<uuid>.jpg"  ->  label = "bud borer"
    Splits on the first "_original_" (case-insensitive) and uses everything
    before it as the class label.
    """
    counts = {}
    for dirpath, _, filenames in os.walk(extracted_root):
        for fname in filenames:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            lower = fname.lower()
            marker = "_original_"
            if marker not in lower:
                counts.setdefault("_UNRECOGNIZED_", 0)
                counts["_UNRECOGNIZED_"] += 1
                continue

            split_idx = lower.index(marker)
            raw_label = fname[:split_idx].strip()
            class_name = raw_label.title().replace(" ", "_")
            dest_name = f"{plant}___{class_name}"
            dest_path = os.path.join(RAW_DIR, dest_name)
            os.makedirs(dest_path, exist_ok=True)

            src = os.path.join(dirpath, fname)
            dest = os.path.join(dest_path, fname)
            if os.path.exists(dest):
                dest = os.path.join(dest_path, f"{counts.get(dest_name, 0)}_{fname}")
            shutil.copy(src, dest)

            counts[dest_name] = counts.get(dest_name, 0) + 1

    if counts.pop("_UNRECOGNIZED_", 0):
        print(f"  [!] Some files under {extracted_root} didn't match the expected "
              f"'_original_' filename pattern and were skipped — check them manually.")

    print(f"  Found {len(counts)} class(es) for {plant} (parsed from filenames):")
    for dest_name, count in sorted(counts.items()):
        print(f"    {dest_name}: {count} images -> raw_data/{dest_name}/")


def reorganize(plant: str, extracted_root: str):
    class_folders = find_class_folders(extracted_root)
    if not class_folders:
        print(f"  [!] No class folders with images found under {extracted_root}")
        print("      Open this folder manually and check its structure.")
        return

    print(f"  Found {len(class_folders)} class folder(s) for {plant}:")
    for folder in class_folders:
        class_name = os.path.basename(folder).strip().replace(" ", "_")
        dest_name = f"{plant}___{class_name}"
        dest_path = os.path.join(RAW_DIR, dest_name)
        os.makedirs(dest_path, exist_ok=True)

        count = 0
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy(os.path.join(folder, fname), os.path.join(dest_path, fname))
                count += 1
        print(f"    {dest_name}: {count} images -> raw_data/{dest_name}/")


def main():
    ensure_kaggle_installed()
    os.makedirs(RAW_DIR, exist_ok=True)

    for entry in DATASETS:
        slug, plant = entry["slug"], entry["plant"]
        zip_dir = os.path.join(DOWNLOAD_DIR, plant)
        extract_dir = os.path.join(DOWNLOAD_DIR, plant, "extracted")

        try:
            download_dataset(slug, zip_dir)
        except subprocess.CalledProcessError:
            print(f"  [!] Download failed for {slug}. Check the slug is still valid on Kaggle, "
                  f"or that you've accepted the dataset's rules on its Kaggle page first.")
            continue

        unzip_all(zip_dir, extract_dir)

        if entry.get("label_from_filename"):
            reorganize_by_filename(plant, extract_dir)
        else:
            reorganize(plant, extract_dir)

    print("\nDone. Review raw_data/ folder names before running split_dataset.py —")
    print("merge/rename any duplicate or oddly-named class folders you spot.")


if __name__ == "__main__":
    main()
