"""
split_dataset.py
-------------------
Takes raw_data/<Plant___Disease>/*.jpg (produced by
download_and_prepare_dataset.py, or organized by hand) and splits each
class into dataset/train, dataset/val, dataset/test in the proportions
below — the exact layout train.py expects.

Usage:
    python split_dataset.py
"""

import os
import random
import shutil

RAW_DIR = "raw_data"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # must sum to 1.0 with the two above

MIN_IMAGES_PER_CLASS = 20  # warn if a class has fewer than this
SEED = 42


def main():
    random.seed(SEED)

    if not os.path.isdir(RAW_DIR):
        print(f"'{RAW_DIR}/' not found. Run download_and_prepare_dataset.py first, "
              f"or manually create raw_data/<Plant___Disease>/ folders of images.")
        return

    class_names = [
        d for d in sorted(os.listdir(RAW_DIR))
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ]
    if not class_names:
        print(f"No class folders found inside '{RAW_DIR}/'.")
        return

    print(f"Found {len(class_names)} classes. Splitting {TRAIN_RATIO:.0%}/"
          f"{VAL_RATIO:.0%}/{TEST_RATIO:.0%} into train/val/test...\n")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    for class_name in class_names:
        class_dir = os.path.join(RAW_DIR, class_name)
        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        if len(images) < MIN_IMAGES_PER_CLASS:
            print(f"  [!] {class_name}: only {len(images)} images — consider "
                  f"gathering more before training (aim for {MIN_IMAGES_PER_CLASS}+).")

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        # remainder goes to test, so all images are used even with rounding
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, file_list in splits.items():
            dest_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in file_list:
                shutil.copy(os.path.join(class_dir, fname), os.path.join(dest_dir, fname))

        print(f"  {class_name}: {n_total} total -> "
              f"train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")

    print("\nDone. dataset/train, dataset/val, dataset/test are ready.")
    print("Next: python train.py")


if __name__ == "__main__":
    main()
