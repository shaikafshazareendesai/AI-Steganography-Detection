
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
clean_dir = ROOT / "dataset" / "train" / "clean"
stego_dir = ROOT / "dataset" / "train" / "stego"

for i in range(5):
    clean_path = clean_dir / f"image_{i:04d}.png"
    stego_path = stego_dir / f"stego_{i:04d}.png"

    if not clean_path.exists() or not stego_path.exists():
        print(f"Missing image pair: {clean_path.name}, {stego_path.name}")
        continue

    clean = cv2.imread(str(clean_path))
    stego = cv2.imread(str(stego_path))

    if clean is None or stego is None:
        print(f"Could not read image pair {i}")
        continue

    if clean.shape != stego.shape:
        print(f"Image dimensions differ for pair {i}")
        continue

    difference = cv2.absdiff(clean, stego)

    print(f"\nPair {i}:")
    print("Identical:", np.array_equal(clean, stego))
    print("Changed pixel channels:", np.count_nonzero(difference))
    print("Mean absolute difference:", difference.mean())
