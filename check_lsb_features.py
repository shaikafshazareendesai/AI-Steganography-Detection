
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"


def get_lsb_features(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Extract the least significant bit from every color channel.
    lsb = image & 1

    return {
        "lsb_ones_ratio": float(lsb.mean()),
        "lsb_zero_ratio": float(1.0 - lsb.mean()),
        "horizontal_lsb_changes": float(
            np.mean(lsb[:, 1:, :] != lsb[:, :-1, :])
        ),
        "vertical_lsb_changes": float(
            np.mean(lsb[1:, :, :] != lsb[:-1, :, :])
        ),
    }


def inspect_class(split, label, limit=100):
    folder = DATASET / split / label
    paths = sorted(folder.glob("*.png"))[:limit]

    if not paths:
        raise FileNotFoundError(f"No PNG images found in {folder}")

    features = [get_lsb_features(path) for path in paths]

    print(f"\n{split.upper()} - {label.upper()} ({len(paths)} images)")

    for key in features[0]:
        values = np.array([item[key] for item in features])
        print(
            f"{key}: mean={values.mean():.6f}, "
            f"std={values.std():.6f}, "
            f"min={values.min():.6f}, "
            f"max={values.max():.6f}"
        )


if __name__ == "__main__":
    for label in ("clean", "stego"):
        inspect_class("train", label)
