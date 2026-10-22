
import random
from pathlib import Path

import cv2
import numpy as np

from extract import hide_message, extract_message


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"

IMAGE_SIZE = (128, 128)

DATASET_CONFIG = {
    "train": {"clean": 400, "stego": 400},
    "val": {"clean": 100, "stego": 100},
    "test": {"clean": 100, "stego": 100},
}

RANDOM_SEED = 42

# A single message, without repeating it multiple times.
STEGO_MESSAGE = (
    "This is a test message for the AI-based steganography "
    "detection system. The purpose is to evaluate whether a "
    "CNN can distinguish clean images from images containing "
    "hidden information."
)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_clean_image():
    """
    Generate a synthetic clean RGB image using different
    visual patterns and a small amount of Gaussian noise.
    """

    width, height = IMAGE_SIZE

    pattern_type = random.choice(
        ["noise", "gradient", "circles", "lines"]
    )

    if pattern_type == "noise":
        image = np.random.randint(
            0, 256, (height, width, 3), dtype=np.uint8
        )

    elif pattern_type == "gradient":
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)

        xx, yy = np.meshgrid(x, y)

        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = xx
        image[:, :, 1] = yy
        image[:, :, 2] = ((xx.astype(np.uint16) +
                           yy.astype(np.uint16)) // 2).astype(
                               np.uint8
                           )

    elif pattern_type == "circles":
        image = np.zeros((height, width, 3), dtype=np.uint8)

        background = tuple(
            random.randint(0, 255) for _ in range(3)
        )
        image[:] = background

        for _ in range(random.randint(5, 15)):
            center = (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            )
            radius = random.randint(3, 25)
            color = tuple(
                random.randint(0, 255) for _ in range(3)
            )

            cv2.circle(
                image,
                center,
                radius,
                color,
                thickness=-1,
            )

    else:  # lines
        image = np.zeros((height, width, 3), dtype=np.uint8)

        image[:] = tuple(
            random.randint(0, 255) for _ in range(3)
        )

        for _ in range(random.randint(5, 20)):
            point1 = (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            )
            point2 = (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            )
            color = tuple(
                random.randint(0, 255) for _ in range(3)
            )

            cv2.line(
                image,
                point1,
                point2,
                color,
                thickness=random.randint(1, 3),
            )

    # Add a small amount of Gaussian noise.
    noise = np.random.normal(
        loc=0,
        scale=2,
        size=image.shape,
    )

    image = np.clip(
        image.astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)

    return image


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_dataset_directories():
    """Create directories for train, validation, and test."""

    for split in DATASET_CONFIG:
        for category in ("clean", "stego"):
            directory = DATASET_DIR / split / category
            directory.mkdir(parents=True, exist_ok=True)

    print("Dataset directories created successfully.")


# ============================================================
# CLEAN IMAGE GENERATION
# ============================================================

def generate_clean_images(split, count):
    """Generate and save synthetic clean images."""

    output_dir = DATASET_DIR / split / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {count} clean images for {split}...")

    for index in range(count):
        image = generate_clean_image()

        output_path = output_dir / f"clean_{index:04d}.png"

        success = cv2.imwrite(str(output_path), image)

        if not success:
            raise RuntimeError(
                f"Failed to save clean image: {output_path}"
            )

        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"  Clean images: {index + 1}/{count}")

    print(f"Saved clean images in: {output_dir}")


# ============================================================
# STEGO IMAGE GENERATION
# ============================================================

def generate_stego_images(split, count):
    """
    Generate clean images and embed the same single message
    into each image using the existing hide_message function.
    """

    output_dir = DATASET_DIR / split / "stego"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {count} stego images for {split}...")

    for index in range(count):
        clean_image = generate_clean_image()

        # Save a temporary clean image before embedding.
        temporary_path = (
            output_dir / f"_temporary_{index:04d}.png"
        )

        stego_path = output_dir / f"stego_{index:04d}.png"

        success = cv2.imwrite(str(temporary_path), clean_image)

        if not success:
            raise RuntimeError(
                f"Failed to save temporary image: {temporary_path}"
            )

        try:
            # Embed the message using your existing extract.py.
            hide_message(
                str(temporary_path),
                STEGO_MESSAGE,
                str(stego_path),
            )

        finally:
            # Remove the temporary clean image.
            if temporary_path.exists():
                temporary_path.unlink()

        if not stego_path.exists():
            raise RuntimeError(
                f"Stego image was not created: {stego_path}"
            )

        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"  Stego images: {index + 1}/{count}")

    print(f"Saved stego images in: {output_dir}")


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset():
    """Generate clean and stego images for all dataset splits."""

    print("=" * 60)
    print("AI-BASED STEGANOGRAPHY DETECTION")
    print("SYNTHETIC DATASET GENERATOR")
    print("=" * 60)

    print(f"\nProject directory: {ROOT_DIR}")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Image size: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Message length: {len(STEGO_MESSAGE)} characters")

    create_dataset_directories()

    for split, configuration in DATASET_CONFIG.items():
        generate_clean_images(
            split,
            configuration["clean"],
        )

        generate_stego_images(
            split,
            configuration["stego"],
        )

    print("\nDataset generation completed.")

    verify_dataset()


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_dataset():
    """Verify image counts and test hidden-message extraction."""

    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    all_counts_correct = True

    for split, configuration in DATASET_CONFIG.items():
        print(f"\n{split.upper()} DATASET")

        for category in ("clean", "stego"):
            directory = DATASET_DIR / split / category

            image_files = [
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                and not path.name.startswith("_temporary_")
            ]

            expected_count = configuration[category]
            actual_count = len(image_files)

            print(
                f"  {category.capitalize()}: "
                f"{actual_count}/{expected_count}"
            )

            if actual_count != expected_count:
                all_counts_correct = False
                print(
                    "  WARNING: Image count does not match "
                    "the expected count."
                )

    print("\nChecking hidden-message extraction...")

    test_stego_dir = DATASET_DIR / "test" / "stego"

    stego_files = sorted(
        path
        for path in test_stego_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and not path.name.startswith("_temporary_")
    )

    if not stego_files:
        print("ERROR: No test stego images were found.")
        return False

    extraction_successes = 0
    files_to_test = stego_files[:3]

    for image_path in files_to_test:
        try:
            extracted_message = extract_message(str(image_path))

            if extracted_message == STEGO_MESSAGE:
                extraction_successes += 1
                print(f"  PASS: {image_path.name}")
                print(
                    f"        Extracted {len(extracted_message)} "
                    "characters."
                )
            else:
                print(f"  FAIL: {image_path.name}")

                if extracted_message is None:
                    print("        No message was extracted.")
                else:
                    print(
                        "        Extracted message does not match "
                        "the expected message."
                    )

        except Exception as error:
            print(f"  FAIL: {image_path.name}")
            print(f"        Error: {error}")

    extraction_successful = (
        extraction_successes == len(files_to_test)
    )

    print("\n" + "-" * 60)

    if all_counts_correct:
        print("Image counts: PASS")
    else:
        print("Image counts: CHECK REQUIRED")

    if extraction_successful:
        print("Message extraction: PASS")
    else:
        print("Message extraction: CHECK REQUIRED")

    if all_counts_correct and extraction_successful:
        print("\nDATASET VERIFICATION PASSED.")
        return True

    print("\nDATASET VERIFICATION NEEDS ATTENTION.")
    return False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_dataset()
