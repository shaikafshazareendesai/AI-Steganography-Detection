
from pathlib import Path
import tempfile

import cv2
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from extract import hide_message, extract_message


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
MODEL_DIR = ROOT / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lsb_random_forest.joblib"

RANDOM_SEED = 42

# A short message, matching the format used by the app encoder.
TEST_MESSAGE = (
    "Hello! This is a test message for my AI "
    "Steganography Detection System."
)

FEATURE_NAMES = [
    "LSB ones ratio",
    "Horizontal LSB changes",
    "Vertical LSB changes",
]


# ============================================================
# 2. EXTRACT LSB FEATURES
# ============================================================

def extract_features(image_path):
    """
    Extract three statistical LSB features.

    Features:
    1. LSB ones ratio
    2. Horizontal LSB changes
    3. Vertical LSB changes
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    lsb = image & 1

    lsb_ones_ratio = float(lsb.mean())

    horizontal_changes = float(
        np.mean(lsb[:, 1:, :] != lsb[:, :-1, :])
    )

    vertical_changes = float(
        np.mean(lsb[1:, :, :] != lsb[:-1, :, :])
    )

    return [
        lsb_ones_ratio,
        horizontal_changes,
        vertical_changes,
    ]


# ============================================================
# 3. LOAD EXISTING DATASET SPLIT
# ============================================================

def load_split(split):
    """
    Load clean and stego PNG images from a dataset split.

    Labels:
        Clean = 0
        Stego = 1
    """

    features = []
    labels = []

    for class_name, class_label in [
        ("clean", 0),
        ("stego", 1),
    ]:
        folder = DATASET / split / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        image_paths = sorted(folder.glob("*.png"))

        if not image_paths:
            raise FileNotFoundError(
                f"No PNG images found in: {folder}"
            )

        print(
            f"Loading {len(image_paths)} "
            f"{class_name} images from {split}..."
        )

        for image_path in image_paths:
            features.append(extract_features(image_path))
            labels.append(class_label)

    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.int32),
    )


# ============================================================
# 4. PRINT EVALUATION RESULTS
# ============================================================

def evaluate_model(model, X_test, y_test, title):
    """Print accuracy, confusion matrix and classification report."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    print(f"Number of images: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nConfusion Matrix")
    print("Rows = Actual classes; Columns = Predicted classes")
    print("Class order: Clean (0), Stego (1)")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Clean", "Stego"],
            zero_division=0,
        )
    )

    return accuracy


# ============================================================
# 5. TEST WITH THE ACTUAL APP ENCODER
# ============================================================

def evaluate_app_encoder(model):
    """
    Create stego images from held-out clean test images using
    the same hide_message() function as the Streamlit app.

    The original clean image and its encoded copy are evaluated
    as a separate experiment. Neither image is added to training.
    """

    print("\n" + "=" * 65)
    print("STREAMLIT ENCODER COMPATIBILITY TEST")
    print("=" * 65)

    clean_folder = DATASET / "test" / "clean"

    if not clean_folder.exists():
        print("Cannot run encoder test: test/clean folder is missing.")
        return None

    clean_paths = sorted(clean_folder.glob("*.png"))

    if not clean_paths:
        print("Cannot run encoder test: no clean test PNG files found.")
        return None

    X_encoder_test = []
    y_encoder_test = []
    message_checks = []

    with tempfile.TemporaryDirectory(
        prefix="stego_encoder_test_"
    ) as temporary_directory:

        temporary_root = Path(temporary_directory)

        for index, clean_path in enumerate(clean_paths):
            stego_path = temporary_root / f"stego_{index:04d}.png"

            # Original clean image.
            X_encoder_test.append(extract_features(clean_path))
            y_encoder_test.append(0)

            # Encode a message into a copy of the same clean image.
            try:
                result = hide_message(
                    str(clean_path),
                    TEST_MESSAGE,
                    str(stego_path),
                )

                if result is False or not stego_path.exists():
                    print(
                        f"WARNING: Encoding failed for "
                        f"{clean_path.name}"
                    )
                    continue

                X_encoder_test.append(extract_features(stego_path))
                y_encoder_test.append(1)

                # Verify extraction separately from classification.
                try:
                    extracted = extract_message(str(stego_path))
                    message_checks.append(extracted == TEST_MESSAGE)
                except Exception:
                    message_checks.append(False)

            except Exception as error:
                print(
                    f"WARNING: Could not encode {clean_path.name}: "
                    f"{error}"
                )

    if not X_encoder_test:
        print("No encoder test images were generated.")
        return None

    X_encoder_test = np.asarray(
        X_encoder_test,
        dtype=np.float64,
    )
    y_encoder_test = np.asarray(
        y_encoder_test,
        dtype=np.int32,
    )

    print(f"Generated evaluation images: {len(y_encoder_test)}")
    print(
        "Successful message extractions: "
        f"{sum(message_checks)}/{len(message_checks)}"
        if message_checks
        else "No messages were verified."
    )

    accuracy = evaluate_model(
        model,
        X_encoder_test,
        y_encoder_test,
        "APP-ENCODER EVALUATION RESULTS",
    )

    # Report predictions specifically for the encoded images.
    stego_mask = y_encoder_test == 1

    if np.any(stego_mask):
        stego_predictions = model.predict(
            X_encoder_test[stego_mask]
        )

        detected = int(np.sum(stego_predictions == 1))
        total = int(len(stego_predictions))

        print("\nStego images created with the app encoder:")
        print(f"Correctly classified as Stego: {detected}/{total}")
        print(
            "Encoder-generated stego detection rate: "
            f"{detected / total * 100:.2f}%"
        )

    return accuracy


# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

def main():
    print("=" * 65)
    print("LSB FEATURE-BASED STEGANOGRAPHY CLASSIFIER")
    print("=" * 65)

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    print("\nDataset Summary")
    print("-" * 45)
    print("Training images:", len(y_train))
    print("Testing images:", len(y_test))
    print("Training feature shape:", X_train.shape)
    print("Testing feature shape:", X_test.shape)

    if len(np.unique(y_train)) != 2:
        raise ValueError(
            "Training data must contain both clean and stego images."
        )

    if len(np.unique(y_test)) != 2:
        raise ValueError(
            "Test data must contain both clean and stego images."
        )

    # Train only on the training split.
    print("\nTraining Random Forest classifier...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Training completed successfully.")

    # Save the trained model.
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Model location:", MODEL_PATH)

    # Evaluate on the existing held-out test split.
    dataset_accuracy = evaluate_model(
        model,
        X_test,
        y_test,
        "ORIGINAL SYNTHETIC TEST DATASET RESULTS",
    )

    # Feature importance.
    print("\nFeature Importances")
    print("-" * 45)

    for name, importance in zip(
        FEATURE_NAMES,
        model.feature_importances_,
    ):
        print(f"{name}: {importance:.4f}")

    # Verify that the saved model can be reloaded.
    print("\nVerifying saved model...")

    loaded_model = joblib.load(MODEL_PATH)

    if not np.array_equal(
        loaded_model.predict(X_test),
        model.predict(X_test),
    ):
        raise RuntimeError(
            "Saved model predictions differ from the trained model."
        )

    print("Saved model loaded and verified successfully.")

    # Additional evaluation using the actual app encoder.
    encoder_accuracy = evaluate_app_encoder(loaded_model)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    print(
        f"Original synthetic test accuracy: "
        f"{dataset_accuracy * 100:.2f}%"
    )

    if encoder_accuracy is not None:
        print(
            f"App-encoder experiment accuracy: "
            f"{encoder_accuracy * 100:.2f}%"
        )

    print(f"Model saved at: {MODEL_PATH}")
    print("\nEvaluation completed.")


# ============================================================
# 7. RUN
# ============================================================

if __name__ == "__main__":
    main()
