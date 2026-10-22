
"""
Prediction Script for Steganography Detection
Uses a trained Random Forest classifier with LSB statistical features.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from PIL import Image

from extract import extract_message


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "lsb_random_forest.joblib"

CLASS_NAMES = ["Clean", "Stego"]

FEATURE_NAMES = [
    "LSB ones ratio",
    "Horizontal LSB changes",
    "Vertical LSB changes",
]


class SteganographyPredictor:
    """Detect potential steganography using LSB statistical features."""

    def __init__(self, model_path=None, image_size=(96, 96)):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.image_size = image_size
        self.model = None
        self.class_names = CLASS_NAMES

    # --------------------------------------------------
    # 1. LOAD TRAINED RANDOM FOREST MODEL
    # --------------------------------------------------

    def load_model(self) -> bool:
        """Load the saved Random Forest model."""

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    "Run python test_lsb_classifier.py first."
                )

            self.model = joblib.load(self.model_path)

            if not hasattr(self.model, "predict_proba"):
                raise ValueError(
                    "The loaded model does not support predict_proba."
                )

            print(f"Random Forest model loaded: {self.model_path}")
            return True

        except Exception as error:
            print(f"Error loading model: {error}")
            return False

    def _ensure_model_loaded(self):
        if self.model is None and not self.load_model():
            raise RuntimeError(
                "Could not load the Random Forest model. "
                "Check that model/lsb_random_forest.joblib exists."
            )

    # --------------------------------------------------
    # 2. EXTRACT THE SAME FEATURES USED DURING TRAINING
    # --------------------------------------------------

    @staticmethod
    def extract_features_from_array(image_array: np.ndarray) -> np.ndarray:
        """
        Extract the same three features used by test_lsb_classifier.py.

        The image is not resized because the classifier was trained
        using features extracted from the original images.
        """

        if image_array is None:
            raise ValueError("Image is empty or could not be read.")

        image_array = np.asarray(image_array)

        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.ndim == 3 and image_array.shape[-1] == 4:
            image_array = image_array[..., :3]

        if image_array.ndim != 3 or image_array.shape[-1] != 3:
            raise ValueError("Expected a grayscale, RGB, or RGBA image.")

        if image_array.shape[0] < 2 or image_array.shape[1] < 2:
            raise ValueError("Image must be at least 2 x 2 pixels.")

        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        lsb = image_array & 1

        lsb_ones_ratio = float(lsb.mean())

        horizontal_changes = float(
            np.mean(lsb[:, 1:, :] != lsb[:, :-1, :])
        )

        vertical_changes = float(
            np.mean(lsb[1:, :, :] != lsb[:-1, :, :])
        )

        return np.asarray(
            [[
                lsb_ones_ratio,
                horizontal_changes,
                vertical_changes,
            ]],
            dtype=np.float64,
        )

    @classmethod
    def extract_features(cls, image_path: str) -> np.ndarray:
        """Read an image from disk and extract its LSB features."""

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return cls.extract_features_from_array(image)

    # --------------------------------------------------
    # 3. PREDICTION HELPERS
    # --------------------------------------------------

    def _predict_features(
        self,
        features: np.ndarray,
    ) -> Tuple[str, float]:
        """Return the predicted label and its class probability."""

        self._ensure_model_loaded()

        probabilities = self.model.predict_proba(features)[0]
        classes = list(self.model.classes_)

        stego_class_index = classes.index(1)
        clean_class_index = classes.index(0)

        stego_probability = float(probabilities[stego_class_index])
        clean_probability = float(probabilities[clean_class_index])

        if stego_probability >= clean_probability:
            return "Stego", stego_probability

        return "Clean", clean_probability

    def predict_single_image(
        self,
        image_path: str,
        threshold: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Predict whether an image is Clean or Stego.

        Returns:
            Tuple containing predicted label and its probability.
        """

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        features = self.extract_features(image_path)
        return self._predict_features(features)

    def predict_from_array(
        self,
        image_array: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[str, float]:
        """Predict from an in-memory NumPy image array."""

        features = self.extract_features_from_array(image_array)
        return self._predict_features(features)

    def predict_from_pil_image(
        self,
        image: Image.Image,
        threshold: Optional[float] = None,
    ) -> Tuple[str, float]:
        """Predict from a PIL image."""

        rgb_image = image.convert("RGB")
        image_array = np.asarray(rgb_image, dtype=np.uint8)

        return self.predict_from_array(image_array, threshold)

    # --------------------------------------------------
    # 4. HIDDEN MESSAGE EXTRACTION
    # --------------------------------------------------

    @staticmethod
    def extract_hidden_message(
        image_path: str,
        allow_unverified_raw: bool = False,
    ) -> Optional[str]:
        """Attempt to extract a hidden message from an image."""

        try:
            return extract_message(
                image_path,
                allow_unverified_raw=allow_unverified_raw,
            )
        except Exception as error:
            print(f"Message extraction failed: {error}")
            return None

    @staticmethod
    def extract_hidden_message_from_pil(
        image: Image.Image,
        allow_unverified_raw: bool = False,
    ) -> Optional[str]:
        """Attempt extraction from a PIL image using a temporary PNG."""

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name

            try:
                image.convert("RGB").save(temp_path, format="PNG")

                return extract_message(
                    temp_path,
                    allow_unverified_raw=allow_unverified_raw,
                )
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as error:
            print(f"Message extraction failed: {error}")
            return None

    # --------------------------------------------------
    # 5. ANALYZE A FILE
    # --------------------------------------------------

    def analyze_image(
        self,
        image_path: str,
        threshold: Optional[float] = None,
    ) -> Dict:
        """Classify an image and attempt extraction if Stego is predicted."""

        image_path = str(image_path)

        if not os.path.isfile(image_path):
            return {
                "image_path": image_path,
                "success": False,
                "error": f"Image file not found: {image_path}",
            }

        try:
            prediction, confidence = self.predict_single_image(
                image_path,
                threshold=threshold,
            )

            result = {
                "image_path": image_path,
                "prediction": prediction,
                "confidence": confidence,
                "success": True,
                "hidden_message": None,
            }

            if prediction == "Stego":
                result["hidden_message"] = self.extract_hidden_message(
                    image_path,
                    allow_unverified_raw=True,
                )

            return result

        except Exception as error:
            return {
                "image_path": image_path,
                "success": False,
                "error": str(error),
            }

    def analyze_pil_image(
        self,
        image: Image.Image,
        image_name: str = "uploaded_image",
        threshold: Optional[float] = None,
    ) -> Dict:
        """Analyze an uploaded PIL image without requiring a permanent file."""

        try:
            prediction, confidence = self.predict_from_pil_image(
                image,
                threshold=threshold,
            )

            result = {
                "image_path": image_name,
                "prediction": prediction,
                "confidence": confidence,
                "success": True,
                "hidden_message": None,
            }

            if prediction == "Stego":
                result["hidden_message"] = (
                    self.extract_hidden_message_from_pil(
                        image,
                        allow_unverified_raw=True,
                    )
                )

            return result

        except Exception as error:
            return {
                "image_path": image_name,
                "success": False,
                "error": str(error),
            }

    # --------------------------------------------------
    # 6. BATCH PREDICTION
    # --------------------------------------------------

    def batch_predict(self, image_paths: List[str]) -> List[Dict]:
        """Analyze multiple image files."""

        return [
            self.analyze_image(image_path)
            for image_path in image_paths
        ]

    # --------------------------------------------------
    # 7. MODEL INFORMATION
    # --------------------------------------------------

    def get_model_info(self) -> Dict:
        """Return basic information about the loaded model."""

        try:
            self._ensure_model_loaded()

            return {
                "model_path": str(self.model_path),
                "model_type": type(self.model).__name__,
                "features": FEATURE_NAMES,
                "classes": list(self.model.classes_),
                "feature_importances": (
                    self.model.feature_importances_.tolist()
                    if hasattr(self.model, "feature_importances_")
                    else None
                ),
            }

        except Exception as error:
            return {"error": str(error)}


# --------------------------------------------------
# 8. TEST THE PREDICTOR ON EXAMPLE DATASET IMAGES
# --------------------------------------------------

def test_predictor():
    print("\nTesting Random Forest Steganography Predictor")
    print("=" * 55)

    predictor = SteganographyPredictor()

    if not predictor.load_model():
        return

    test_images = []

    for label in ("clean", "stego"):
        folder = PROJECT_ROOT / "dataset" / "test" / label

        if folder.exists():
            images = sorted(
                path
                for path in folder.iterdir()
                if path.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".bmp", ".webp"
                }
            )

            test_images.extend(str(path) for path in images[:2])

    if not test_images:
        print("No test images found in dataset/test/clean or dataset/test/stego.")
        return

    for image_path in test_images:
        result = predictor.analyze_image(image_path)

        print(f"\nImage: {result['image_path']}")

        if not result["success"]:
            print("Error:", result["error"])
            continue

        print("Prediction:", result["prediction"])
        print(f"Confidence: {result['confidence']:.2%}")

        if result["hidden_message"]:
            print("Extracted message:", result["hidden_message"])
        elif result["prediction"] == "Stego":
            print("No message could be extracted.")

    print("\nModel information:")
    print(predictor.get_model_info())


# --------------------------------------------------
# 9. COMMAND-LINE INTERFACE
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LSB Feature-Based Steganography Detection"
    )

    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to the image you want to analyze",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the saved Random Forest model",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the classifier using sample dataset images",
    )

    args = parser.parse_args()

    if args.test:
        test_predictor()
        return

    if not args.image_path:
        parser.error("Provide an image path or use --test.")

    predictor = SteganographyPredictor(model_path=args.model)
    result = predictor.analyze_image(args.image_path)

    print("\nSteganography Detection Results")
    print("=" * 45)

    if not result["success"]:
        print("Error:", result["error"])
        return

    print("Image:", result["image_path"])
    print("Prediction:", result["prediction"])
    print(f"Confidence: {result['confidence']:.2%}")

    if result["prediction"] == "Stego":
        print(
            "\nThe image was classified as Stego by the model. "
            "This does not prove that a hidden message exists."
        )

        if result["hidden_message"]:
            print("\nExtracted message:")
            print(result["hidden_message"])
        else:
            print("No hidden message could be extracted.")
    else:
        print("\nThe image was classified as Clean by the model.")


if __name__ == "__main__":
    main()
