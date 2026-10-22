
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "dataset" / "test"
MODEL_PATH = ROOT / "model" / "stego_detector.keras"
IMG_SIZE = (96, 96)
BATCH_SIZE = 32

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

for label in ("clean", "stego"):
    if not (TEST_DIR / label).is_dir():
        raise FileNotFoundError(f"Missing test folder: {TEST_DIR / label}")

# Keep class ordering explicit: clean=0, stego=1
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=["clean", "stego"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = np.concatenate([
    labels.numpy().ravel() for _, labels in test_ds
]).astype(int)

probabilities = model.predict(test_ds, verbose=1).ravel()
y_pred = (probabilities >= 0.5).astype(int)

print("\n=== SYNTHETIC TEST SET RESULTS ===")
print("Test images:", len(y_true))
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")

print("\nConfusion matrix (rows=actual, columns=predicted):")
print("Labels: [clean, stego]")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Clean", "Stego"],
    zero_division=0,
))

print(
    "\nNote: These results measure performance on the project's synthetic "
    "test dataset, not on real-world images."
)
