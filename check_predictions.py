
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "stego_detector.keras"
TEST_DIR = ROOT / "dataset" / "test"

ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=["clean", "stego"],
    image_size=(96, 96),
    batch_size=32,
    shuffle=False,
    label_mode="binary",
)

model = tf.keras.models.load_model(MODEL_PATH)
probabilities = model.predict(ds, verbose=1).ravel()
labels = np.concatenate([
    y.numpy().ravel() for _, y in ds
]).astype(int)

for label, name in [(0, "Clean"), (1, "Stego")]:
    p = probabilities[labels == label]
    print(f"\n{name} images:")
    print(f"Mean stego probability: {p.mean():.4f}")
    print(f"Minimum probability: {p.min():.4f}")
    print(f"Maximum probability: {p.max():.4f}")
    print(f"Predicted as stego: {(p >= 0.5).sum()}/{len(p)}")
