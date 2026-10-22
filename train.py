"""Train a compact CNN using dataset/{train,val}/{clean,stego} folders.
Requires the optional dependencies in requirements-ml.txt.
"""
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dataset'
MODEL_DIR = ROOT / 'model'
MODEL_DIR.mkdir(exist_ok=True)
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 30

for split in ('train', 'val'):
    for label in ('clean', 'stego'):
        if not (DATA / split / label).is_dir():
            raise SystemExit(f'Missing folder: {DATA / split / label}. Run python generate_dataset.py first.')

train_ds = tf.keras.utils.image_dataset_from_directory(DATA / 'train', image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', seed=42)
print("Class names:", train_ds.class_names)
print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
val_ds = tf.keras.utils.image_dataset_from_directory(DATA / 'val', image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(24, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(48, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(96, 3, padding='same', activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / 'stego_detector.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f'Model saved to {MODEL_DIR / "stego_detector.keras"}')
print('Evaluate on a separate real-image test set before reporting performance.')
