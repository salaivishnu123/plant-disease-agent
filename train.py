"""
train.py
--------
Trains a lightweight MobileNetV2 transfer-learning classifier over the
combined Areca Nut / Coconut / Mango dataset.

Expects this layout (see README.md for dataset sources):

    dataset/
      train/<Plant>___<Disease>/*.jpg
      val/<Plant>___<Disease>/*.jpg
      test/<Plant>___<Disease>/*.jpg

Produces:
      weights/best_model.keras
      models/class_indices.json     (index -> "Plant___Disease")
      models/training_history.png
      models/confusion_matrix.png
      models/classification_report.txt
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Config — tweak these for your hardware
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)      # drop to (160, 160) if training is too slow
BATCH_SIZE = 16            # drop to 8 on very low-RAM machines
EPOCHS_HEAD = 12           # phase 1: train only the new head
EPOCHS_FINETUNE = 8        # phase 2: unfreeze top backbone layers
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 1e-5
UNFREEZE_LAST_N_LAYERS = 30

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

WEIGHTS_DIR = "weights"
MODELS_DIR = "models"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.keras")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")


def validate_dataset_layout(base_dir=DATASET_DIR):
    """Verify that train/val/test contain class folders with image files."""
    required_dirs = {
        "train": os.path.join(base_dir, "train"),
        "val": os.path.join(base_dir, "val"),
        "test": os.path.join(base_dir, "test"),
    }

    for split_name, split_dir in required_dirs.items():
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Missing dataset split folder: {split_dir}. Create the folder and populate it with class subfolders such as Mango___Healthy."
            )

        class_dirs = [
            os.path.join(split_dir, name)
            for name in sorted(os.listdir(split_dir))
            if os.path.isdir(os.path.join(split_dir, name))
        ]
        if not class_dirs:
            raise ValueError(
                f"No class folders found in {split_dir}. Expected subfolders like Mango___Healthy or Coconut___Bud_Rot."
            )

        for class_dir in class_dirs:
            image_files = [
                name
                for name in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, name))
                and os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
            if not image_files:
                raise ValueError(
                    f"Class folder {class_dir} has no image files. Add at least one image before training."
                )

    return required_dirs


def build_generators():
    """Augmentation on train only; val/test are just rescaled."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
    )
    val_gen = eval_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )
    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )
    return train_gen, val_gen, test_gen


def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # phase 1: frozen backbone

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def compute_weights(train_gen):
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))


def plot_history(history_phase1, history_phase2, path):
    acc = history_phase1.history["accuracy"] + history_phase2.history["accuracy"]
    val_acc = history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
    loss = history_phase1.history["loss"] + history_phase2.history["loss"]
    val_loss = history_phase1.history["val_loss"] + history_phase2.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.savefig(path)
    plt.close(fig)


def evaluate(model, test_gen, class_names):
    test_gen.reset()
    probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    with open(os.path.join(MODELS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.5), max(6, len(class_names) * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
    plt.close(fig)


def main():
    validate_dataset_layout()
    train_gen, val_gen, test_gen = build_generators()
    num_classes = len(train_gen.class_indices)
    if num_classes <= 0:
        raise ValueError(
            "No classes were discovered from the dataset. Ensure each split contains class folders with image files."
        )
    print(f"Found {num_classes} classes: {list(train_gen.class_indices.keys())}")

    # Save index -> class name mapping for the agents to use at inference time
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open(CLASS_INDICES_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=2)

    class_weights = compute_weights(train_gen)

    model, base_model = build_model(num_classes)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    # ---- Phase 1: train head only ----
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ---- Phase 2: fine-tune top backbone layers ----
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_FINETUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    plot_history(history1, history2, os.path.join(MODELS_DIR, "training_history.png"))

    class_names = [idx_to_class[i] for i in range(num_classes)]
    evaluate(model, test_gen, class_names)

    print(f"\nDone. Best model saved to {BEST_MODEL_PATH}")
    print(f"Class mapping saved to {CLASS_INDICES_PATH}")


if __name__ == "__main__":
    main()
