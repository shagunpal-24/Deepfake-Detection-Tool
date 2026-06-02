from pathlib import Path
import shutil

import cv2
import numpy as np
import tensorflow as tf

from model import build_model


IMG_SIZE = 224
SEQUENCE_LENGTH = 10
DATASET_DIR = Path("dataset")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
THRESHOLD_FILE = Path("model_threshold.txt")
CLASS_NAMES = ("real", "fake")
VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
BATCH_SIZE = 4
SPLIT_RATIO = 0.8
SEED = 42


def load_media(path):
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    elif hasattr(path, "item"):
        path = path.item()
        if isinstance(path, bytes):
            path = path.decode("utf-8")
    path = Path(path)
    if path.suffix.lower() in IMAGE_TYPES:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        frames = [img]
    else:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
        cap.release()

    if not frames:
        raise ValueError(f"No readable frames found: {path}")

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])

    return np.asarray(frames[:SEQUENCE_LENGTH], dtype=np.float32) / 255.0


def collect_files(folder):
    allowed = VIDEO_TYPES | IMAGE_TYPES
    if not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def create_dataset_structure():
    for split in ("train", "val"):
        for class_name in CLASS_NAMES:
            (DATASET_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


def unique_destination(folder, file_path):
    destination = folder / file_path.name
    if not destination.exists():
        return destination

    counter = 1
    while True:
        candidate = folder / f"{file_path.stem}_{counter}{file_path.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_files(files, destination_folder):
    destination_folder.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        destination = unique_destination(destination_folder, file_path)
        shutil.move(str(file_path), str(destination))


def split_class_files(class_name):
    old_class_dir = DATASET_DIR / class_name
    train_class_dir = TRAIN_DIR / class_name
    val_class_dir = VAL_DIR / class_name

    old_files = collect_files(old_class_dir)
    if old_files:
        move_files(old_files, train_class_dir)

    train_files = collect_files(train_class_dir)
    val_files = collect_files(val_class_dir)
    if val_files or len(train_files) < 2:
        return

    rng = np.random.default_rng(SEED)
    shuffled = list(train_files)
    rng.shuffle(shuffled)

    split_index = int(len(shuffled) * SPLIT_RATIO)
    split_index = min(max(split_index, 1), len(shuffled) - 1)
    move_files(shuffled[split_index:], val_class_dir)


def prepare_dataset_folders():
    create_dataset_structure()
    for class_name in CLASS_NAMES:
        split_class_files(class_name)


def collect_labeled_files(split_dir):
    samples = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_files = collect_files(split_dir / class_name)
        samples.extend((path, label) for path in class_files)
        print(f"{split_dir.name}/{class_name} samples: {len(class_files)}")
    return samples


def load_media_tf(path, label):
    frames = tf.numpy_function(load_media, [path], tf.float32)
    frames.set_shape((SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    label = tf.cast(label, tf.float32)
    label.set_shape(())
    return frames, label


def make_tf_dataset(samples, shuffle=False):
    paths = [str(path) for path, _ in samples]
    labels = [label for _, label in samples]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(samples), seed=SEED, reshuffle_each_iteration=True)
    return (
        dataset
        .map(load_media_tf, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def class_counts(samples):
    labels = np.asarray([label for _, label in samples], dtype=np.float32)
    return int(np.sum(labels == 0)), int(np.sum(labels == 1))


def build_datasets():
    prepare_dataset_folders()
    train_samples = collect_labeled_files(TRAIN_DIR)
    val_samples = collect_labeled_files(VAL_DIR)

    train_real, train_fake = class_counts(train_samples)
    if train_real == 0 or train_fake == 0:
        raise SystemExit(
            "Training needs both classes. Put files in dataset/real and dataset/fake, "
            "or in dataset/train/real and dataset/train/fake."
        )

    train_dataset = make_tf_dataset(train_samples, shuffle=True)
    val_dataset = make_tf_dataset(val_samples, shuffle=False) if val_samples else None
    return train_dataset, val_dataset, train_samples, val_samples


def class_weights(samples):
    real_count, fake_count = class_counts(samples)
    total = len(samples)
    return {
        0: total / (2 * max(real_count, 1)),
        1: total / (2 * max(fake_count, 1)),
    }


def tune_threshold(model, val_dataset, val_samples):
    if not val_samples:
        print("WARNING: Validation data is empty. Skipping threshold calculation.")
        return None

    val_real, val_fake = class_counts(val_samples)
    if val_real == 0 or val_fake == 0:
        print("WARNING: Validation data needs both real and fake samples. Skipping threshold calculation.")
        return None

    y_true = np.asarray([label for _, label in val_samples], dtype=np.int32)
    y_scores = model.predict(val_dataset, verbose=0).reshape(-1)

    best_threshold = 0.5
    best_accuracy = -1.0
    best_matrix = None

    for threshold in np.arange(0.30, 0.701, 0.01):
        y_pred = (y_scores >= threshold).astype(np.int32)
        accuracy = float(np.mean(y_pred == y_true))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
            best_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Validation accuracy: {best_accuracy:.4f}")
    print("Confusion matrix:")
    print(best_matrix)
    print("Matrix labels: rows=true [real, fake], columns=predicted [real, fake]")
    THRESHOLD_FILE.write_text(f"{best_threshold:.4f}\n", encoding="utf-8")
    print(f"Saved threshold to {THRESHOLD_FILE}")
    return best_threshold


def main():
    train_dataset, val_dataset, train_samples, val_samples = build_datasets()
    weights = class_weights(train_samples)

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Class weights: {weights}")

    model = build_model()
    monitor_metric = "val_loss" if val_dataset is not None else "loss"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "deepfake_model.h5",
            monitor=monitor_metric,
            save_best_only=True,
        ),
    ]

    fit_kwargs = {
        "x": train_dataset,
        "epochs": 10,
        "class_weight": weights,
        "callbacks": callbacks,
    }
    if val_dataset is not None:
        fit_kwargs["validation_data"] = val_dataset
    else:
        print("WARNING: Validation data is empty. Training without validation data.")

    history = model.fit(
        **fit_kwargs,
    )

    model.save("deepfake_model.h5")
    print("Model trained and saved as deepfake_model.h5")
    print("Final train accuracy:", history.history["accuracy"][-1])
    if "val_accuracy" in history.history:
        print("Final validation accuracy:", history.history["val_accuracy"][-1])

    if val_dataset is None:
        print("WARNING: Validation data is empty. Skipping threshold calculation.")
    else:
        tune_threshold(model, val_dataset, val_samples)


if __name__ == "__main__":
    main()
