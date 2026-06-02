import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_loader import extract_frames, load_image
from utils import predict_details


VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}


def collect_files(folder):
    allowed = VIDEO_TYPES | IMAGE_TYPES
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in allowed)


def load_sample(path, max_frames):
    if path.suffix.lower() in IMAGE_TYPES:
        return [load_image(str(path))]
    return extract_frames(str(path), max_frames=max_frames)


def evaluate_folder(model, folder, true_label, threshold, max_frames, limit=None):
    rows = []
    files = collect_files(folder)
    if limit:
        files = files[:limit]

    for path in files:
        try:
            frames = load_sample(path, max_frames=max_frames)
            details = predict_details(model, frames, threshold=threshold, source=str(path))
        except (OSError, ValueError) as exc:
            print(f"Skipping {path}: {exc}")
            continue
        predicted_label = 1 if details["label"] == "Fake" else 0
        rows.append(
            {
                "file": path.name,
                "true": true_label,
                "predicted": predicted_label,
                "fake_probability": details["fake_probability"],
                "real_probability": details["real_probability"],
                "label": details["label"],
            }
        )
    return rows


def confusion_counts(rows, threshold=None):
    if threshold is not None:
        for row in rows:
            row["predicted"] = 1 if row["fake_probability"] > threshold else 0

    tp = sum(row["true"] == 1 and row["predicted"] == 1 for row in rows)
    tn = sum(row["true"] == 0 and row["predicted"] == 0 for row in rows)
    fp = sum(row["true"] == 0 and row["predicted"] == 1 for row in rows)
    fn = sum(row["true"] == 1 and row["predicted"] == 0 for row in rows)
    return tp, tn, fp, fn


def print_confusion(rows, threshold=None):
    tp, tn, fp, fn = confusion_counts(rows, threshold=threshold)
    total = max(len(rows), 1)

    label = f" @ threshold {threshold:.2f}" if threshold is not None else ""
    print(f"\nConfusion matrix{label}")
    print("                 Pred Real   Pred Fake")
    print(f"Actual Real      {tn:9d}   {fp:9d}")
    print(f"Actual Fake      {fn:9d}   {tp:9d}")
    print(f"Accuracy: {(tp + tn) / total:.3f}")
    return (tp + tn) / total


def threshold_sweep(rows):
    print("\nThreshold sweep")
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in np.arange(0.30, 0.71, 0.05):
        accuracy = print_confusion(rows, threshold=float(threshold))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    print(f"\nSuggested threshold from this validation set: {best_threshold:.2f}")
    print("Use a separate validation set, not your training samples, when choosing this value.")


def main():
    parser = argparse.ArgumentParser(description="Debug deepfake model predictions.")
    parser.add_argument("--real", default="dataset/train/real", help="Folder with known real samples")
    parser.add_argument("--fake", default="dataset/train/fake", help="Folder with known fake samples")
    parser.add_argument("--model", default="deepfake_model.h5", help="Model path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fake threshold")
    parser.add_argument("--max-frames", type=int, default=30, help="Frames to sample from videos")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per class, e.g. 5")
    parser.add_argument("--sweep", action="store_true", help="Test thresholds from 0.30 to 0.70")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    rows = []
    rows.extend(evaluate_folder(model, Path(args.real), 0, args.threshold, args.max_frames, args.limit))
    rows.extend(evaluate_folder(model, Path(args.fake), 1, args.threshold, args.max_frames, args.limit))

    if not rows:
        raise SystemExit(
            "No validation samples found. Add files to dataset/real and dataset/fake, "
            "then run this again. Example: python diagnose_model.py --limit 5 --sweep"
        )

    print("\nRaw predictions")
    for row in rows:
        true_name = "Fake" if row["true"] == 1 else "Real"
        safe_file = row["file"].encode("ascii", errors="backslashreplace").decode("ascii")
        print(
            f"{safe_file}: true={true_name:4s} predicted={row['label']:4s} "
            f"fake_prob={row['fake_probability']:.4f} real_prob={row['real_probability']:.4f}"
        )

    print_confusion(rows)
    if args.sweep:
        threshold_sweep(rows)

    fake_probs = np.array([row["fake_probability"] for row in rows], dtype=np.float32)
    if len(fake_probs):
        print("\nFake probability summary")
        print(f"min={fake_probs.min():.4f} mean={fake_probs.mean():.4f} max={fake_probs.max():.4f}")
        print("Try thresholds between 0.30 and 0.70 and compare the confusion matrix.")


if __name__ == "__main__":
    main()
