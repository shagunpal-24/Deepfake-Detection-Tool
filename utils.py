import csv
from datetime import datetime
from pathlib import Path

import numpy as np


PREDICTION_LOG = Path("prediction_debug_log.csv")
MIN_SEQUENCE_LENGTH = 10


def log_prediction(source, fake_probability, threshold, label):
    file_exists = PREDICTION_LOG.exists()
    with PREDICTION_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "source", "fake_probability", "real_probability", "threshold", "label"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source": source,
                "fake_probability": f"{fake_probability:.6f}",
                "real_probability": f"{1.0 - fake_probability:.6f}",
                "threshold": f"{threshold:.2f}",
                "label": label,
            }
        )


def prediction_distribution():
    if not PREDICTION_LOG.exists():
        return None

    scores = []
    with PREDICTION_LOG.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["fake_probability"]))

    if not scores:
        return None

    values = np.asarray(scores, dtype=np.float32)
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def predict_details(model, frames, threshold=0.5, source="unknown", debug=True):
    if len(frames) == 0:
        raise ValueError("No frames were available for prediction.")

    frames = list(frames)
    while len(frames) < MIN_SEQUENCE_LENGTH:
        frames.append(frames[-1])

    frames = np.asarray(frames, dtype=np.float32)
    frames = np.expand_dims(frames, axis=0)
    fake_probability = float(model.predict(frames, verbose=0)[0][0])
    real_probability = 1.0 - fake_probability
    label = "Fake" if fake_probability > threshold else "Real"
    confidence = fake_probability if label == "Fake" else real_probability

    if debug:
        safe_source = str(source).encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"Prediction score: {fake_probability:.4f} -> {label} (threshold={threshold:.2f}, source={safe_source})")
        log_prediction(source, fake_probability, threshold, label)

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "threshold": threshold,
    }


def predict(model, frames, threshold=0.5, source="unknown"):
    details = predict_details(model, frames, threshold=threshold, source=source)
    return details["label"], details["confidence"]
