import cv2
import numpy as np

IMG_SIZE = 224
SEQUENCE_LENGTH = 10


def preprocess_frame(frame, color_format="BGR"):
    if frame is None:
        raise ValueError("Could not read this frame.")

    if color_format.upper() == "BGR":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif color_format.upper() != "RGB":
        raise ValueError("color_format must be BGR or RGB.")

    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    return frame.astype(np.float32) / 255.0


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame, color_format="BGR"))
    
    cap.release()
    return np.asarray(frames, dtype=np.float32)


def load_image(image_path):
    img = read_image(image_path)
    if img is None:
        raise ValueError("Could not read this image. Please upload a valid JPG, PNG, or WEBP file.")

    return preprocess_frame(img, color_format="BGR")
