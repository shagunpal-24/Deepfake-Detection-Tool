# Deepfake Detection Tool

## Overview
Deepfake Detection Tool is a deep learning-based application designed to identify manipulated facial images and distinguish them from authentic images. The system leverages computer vision and machine learning techniques to detect AI-generated content.

## Features
- Detects real and deepfake images
- Image preprocessing and feature extraction
- Deep learning-based classification
- User-friendly interface
- Fast prediction results

## Tech Stack
- Python
- TensorFlow
- OpenCV
- NumPy
- Streamlit

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Project Structure

- `app.py` - Main application
- `model.py` - Model architecture
- `train.py` - Model training
- `data_loader.py` - Dataset loading
- `utils.py` - Helper functions
- `deepfake_model.h5` - Trained model

## Future Improvements
- Video deepfake detection
- Real-time webcam detection
- Improved model accuracy
- Cloud deployment

## Author
Shagun Pal# Deepfake Detection Tool

## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Train model:
   python train.py

3. Run app:
   streamlit run app.py

## Features
- Detect deepfake in images and videos
- CNN + LSTM model
- Simple UI using Streamlit
