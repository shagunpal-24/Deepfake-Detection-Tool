import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, LSTM, Rescaling, TimeDistributed
from tensorflow.keras.models import Sequential


def build_model():
    feature_extractor = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    feature_extractor.trainable = False

    model = Sequential(
        [
            Input(shape=(None, 224, 224, 3)),
            Rescaling(2.0, offset=-1.0),
            TimeDistributed(feature_extractor),
            TimeDistributed(GlobalAveragePooling2D()),
            LSTM(64),
            Dropout(0.35),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
