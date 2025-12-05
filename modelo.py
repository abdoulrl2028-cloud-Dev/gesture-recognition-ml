import json
from typing import Tuple, Dict
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split


def build_model(input_shape: Tuple[int, int, int], num_classes: int):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(X: np.ndarray, y: np.ndarray, label_map: Dict[int, str], output_dir: str = "model", epochs: int = 15, batch_size: int = 32):
    import os
    os.makedirs(output_dir, exist_ok=True)

    num_classes = len(set(y))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    model = build_model(X.shape[1:], num_classes)

    checkpoint = ModelCheckpoint(os.path.join(output_dir, "best_model.h5"), save_best_only=True, monitor="val_loss")
    early = EarlyStopping(patience=6, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early])

    model.save(os.path.join(output_dir, "final_model.h5"))
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)

    return model


if __name__ == "__main__":
    import argparse
    from preprocessamento import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset", help="Pasta do dataset")
    parser.add_argument("--out", default="model", help="Pasta de saída do modelo")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    X, y, mapping = load_dataset(args.data)
    train_model(X, y, mapping, args.out, epochs=args.epochs)
