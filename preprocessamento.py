import os
from typing import Tuple, Dict
import numpy as np
import cv2


def load_dataset(data_dir: str, img_size: Tuple[int, int] = (64, 64)) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    X = []
    y = []
    label_map = {}

    for idx, cls in enumerate(classes):
        label_map[idx] = cls
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                img = img.astype("float32") / 255.0
                X.append(img)
                y.append(idx)
            except Exception:
                continue

    X = np.array(X)
    y = np.array(y)
    return X, y, label_map


def save_npz(X, y, out_path: str = "dataset.npz"):
    np.savez_compressed(out_path, X=X, y=y)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset", help="Pasta do dataset")
    parser.add_argument("--size", type=int, default=64, help="Tamanho das imagens (px)")
    args = parser.parse_args()
    X, y, mapping = load_dataset(args.data, (args.size, args.size))
    print("Formas:", X.shape, y.shape)
    save_npz(X, y)
