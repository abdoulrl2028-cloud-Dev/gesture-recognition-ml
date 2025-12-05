import json
import numpy as np
import cv2
from typing import Tuple

from tensorflow.keras.models import load_model


class Detector:
    def __init__(self, model_path: str, label_map_path: str, img_size: Tuple[int, int] = (64, 64)):
        self.model = load_model(model_path)
        with open(label_map_path, "r") as f:
            lm = json.load(f)
        self.label_map = {int(k): v for k, v in lm.items()}
        self.img_size = img_size

    def preprocess(self, frame: np.ndarray):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, 0)

    def predict(self, frame: np.ndarray):
        x = self.preprocess(frame)
        probs = self.model.predict(x)[0]
        idx = int(np.argmax(probs))
        return self.label_map.get(idx, "unknown"), float(probs[idx])


if __name__ == "__main__":
    import argparse
    from capture import capture_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    det = Detector(args.model, args.labels)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        box_size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        x1, y1 = cx - box_size // 2, cy - box_size // 2
        x2, y2 = x1 + box_size, y1 + box_size
        roi = frame[y1:y2, x1:x2]
        label, prob = det.predict(roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {prob:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
