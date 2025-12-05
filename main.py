import argparse
import os

from captura import capture_dataset
from preprocessamento import load_dataset
from modelo import train_model
from detector import Detector
from automacao import perform_action


def cmd_collect(args):
    capture_dataset(args.gesture, args.out, samples=args.samples)


def cmd_train(args):
    X, y, mapping = load_dataset(args.data)
    train_model(X, y, mapping, output_dir=args.out, epochs=args.epochs)


def cmd_detect(args):
    model_path = os.path.join(args.model_dir, "best_model.h5")
    labels = os.path.join(args.model_dir, "label_map.json")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, "final_model.h5")
    det = Detector(model_path, labels)
    cap = cv2.VideoCapture(0)
    try:
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
            cv2.imshow("Detect", frame)
            if prob > args.threshold:
                perform_action(label)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import cv2

    parser = argparse.ArgumentParser(description="Gestos-ML: coleta, treino e detecção")
    sub = parser.add_subparsers(dest="cmd")

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("gesture")
    p_collect.add_argument("--out", default="dataset")
    p_collect.add_argument("--samples", type=int, default=200)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", default="dataset")
    p_train.add_argument("--out", default="model")
    p_train.add_argument("--epochs", type=int, default=15)

    p_detect = sub.add_parser("detect")
    p_detect.add_argument("--model-dir", default="model")
    p_detect.add_argument("--threshold", type=float, default=0.7)

    args = parser.parse_args()
    if args.cmd == "collect":
        cmd_collect(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "detect":
        cmd_detect(args)
    else:
        parser.print_help()
