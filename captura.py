import cv2
import os
import time


def capture_dataset(gesture_name: str, out_dir: str = "dataset", samples: int = 200, cam_index: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    gesture_dir = os.path.join(out_dir, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera")

    print(f"Coletando {samples} imagens para gesto '{gesture_name}' em {gesture_dir}")
    count = 0
    start = time.time()

    while count < samples:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        box_size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        x1, y1 = cx - box_size // 2, cy - box_size // 2
        x2, y2 = x1 + box_size, y1 + box_size

        roi = frame[y1:y2, x1:x2]
        img_path = os.path.join(gesture_dir, f"{count:04d}.jpg")
        cv2.imwrite(img_path, roi)
        count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{gesture_name}: {count}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Capture - Press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start
    print(f"Coleta finalizada: {count} imagens em {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Captura imagens para dataset de gestos")
    parser.add_argument("gesture", help="Nome do gesto (nome da pasta)")
    parser.add_argument("--out", default="dataset", help="Pasta de saída")
    parser.add_argument("--samples", type=int, default=200, help="Número de imagens a capturar")
    args = parser.parse_args()

    capture_dataset(args.gesture, args.out, args.samples)
