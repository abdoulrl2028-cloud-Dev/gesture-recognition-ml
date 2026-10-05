<p align="center">
  <img src="https://raw.githubusercontent.com/abdoulrl2028-cloud-Dev/abdoulrl2028-cloud-Dev/main/assets/projects/gesture.jpg" alt="Gesture recognition" width="100%">
</p>

# Gesture Recognition ML

Sample project that captures, preprocesses, trains, and detects gestures in real time with a webcam. Built with Python, OpenCV, and machine learning.

## Structure

- `captura.py` — collects gesture images from the webcam into `dataset/<gesture>`
- `preprocessamento.py` — loads images and prepares NumPy arrays
- `modelo.py` — defines and trains a CNN (Keras/TensorFlow)
- `detector.py` — loads the saved model and runs live inference
- `automacao.py` — maps detected gestures to actions (`pyautogui`)
- `main.py` — CLI for `collect`, `train`, and `detect`
- `requirements.txt` — dependencies

## Examples

Collect images for a gesture:

```bash
python main.py collect thumbs_up --samples 300
```

Train from the `dataset` folder:

```bash
python main.py train --data dataset --out model --epochs 20
```

Detect live and run the mapped actions:

```bash
python main.py detect --model-dir model --threshold 0.75
```

## Notes

- Change `img_size` in `preprocessamento.py` and `detector.py` if you want a different resolution.
- `pyautogui` may need extra permissions. On Linux you may need system packages.
- For a small dataset, raise `epochs` and add augmentation yourself.
