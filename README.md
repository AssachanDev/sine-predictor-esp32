# Sine Wave Prediction on ESP32

[![PlatformIO](https://img.shields.io/badge/PlatformIO-Compatible-orange?style=for-the-badge&logo=platformio)](https://platformio.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/lite/microcontrollers)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end **TinyML** project: train a neural network in Python, convert it to TensorFlow Lite, and run real-time inference on an ESP32 microcontroller.

---

## Demo

<div align="center">
  <video src="https://github.com/user-attachments/assets/76f0b787-3226-4e50-a557-f58485930ec4" width="100%" controls>
  </video>
</div>

---

## Overview

This project demonstrates the complete Edge AI lifecycle:

1. **Train** — a small MLP learns to approximate `sin(x)` using Keras
2. **Quantize & Convert** — TFLite post-training quantization shrinks the model to ~3 KB
3. **Deploy** — the model runs on-device via TensorFlow Lite for Microcontrollers (TFLM)
4. **Monitor** — serial output streams predictions vs. ground truth in CSV format

### Key Features

- **MLP architecture** — two 16-neuron hidden layers with ReLU, trained with Adam + MAE loss
- **Post-training quantization** — INT8 weights with float I/O for smaller Flash footprint
- **Reproducible training** — fixed random seeds for consistent results
- **Validation split** — 20% holdout set reported during training
- **Robust firmware** — schema version check, error logging, hardware-seeded RNG, CSV serial output

---

## Project Structure

```text
.
├── train_model.py          # Python: train, evaluate, and export the model
├── src/main.cpp            # ESP32: load model and run inference loop
├── include/model_data.h    # Auto-generated: TFLite model as a C array
├── platformio.ini          # PlatformIO board configuration
└── sine_model.tflite       # Exported TFLite flatbuffer
```

---

## Model Architecture

```
Input (1)  →  Dense(16, ReLU)  →  Dense(16, ReLU)  →  Dense(1)
```

| Layer  | Neurons | Activation |
| :----- | :------ | :--------- |
| Hidden 1 | 16    | ReLU       |
| Hidden 2 | 16    | ReLU       |
| Output   | 1     | Linear     |

Trained on 2 000 random angles in `[−π, π]` for 500 epochs with an 80/20 train-validation split.

---

## Getting Started

### Prerequisites

- Python 3.8+
- [PlatformIO CLI](https://docs.platformio.org/en/latest/core/installation/index.html)
- ESP32 DevKit (any variant)

---

### 1. Train the Model

```bash
pip install tensorflow numpy

python train_model.py
```

Expected output:

```
Train MAE: 0.00412  |  Val MAE: 0.00489
  sin(0.0000) → Actual: 0.0000, Predicted: 0.0012, Error: 0.0012
  sin(1.5708) → Actual: 1.0000, Predicted: 0.9983, Error: 0.0017
  ...
TFLite model saved as sine_model.tflite (2,960 bytes)
```

---

### 2. Convert to C Header

Embed the TFLite binary into firmware as a C array:

```bash
xxd -i sine_model.tflite > include/model_data.h
```

> This step is only needed after retraining. The file is pre-generated and tracked in the repository.

---

### 3. Flash the ESP32

```bash
# Build and upload firmware
pio run --target upload

# Open serial monitor (115200 baud)
pio device monitor
```

---

## Serial Output

The firmware prints CSV-formatted data to serial — easy to pipe into a plotter or logger:

```
Sine Predictor ready.
Angle,Actual,AI,Error,Accuracy
0.5236,0.5000,0.4981,0.0019,99.91%
-1.5708,-1.0000,-0.9977,0.0023,99.88%
2.0944,0.8660,0.8642,0.0018,99.91%
```

### Accuracy Formula

```
accuracy = clamp((1 − |error| / 2) × 100, 0, 100)
```

Error is normalised by the full sine range `[−1, 1]` (width = 2), so accuracy is always in `[0 %, 100 %]`.

---

## Performance

| Angle (rad) | Actual | Predicted | Abs Error |
| :---------- | :----- | :-------- | :-------- |
| `0.52`      | `0.500` | `0.498`  | `0.002`   |
| `-1.57`     | `-1.000` | `-0.998` | `0.002`  |
| `2.09`      | `0.866` | `0.864`  | `0.002`   |

Typical MAE on validation set: **< 0.005**

---

## Hardware Requirements

| Component | Specification |
| :-------- | :------------ |
| Microcontroller | ESP32 (DevKit V1 recommended) |
| Flash | ≥ 4 MB |
| Connection | Micro-USB or USB-C |

---

## License

[MIT](https://opensource.org/licenses/MIT)
