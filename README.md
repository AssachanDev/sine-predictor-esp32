# 🌊 Sine Wave Prediction on ESP32
[![PlatformIO](https://img.shields.io/badge/PlatformIO-Compatible-orange?style=for-the-badge&logo=platformio)](https://platformio.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/lite/microcontrollers)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end **TinyML** project demonstrating how to train a neural network in Python and deploy it to an ESP32 for real-time inference.

---

---

## 🎥 Demo
<div align="center">
  <video src="https://github.com/user-attachments/assets/76f0b787-3226-4e50-a557-f58485930ec4" width="100%" controls>
  </video>
</div>

---

## 🚀 Overview

This repository showcases the full lifecycle of an Edge AI application. By training a simple model to learn the mathematical `sin()` function, we can see how complex logic can be compressed into a lightweight model and executed on low-power hardware.

### ✨ Key Features
- 🧠 **Smart Logic**: Multi-layer perceptron (MLP) trained with Keras.
- ⚡ **Ultra-Fast**: Sub-millisecond inference on the ESP32.
- 📉 **Optimized**: Uses TensorFlow Lite for Microcontrollers (TFLM) to minimize RAM/Flash usage.
- 📈 **Real-time Monitoring**: Serial output shows Prediction vs. Ground Truth accuracy.

---

## 🛠️ Project Structure

```text
.
├── 🐍 train_model.py      # Python training script
├── 🔌 src/main.cpp        # ESP32 Inference logic
├── 📄 include/model_data.h # Compiled TFLite model array
├── ⚙️ platformio.ini     # Hardware configuration
└── 📦 sine_model.tflite   # Exported flatbuffer model
```

---

## 📥 Getting Started

### 1. Training (Python)
If you want to tweak the brain:
```bash
# Setup environment
pip install tensorflow numpy

# Train and export
python train_model.py
```

### 2. Conversion (Optional)
Convert the `.tflite` file into a C-header array:
```bash
xxd -i sine_model.tflite > include/model_data.h
```

### 3. Deployment (Hardware)
Flash the ESP32 using PlatformIO:
```bash
# Build and Upload
pio run --target upload

# Monitor output
pio device monitor
```

---

## 📊 Performance & Results

Once the device is running, open the Serial Monitor to see the AI in action:

| Angle (rad) | Actual Value | AI Prediction | Accuracy |
| :--- | :--- | :--- | :--- |
| `0.52` | `0.50` | `0.49` | `99.00%` |
| `-1.57` | `-1.00` | `-0.98` | `98.00%` |
| `2.10` | `0.86` | `0.85` | `98.80%` |

---

## 🔌 Hardware Requirements
* **Microcontroller:** ESP32 (DevKit V1 recommended)
* **Connection:** Micro-USB / USB-C cable

---

## 🤝 Contributing
Feel free to fork this project and experiment with more complex functions like `cos(x)` or even multi-input sensors!

---
*Built with ❤️ for the TinyML community.*
