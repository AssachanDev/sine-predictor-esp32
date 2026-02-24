# Sine Wave Prediction with TensorFlow Lite for Microcontrollers (ESP32)

This project demonstrates a complete end-to-end workflow for TinyML: training a simple neural network to predict sine wave values and deploying it to an ESP32 microcontroller using TensorFlow Lite for Microcontrollers.

## Project Overview

The project consists of two main parts:
1.  **Training**: A Python script using TensorFlow/Keras to train a model that approximates the sine function.
2.  **Deployment**: An Arduino/C++ project that runs the trained model on an ESP32 to perform real-time inference.

## Project Structure

- `train_model.py`: Python script to train the model and export it to `.tflite` format.
- `src/main.cpp`: ESP32 source code that initializes the TFLite interpreter and runs inference.
- `include/model_data.h`: The trained model exported as a C header file (byte array).
- `platformio.ini`: PlatformIO configuration file.
- `sine_model.h5` / `sine_model.tflite`: Trained model files.

## Prerequisites

### Python (Training)
- Python 3.8+
- TensorFlow 2.x
- NumPy

### Hardware (Deployment)
- ESP32 Development Board (e.g., ESP32 DevKit V1)
- USB Cable

### Software (Deployment)
- [PlatformIO](https://platformio.org/) (recommended as a VS Code extension)

## Getting Started

### 1. Train the Model
If you want to re-train the model, run:
```bash
python train_model.py
```
This will generate `sine_model.tflite`.

### 2. Convert Model to C Header
To use the model on a microcontroller, it must be converted to a C array. You can use the `xxd` tool:
```bash
xxd -i sine_model.tflite > include/model_data.h
```
*Note: You may need to update the variable names in `include/model_data.h` to match those used in `src/main.cpp` (`sine_model_tflite` and `sine_model_tflite_len`).*

### 3. Build and Upload
1.  Connect your ESP32 to your computer.
2.  Open the project in VS Code with PlatformIO.
3.  Click the **Upload** button (arrow icon in the bottom status bar) or run:
    ```bash
    pio run --target upload
    ```

## Results
Once uploaded, open the Serial Monitor (baud rate `115200`). You will see the ESP32 generating random angles, predicting their sine values using the AI model, and comparing them to the actual values:

```text
Angle:0.52,Actual:0.50,AI:0.49,Err:0.0100,Acc:99.00%
Angle:-1.57,Actual:-1.00,AI:-0.98,Err:0.0200,Acc:98.00%
```

## How it Works
- **Model Architecture**: A simple Multi-Layer Perceptron (MLP) with one input, two hidden layers of 16 neurons (ReLU activation), and one output.
- **Inference**: The ESP32 uses the `TensorFlowLite_ESP32` library to load the model from memory and perform inference in the `loop()` function.
- **Optimization**: The model is converted to TFLite format to minimize memory footprint, making it suitable for resource-constrained microcontrollers.
