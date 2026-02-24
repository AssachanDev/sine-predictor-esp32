import numpy as np
import tensorflow as tf
from tensorflow import keras

angles = np.random.uniform(low=-np.pi, high=np.pi, size=(2000, 1))
sines = np.sin(angles)

model = keras.Sequential(
    [
        keras.layers.Dense(16, activation="relu", input_shape=(1,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mae")

print("Training...")
model.fit(angles, sines, epochs=500, verbose=0)

test_angle = np.array([[1.57]])
print(f"Test 1.57 (90 deg): {model.predict(test_angle)}")

model.save("sine_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("sine_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model saved as sine_model.tflite")
