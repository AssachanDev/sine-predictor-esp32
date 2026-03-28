import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- Configuration ---
NUM_SAMPLES = 2000
EPOCHS = 500
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- Data Generation ---
angles = np.random.uniform(low=-np.pi, high=np.pi, size=(NUM_SAMPLES, 1))
sines = np.sin(angles)

# --- Model Definition ---
model = keras.Sequential(
    [
        keras.layers.Dense(16, activation="relu", input_shape=(1,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ],
    name="sine_predictor",
)

model.compile(optimizer="adam", loss="mae", metrics=["mse"])
model.summary()

# --- Training ---
print("\nTraining...")
history = model.fit(
    angles,
    sines,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=0,
)

final_loss = history.history["loss"][-1]
final_val_loss = history.history["val_loss"][-1]
print(f"Train MAE: {final_loss:.5f}  |  Val MAE: {final_val_loss:.5f}")

# --- Evaluation ---
test_angles = np.array([[0.0], [np.pi / 2], [-np.pi / 2], [np.pi / 4]])
for a in test_angles:
    pred = model.predict(a.reshape(1, 1), verbose=0)[0][0]
    actual = np.sin(a[0])
    print(f"  sin({a[0]:.4f}) → Actual: {actual:.4f}, Predicted: {pred:.4f}, Error: {abs(actual - pred):.4f}")

# --- Save Keras Model ---
model.save("sine_model.h5")
print("\nKeras model saved as sine_model.h5")

# --- TFLite Conversion (with quantization) ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

representative_data = np.random.uniform(-np.pi, np.pi, (500, 1)).astype(np.float32)

def representative_dataset():
    for sample in representative_data:
        yield [sample.reshape(1, 1)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open("sine_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved as sine_model.tflite ({len(tflite_model):,} bytes)")
print("\nDone! Run: xxd -i sine_model.tflite > include/model_data.h")
