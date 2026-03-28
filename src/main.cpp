#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino.h>
#include <math.h>

// --- Constants ---
constexpr int kTensorArenaSize = 4096;  // bytes allocated for model tensors
constexpr long kBaudRate = 115200;
constexpr float kPi = 3.14159265f;
constexpr unsigned long kInferenceDelayMs = 1000;

// --- TFLite Globals ---
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
tflite::AllOpsResolver resolver;

uint8_t tensor_arena[kTensorArenaSize];

// Returns a random float in [min, max]
float randomFloat(float min, float max) {
  return min + (max - min) * ((float)random(0, 100000) / 100000.0f);
}

void setup() {
  Serial.begin(kBaudRate);
  randomSeed(analogRead(0));  // seed from floating analog pin for better randomness

  // Load and validate model
  model = tflite::GetModel(sine_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema mismatch: expected %d, got %d",
                           TFLITE_SCHEMA_VERSION, model->version());
    while (true) {}  // halt — unrecoverable
  }

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensor memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors failed");
    while (true) {}  // halt — unrecoverable
  }

  Serial.println("Sine Predictor ready.");
  Serial.println("Angle,Actual,AI,Error,Accuracy");
}

void loop() {
  float angle = randomFloat(-kPi, kPi);

  // Set input
  TfLiteTensor *input = interpreter->input(0);
  input->data.f[0] = angle;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    delay(kInferenceDelayMs);
    return;
  }

  // Read output
  TfLiteTensor *output = interpreter->output(0);
  float prediction = output->data.f[0];

  // Compute metrics
  float actual = sinf(angle);
  float abs_error = fabsf(actual - prediction);

  // Accuracy as % of max possible sine range [−1, 1], clamped to [0, 100]
  float accuracy = fmaxf(0.0f, fminf(100.0f, (1.0f - abs_error / 2.0f) * 100.0f));

  // CSV-style serial output for easy plotting
  Serial.print(angle, 4);
  Serial.print(",");
  Serial.print(actual, 4);
  Serial.print(",");
  Serial.print(prediction, 4);
  Serial.print(",");
  Serial.print(abs_error, 4);
  Serial.print(",");
  Serial.print(accuracy, 2);
  Serial.println("%");

  delay(kInferenceDelayMs);
}
