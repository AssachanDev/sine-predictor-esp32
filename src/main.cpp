#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino.h>

const unsigned char *model_data = sine_model_tflite;
const size_t model_size = sine_model_tflite_len;

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
tflite::AllOpsResolver resolver;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];

float randomFloat(float min, float max) {
  return min + (max - min) * (float)random(0, 10000) / 10000.0;
}

void setup() {
  Serial.begin(115200);

  model = tflite::GetModel(model_data);

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();
}

void loop() {
  float angle = randomFloat(-3.1416, 3.1416);

  TfLiteTensor *input = interpreter->input(0);
  input->data.f[0] = angle;

  if (interpreter->Invoke() != kTfLiteOk) {
    return;
  }

  TfLiteTensor *output = interpreter->output(0);
  float prediction = output->data.f[0];

  float actual_sine = sin(angle);
  float error = abs(actual_sine - prediction);
  float accuracy = (1.0f - error) * 100.0f;
  if (accuracy < 0)
    accuracy = 0;

  Serial.print("Angle:");
  Serial.print(angle, 2);
  Serial.print(",Actual:");
  Serial.print(actual_sine, 2);
  Serial.print(",AI:");
  Serial.print(prediction, 2);
  Serial.print(",Err:");
  Serial.print(error, 4);
  Serial.print(",Acc:");
  Serial.print(accuracy, 2);
  Serial.println("%");

  delay(1000);
}
