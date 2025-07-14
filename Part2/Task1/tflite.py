
# Smart Walking Assistive Device: Edge AI Demo
# This script demonstrates how a lightweight AI model can be used on an edge device
# to detect obstacles or hazards for a walking assistive device.

import tensorflow as tf
import numpy as np


# 1. Load and convert the model to TensorFlow Lite
#    In a real device, you would train a custom model to detect obstacles/terrain.
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to disk
with open("mobilenet_v2.tflite", "wb") as f:
    f.write(tflite_model)

# Load the TFLite model for inference
interpreter = tf.lite.Interpreter(model_path="mobilenet_v2.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# 2. Capture real input data from the webcam (requires OpenCV)
import cv2

# Open the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Press SPACE to capture an image for inference...")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    # Show the live camera feed
    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1)
    if key == 32:  # SPACE key
        break


# Resize and preprocess the captured frame for TFLite model
image = cv2.resize(frame, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
image = np.expand_dims(image, axis=0).astype(np.float32)

cap.release()
cv2.destroyAllWindows()


# 3. Perform inference using the TFLite model
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])

# 4. Decode predictions to get human-readable labels
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

# 5. Output the predictions and measure distance (simulated for now)
print("Predictions:", decoded_predictions)

# --- Distance Measurement Placeholder ---
# Replace this function with real sensor code (e.g., HC-SR04, LiDAR)
def get_distance():
    # Simulate a distance in meters (e.g., from 0.3 to 2.5 meters)
    # In real use, read from the hardware sensor
    return 0.8  # Example: 0.8 meters to nearest object

distance = get_distance()

# --- Decision and Guidance Logic ---
# Use the top-1 prediction for guidance
object_label = decoded_predictions[0][0][1] if decoded_predictions and decoded_predictions[0] else "object"

if distance < 1.0:
    print(f"Warning: {object_label} detected {distance:.2f} meters ahead. Please stop or turn.")
else:
    print("Path is clear. You may proceed.")


# ---
# Explanation:
# Step 1: Loads MobileNetV2 and converts it to TensorFlow Lite for efficient edge inference.
# Step 2: Captures an image from the webcam and preprocesses it for the TFLite model.
# Step 3: Runs inference using the TFLite interpreter to get predictions.
# Step 4: Decodes the predictions to human-readable labels.
# Step 5: Simulates distance measurement and combines it with object detection to provide real-time guidance (e.g., warning if an object is close, or "Path is clear").