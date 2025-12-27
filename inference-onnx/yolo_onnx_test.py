import onnxruntime as ort
import numpy as np
import cv2

# Load the model
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# Load a sample image
img_path = "images/image1.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img_resized = cv2.resize(img_rgb, (640, 640))
input_tensor = np.expand_dims(np.transpose(img_resized, (2, 0, 1)), axis=0)

# Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})

print("ONNX inference ran successfully!")
