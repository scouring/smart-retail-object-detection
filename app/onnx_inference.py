import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "best.onnx"  # relative to this file
CONF_THRESHOLD = 0.25
IMGSZ = 640
DISPLAY_SCALE = 0.6
MAX_DETECTIONS = 300
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

def load_model(model_path=MODEL_PATH):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session

def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMGSZ, IMGSZ))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_resized, (2, 0, 1))[None, ...]
    return img_tensor

def postprocess(outputs, orig_image):
    annotated = orig_image.copy()
    output = outputs[0][0]
    mask = output[:, 4] >= CONF_THRESHOLD
    output = output[mask]
    h, w = orig_image.shape[:2]
    for det in output:
        x1, y1, x2, y2, conf, cls_id = det
        x1 = int(x1 * w / IMGSZ)
        x2 = int(x2 * w / IMGSZ)
        y1 = int(y1 * h / IMGSZ)
        y2 = int(y2 * h / IMGSZ)
        color = COLORS[int(cls_id) % len(COLORS)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{int(cls_id)}:{conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated

def infer(session, image):
    tensor = preprocess(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    annotated = postprocess(outputs, image)
    return annotated
