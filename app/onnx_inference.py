import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "best.onnx"
IMGSZ = 640
CONF_THRESHOLD = 0.25

# Colors for classes
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

def load_model(model_path=MODEL_PATH):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session
    
def preprocess_image(img):
    """Resize and normalize image for ONNX."""
    img_resized = cv2.resize(img, (640, 640))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.transpose(img_resized, (2, 0, 1))
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized


def postprocess(outputs, orig_w, orig_h, orig_image):
    """
    Convert ONNX outputs to annotated image.
    Assumes output shape: (1, N, 6) -> [x1, y1, x2, y2, conf, cls_id]
    """
    annotated = orig_image.copy() # draw on the original image
    output = outputs[0][0]  # shape: (N, 6)
    mask = output[:, 4] >= CONF_THRESHOLD
    output = output[mask]

    for det in output:
        x1, y1, x2, y2, conf, cls_id = det
        x1 = int(x1 * orig_w / IMGSZ)
        x2 = int(x2 * orig_w / IMGSZ)
        y1 = int(y1 * orig_h / IMGSZ)
        y2 = int(y2 * orig_h / IMGSZ)
        color = COLORS[int(cls_id) % len(COLORS)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{int(cls_id)}:{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated

def infer(session, img):
    input_tensor = preprocess_image(img)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    orig_h, orig_w = img.shape[:2]  # get original dimensions
    annotated = postprocess(outputs, orig_w, orig_h, img)
    return annotated

