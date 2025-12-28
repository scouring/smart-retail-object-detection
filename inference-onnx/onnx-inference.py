# infer_onnx.py
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

print("onnx-inference.py started")

MODEL_PATH = "best.onnx"
IMAGE_DIR = Path("images/")
CONF_THRESHOLD = 0.25
IMGSZ = 640
DISPLAY_SCALE = 0.6

# Maximum number of detections (should match ONNX  export)
MAX_DETECTIONS = 300

# Define colors for visualization
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

def load_model(model_path=MODEL_PATH):
    """Load the ONNX model with CPU execution."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


def preprocess(image_path):
    """Read and resize image for YOLO input."""
    orig_image = cv2.imread(str(image_path))
    img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # Resize directly to static shape expected by ONNX
    img_resized = cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)

    img_resized = img_resized.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_resized, (2, 0, 1))[None, ...]  # HWC -> NCHW

    return img_tensor, orig_image

def postprocess(outputs, orig_image, conf_threshold=CONF_THRESHOLD):
    """
    Apply NMS and draw bounding boxes from ONNX output.
    Assumes output shape (1, N, 6) -> [x1, y1, x2, y2, conf, class_id]
    """
    annotated = orig_image.copy()
    output = outputs[0][0]  # shape: (N, 6)

    # Filter by confidence
    mask = output[:, 4] >= conf_threshold
    output = output[mask]

    h_orig, w_orig = orig_image.shape[:2]

    for det in output:
        x1, y1, x2, y2, conf, cls_id = det
        # Scale back to original image size
        x1 = int(x1 * w_orig / IMGSZ)
        x2 = int(x2 * w_orig / IMGSZ)
        y1 = int(y1 * h_orig / IMGSZ)
        y2 = int(y2 * h_orig / IMGSZ)

        color = COLORS[int(cls_id) % len(COLORS)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{int(cls_id)}:{conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return annotated

def infer(session, image_path):
    input_tensor, orig_image = preprocess(image_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    annotated_image = postprocess(outputs, orig_image)
    return annotated_image

def main():
    session = load_model()
    for img_file in IMAGE_DIR.glob("*.*"):
        annotated = infer(session, img_file)
        h, w = annotated.shape[:2]
        display = cv2.resize(
            annotated,
            (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)),
            interpolation=cv2.INTER_AREA
        )
        # cv2.imshow("YOLO Inference", display)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Modifications for Docker and EC2 due to GUI/Qt errors
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(output_dir / img_file.name, annotated)
        


if __name__ == "__main__":
    main()
