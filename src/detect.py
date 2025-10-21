from ultralytics import YOLO
from PIL import Image

model = YOLO('smart_retail_yolov8.pt')

def detect_image(image_path):
    results = model(image_path)
    annotated_img = results[0].plot()
    return Image.fromarray(annotated_img)
