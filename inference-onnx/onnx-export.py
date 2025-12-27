from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Export to ONNX
model.export(format="onnx",
             imgsz=640, 
             dynamic=False,
             nms=True,
             conf=0.25,
             opset=12
             )