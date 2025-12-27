from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt") 

# Run inference on CPU
results = model.predict(
    source = "images/",
    device='cpu',
    conf=0.25,
    imgsz=640
)

# Visualize the result
for result in results:
    annotated_frame = result.plot()
    screen_scale = 0.6
    h, w = annotated_frame.shape[:2]
    display = cv2.resize(
        annotated_frame,
        (int(w * screen_scale), int(h * screen_scale)),
        interpolation=cv2.INTER_AREA
    )
    # cv2.namedWindow("YOLO Inference", cv2.WINDOW_AUTOSIZE)
    # cv2.resizeWindow("YOLO Inference", w, h)
    cv2.imshow("YOLO Inference", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()