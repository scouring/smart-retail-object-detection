from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import io
from app.onnx_inference import load_model, infer

app = FastAPI()
session = load_model()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    annotated = infer(session, img)

    # Encode annotated image as JPEG for response
    _, img_encoded = cv2.imencode('.jpg', annotated)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

