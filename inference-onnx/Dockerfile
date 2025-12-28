FROM python:3.11-slim

# Prevent Python from buffering stdout (so print() shows up)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# OS dependencies required by OpenCV (no GUI)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, model, and data
COPY onnx-inference.py .
COPY best.onnx .
COPY images ./images

# Create output directory explicitly (optional but clean)
RUN mkdir -p outputs

# Run inference
CMD ["python", "onnx-inference.py"]
