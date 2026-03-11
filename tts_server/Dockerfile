# Use a smaller CUDA-enabled Python image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    qwen-tts>=0.1.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    python-multipart>=0.0.6 \
    soundfile>=0.12.0

# Copy application code
COPY app.py .

# Create directory for voice samples
RUN mkdir -p /tmp/voices

# Expose port
EXPOSE 8000

# Run the service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
