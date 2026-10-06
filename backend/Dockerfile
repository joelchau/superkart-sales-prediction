# Lightweight Python 3.13 image – same Python version as the Colab session used for training
FROM python:3.13-slim

# Working directory inside the container
WORKDIR /app

# Install the dependencies first (this layer is cached when only the code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code, the serialized pipeline and the model card
COPY . .

# The API listens on port 7860
EXPOSE 7860

# Docker marks the container unhealthy if /health stops answering
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start the Flask app with Gunicorn: 4 worker processes, listening on all interfaces, port 7860
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "--timeout", "120", "app:superkart_api"]
