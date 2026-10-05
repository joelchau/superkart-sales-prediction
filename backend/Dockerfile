# Lightweight Python 3.13 image – same Python version as the Colab session used for training
FROM python:3.13-slim

# Working directory inside the container
WORKDIR /app

# Install the dependencies first (this layer is cached when only the code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the serialized model
COPY . .

# The API listens on port 7860
EXPOSE 7860

# Start the Flask app with Gunicorn: 4 worker processes, listening on all interfaces, port 7860
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:superkart_api"]
