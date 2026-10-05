# Lightweight Python 3.13 image
FROM python:3.13-slim

# Working directory inside the container
WORKDIR /app

# Install the dependencies first (this layer is cached when only the code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app
COPY . .

# Streamlit listens on port 8501
EXPOSE 8501

# Run the app on port 8501, reachable from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
