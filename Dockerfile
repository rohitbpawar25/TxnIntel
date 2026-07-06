# Use Python 3.9 slim base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (build-essential for scikit-learn / shap C extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model binaries and src code
COPY models/ /app/models/
COPY src/ /app/src/

# Expose API port
EXPOSE 8000

# Set Python Path
ENV PYTHONPATH=/app

# Command to launch the API service
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
