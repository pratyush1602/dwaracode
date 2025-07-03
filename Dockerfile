# Use Python 3.11 as the base image
FROM python:3.11-buster

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8001

# Expose the port
EXPOSE 8000

# Create a script to run the application
RUN echo '#!/bin/bash\n\
python -m uvicorn main:app --host 127.0.0.1 --port $PORT' > /app/start.sh && \
    chmod +x /app/start.sh


# Command to run the application
CMD ["/app/start.sh"] 