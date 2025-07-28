# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . /app

# Install system dependencies for PDF parsing
RUN apt-get update && \
    apt-get install -y build-essential poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run the main script
CMD ["python", "main.py"]
