# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies that might be missing
RUN pip install --no-cache-dir pandas matplotlib gym

# Copy the entire application
COPY . .

# Expose port (Hugging Face Spaces uses PORT env var or defaults to 7860)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Run the application
CMD uvicorn Swarna:app --host=0.0.0.0 --port=${PORT:-7860}

