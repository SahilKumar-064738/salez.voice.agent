# Base image
FROM python:3.10-slim

# System dependencies (VERY IMPORTANT for vosk + audio)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]