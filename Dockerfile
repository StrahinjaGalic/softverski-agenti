FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs /app/charts

# Set Python path
ENV PYTHONPATH=/app/src:/app

# Expose all potential ports
EXPOSE 8000 8001 8002 8010 8011 8012 8013 8014

# Default command (will be overridden by docker-compose)
CMD ["python", "-c", "print('Docker container ready. Use docker-compose to run services.')"]