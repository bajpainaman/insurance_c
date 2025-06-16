# Multi-stage Docker build for Fraud Detection System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py .
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Copy data file if it exists (will be mounted in production)
COPY data.xlsx ./data/ 2>/dev/null || echo "Data file will be mounted at runtime"

# Set permissions
RUN chmod +x main.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; import sklearn; import lightgbm; import xgboost; print('All dependencies loaded successfully')" || exit 1

# Default command
CMD ["python", "main.py"]

# Production stage with minimal dependencies
FROM base as production

# Copy only necessary files
COPY --from=base /app /app
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN adduser --disabled-password --gecos '' fraud-detector && \
    chown -R fraud-detector:fraud-detector /app

USER fraud-detector

# Final command
CMD ["python", "main.py"]