#!/bin/bash

# Fraud Detection System Runner Script
set -e

echo "🚀 Starting Fraud Detection System..."

# Check if data file exists
if [ ! -f "/app/data/data.xlsx" ]; then
    echo "❌ Error: data.xlsx not found in /app/data/"
    echo "Please mount your data file using:"
    echo "docker run -v /path/to/your/data.xlsx:/app/data/data.xlsx fraud-detection"
    exit 1
fi

# Create directories if they don't exist
mkdir -p /app/models /app/logs

echo "📊 Data file found: $(ls -lh /app/data/data.xlsx)"
echo "🔧 Starting training pipeline..."

# Run the main application
python /app/main.py

echo "✅ Training completed successfully!"
echo "📁 Models saved to: /app/models/"
echo "📝 Logs available in: /app/logs/"