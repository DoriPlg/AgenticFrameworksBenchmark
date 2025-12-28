# Dockerfile for GAIA Agent with Code Execution Isolation
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY shared_tools.py .
COPY crewai_gaia.py .
COPY data/data_pull.py ./data/

# Copy parent directory file needed for execution
COPY ../llmforall.py ./

# Create directory for data files (GAIA test files)
RUN mkdir -p /app/data

# Create a non-root user for security
RUN useradd -m -u 1000 gaiauser && \
    chown -R gaiauser:gaiauser /app

# Create necessary directories for CrewAI and HuggingFace storage
RUN mkdir -p /home/gaiauser/.local/share/crewai && \
    mkdir -p /home/gaiauser/.cache/huggingface && \
    mkdir -p /home/gaiauser/.cache && \
    mkdir -p /app/output && \
    mkdir -p /tmp/crewai && \
    chown -R gaiauser:gaiauser /home/gaiauser/.local && \
    chown -R gaiauser:gaiauser /home/gaiauser/.cache && \
    chown -R gaiauser:gaiauser /app/output

# Switch to non-root user
USER gaiauser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/home/gaiauser/.cache/huggingface

# Expose port if needed for monitoring/debugging
EXPOSE 8000

# Default command
CMD ["python3", "crewai_gaia.py"]
