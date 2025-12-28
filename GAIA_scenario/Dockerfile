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
RUN useradd -m -u 1000 gaiauser

# Create necessary directories in /app (already owned by root, will chown below)
RUN mkdir -p /app/hf_cache && \
    mkdir -p /app/crewai_storage && \
    mkdir -p /app/output && \
    chown -R gaiauser:gaiauser /app

# Switch to non-root user
USER gaiauser

# Copy entrypoint script
COPY --chown=gaiauser:gaiauser entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/hf_cache
ENV CREWAI_STORAGE_DIR=/app/crewai_storage

# Expose port if needed for monitoring/debugging
EXPOSE 8000

# Use entrypoint to download data first, then run app
ENTRYPOINT ["/app/entrypoint.sh"]
