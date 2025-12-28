# Dockerfile for GAIA Agent with Code Execution Isolation
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (cached unless system packages change)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first (cached unless requirements change)
COPY requirements.txt .

# Install Python dependencies (this layer is cached!)
RUN pip install --no-cache-dir -r requirements.txt

# Create user and directories BEFORE copying app files
RUN useradd -m -u 1000 gaiauser && \
    mkdir -p /app/data /app/hf_cache /app/crewai_storage /app/output /app/agents && \
    chown -R gaiauser:gaiauser /app

# NOW copy application files (changes here won't invalidate pip install)
COPY --chown=gaiauser:gaiauser gaia_tester.py .
COPY --chown=gaiauser:gaiauser agents/ ./agents/
COPY --chown=gaiauser:gaiauser data/data_pull.py ./data/
COPY --chown=gaiauser:gaiauser ../llmforall.py ./
COPY --chown=gaiauser:gaiauser entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER gaiauser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/hf_cache
ENV CREWAI_STORAGE_DIR=/app/crewai_storage
# Disable LiteLLM logging to avoid clutter
ENV LITELLM_LOG=ERROR
ENV LITELLM_DISABLE_LOGGING=true

# Expose port if needed for monitoring/debugging
EXPOSE 8000

# Use entrypoint to download data first, then run app
ENTRYPOINT ["/app/entrypoint.sh"]
