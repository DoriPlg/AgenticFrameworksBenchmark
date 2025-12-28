# GAIA Agent - Docker Setup

This directory contains a containerized GAIA benchmark agent with secure code execution isolation.

## 🚀 Quick Start

### Build the container:
```bash
docker build -t gaia-agent .
```

### Run with docker-compose:
```bash
# Set environment variables in .env file first
docker-compose up
```

### Run directly:
```bash
docker run --rm \
  -e OPENAI_API_KEY="your-key" \
  -e LANGFUSE_SECRET_KEY="your-key" \
  -e LANGFUSE_PUBLIC_KEY="your-key" \
  -e LANGFUSE_HOST="your-host" \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/output:/app/output \
  gaia-agent
```

## 📁 Directory Structure

```
GAIA_scenario/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration config
├── requirements.txt        # Python dependencies
├── crewai_gaia.py         # Main agent code
├── shared_tools.py        # Tool implementations
├── data/                  # Mount GAIA test files here
└── output/                # Results output directory
```

## 🔒 Security Features

- **Non-root user**: Runs as `gaiauser` (UID 1000)
- **Resource limits**: CPU and memory capped
- **Read-only filesystem**: Except for `/tmp` and `/app/output`
- **Network isolation**: Bridge network for controlled access
- **No privilege escalation**: `no-new-privileges` security option

## 📝 Environment Variables

Create a `.env` file with:
```bash
OPENAI_API_KEY=your_key_here
LANGFUSE_SECRET_KEY=your_key_here
LANGFUSE_PUBLIC_KEY=your_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
```

## 🧪 Testing

Place GAIA test files in the `data/` directory:
```bash
mkdir -p data
# Copy your GAIA CSV, PDF, Excel files here
```

## 🛠️ Development

To modify and test locally:
```bash
# Rebuild after changes
docker-compose build

# Run with live logs
docker-compose up --build

# Shell access for debugging
docker run -it --entrypoint /bin/bash gaia-agent
```

## 📊 Monitoring

Logs are streamed to stdout. To save logs:
```bash
docker-compose up > logs/gaia_$(date +%Y%m%d_%H%M%S).log 2>&1
```

## ⚠️ Important Notes

1. **Code Execution**: The agent executes arbitrary Python code. Always run in an isolated container.
2. **File Access**: GAIA test files must be mounted to `/app/data`
3. **API Keys**: Never commit `.env` file with real keys
4. **Resource Usage**: Adjust CPU/memory limits based on your questions' complexity
