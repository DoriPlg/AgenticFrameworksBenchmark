# GAIA Agent - Docker Setup

This directory contains a **completely isolated** containerized GAIA benchmark agent with secure code execution.

## 🚀 Quick Start

### Set up environment:
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN and other API keys
```

### Build and run (completely isolated):
```bash
docker-compose build
docker-compose up
```

**First run:** The container will automatically download the GAIA dataset (isolated within container). This takes a few minutes.

**Subsequent runs:** Dataset is cached in container volume, starts immediately.

### Run directly:
```bash
docker run --rm \
  -e OPENAI_API_KEY="your-key" \
  -e HF_TOKEN="your-hf-token" \
  -v $(pwd)/output:/app/output \
  gaia-agent
```

## 📁 Directory Structure

```
GAIA_scenario/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration config
├── entrypoint.sh          # Startup script (downloads GAIA data)
├── requirements.txt        # Python dependencies
├── crewai_gaia.py         # Main agent code
├── shared_tools.py        # Tool implementations
├── data/
│   └── data_pull.py       # GAIA dataset downloader
└── output/                # Results output (only mounted volume)
```

## 🔒 Security Features

- **Complete isolation**: All data (including GAIA dataset) stays inside container
- **Non-root user**: Runs as `gaiauser` (UID 1000)
- **Resource limits**: CPU and memory capped
- **Network isolation**: Bridge network for controlled access
- **No privilege escalation**: `no-new-privileges` security option
- **No host file access**: Dataset downloaded and cached internally

## 📝 Environment Variables

Create a `.env` file with:
```bash
OPENAI_API_KEY=your_key_here
LANGFUSE_SECRET_KEY=your_key_here
LANGFUSE_PUBLIC_KEY=your_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
HF_TOKEN=your_huggingface_token_here
```

### Getting Your HuggingFace Token:
1. Go to https://huggingface.co/datasets/gaia-benchmark/GAIA
2. Click "Request Access" and wait for approval
3. Get your token at https://huggingface.co/settings/tokens
4. Add it to your `.env` file

## 📥 Downloading GAIA Dataset

**Automatic:** The container automatically downloads GAIA dataset on first startup.
- Dataset is completely isolated within the container
- Cached in container volume for subsequent runs
- No data touches your host filesystem

**Manual download (if needed):**
```bash
# Enter running container
docker-compose exec gaia-agent bash
# Run download script
python3 data/data_pull.py
```

## 🧪 Testing

The container is completely isolated:
- GAIA dataset downloads automatically on startup
- All test files and data stay within the container
- Only results are written to `./output` (optional mount)

```bash
# First run - downloads dataset
docker-compose up

# Subsequent runs - uses cached dataset
docker-compose up
```

## 🛠️ Development

To modify and test:
```bash
# Rebuild after changes
docker-compose build

# Run with live logs
docker-compose up --build

# Shell access for debugging
docker-compose exec gaia-agent bash
```

## 📊 Monitoring

Logs are streamed to stdout. To save logs:
```bash
docker-compose up > logs/gaia_$(date +%Y%m%d_%H%M%S).log 2>&1
```

## ⚠️ Important Notes

1. **Complete Isolation**: All code execution and data stays within the container
2. **First Run**: Takes a few minutes to download GAIA dataset (15-20GB)
3. **Dataset Cache**: Persists in `crewai-storage` Docker volume between runs
4. **API Keys**: HF_TOKEN is required - get access at https://huggingface.co/datasets/gaia-benchmark/GAIA
5. **Resource Usage**: Adjust CPU/memory limits based on question complexity
6. **No Host Access**: Container cannot access files on your computer (by design)
