# GAIA Benchmark Testing Framework

A **modular, containerized framework** for testing multiple agentic frameworks on the GAIA benchmark with complete isolation and security.

## 🎯 Features

- **Modular Architecture**: Easy to add new agent frameworks
- **Abstract Base Class**: Enforced interface for all agents
- **Environment-Driven Testing**: Configure via environment variables
- **Complete Isolation**: All execution and data stays in container
- **Framework Comparison**: Test multiple frameworks on same questions
- **Automatic Caching**: GAIA dataset cached between runs

## 🚀 Quick Start

### 1. Set up environment:
```bash
cp .env.example .env
# Add your HF_TOKEN, OPENAI_API_KEY, etc.
```

### 2. Build container:
```bash
docker compose build
```

### 3. Run tests:
```bash
# Test CrewAI on 3 questions
FRAMEWORK=crewai NUM_QUESTIONS=3 docker compose run --rm gaia-agent

# Test LangGraph on 5 questions
FRAMEWORK=langgraph NUM_QUESTIONS=5 docker compose run --rm gaia-agent

# Compare all frameworks on 10 questions
TEST_MODE=compare NUM_QUESTIONS=10 docker compose run --rm gaia-agent
```

**First run:** Downloads GAIA dataset automatically (~2-5 minutes). Subsequent runs use cached data.

## 📁 Directory Structure

```
GAIA_scenario/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration config
├── entrypoint.sh          # Startup script
├── requirements.txt        # Python dependencies
├── gaia_tester.py         # Main test runner (modular)
├── agents/
│   ├── __init__.py        # Package initialization
│   ├── base_agent.py      # Abstract base class
│   ├── crewai_agent.py    # CrewAI implementation
│   ├── langgraph_agent.py # LangGraph implementation
│   ├── openai_agent.py    # OpenAI implementation
│   └── tools/
│       └── shared_tools.py # Shared tool implementations
├── data/
│   └── data_pull.py       # GAIA dataset downloader
└── output/                # Results (JSON files)
```

## 🔒 Security Features

- **Complete Isolation**: All data and execution isolated in container
- **Non-Root Execution**: Runs as `gaiauser` (UID 1000)
- **Resource Limits**: CPU (2 cores) and memory (4GB) capped
- **No Privilege Escalation**: Security options enforced
- **Code Execution Sandbox**: Python interpreter runs in container only

## 📝 Environment Variables

### API Keys (required in `.env`):
```bash
OPENAI_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token_here

# Optional: Tracing
LANGFUSE_SECRET_KEY=your_key_here
LANGFUSE_PUBLIC_KEY=your_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Test Configuration (pass at runtime):
| Variable | Default | Description |
|----------|---------|-------------|
| `FRAMEWORK` | `crewai` | Framework to test: `crewai`, `langgraph` |
| `MODEL_IDX` | `0` | Model index from `llmforall.py` |
| `NUM_QUESTIONS` | `5` | Number of GAIA questions to test |
| `START_IDX` | `0` | Starting index in dataset |
| `TEST_MODE` | `single` | `single` or `compare` (all frameworks) |
| `OUTPUT_DIR` | `/app/output` | Results directory |

### Getting HuggingFace Token:
1. Request access: https://huggingface.co/datasets/gaia-benchmark/GAIA
2. Get token: https://huggingface.co/settings/tokens
3. Add to `.env` file

## 🧪 Running Tests

### Single Framework:
```bash
# Quick test with 3 questions
FRAMEWORK=crewai NUM_QUESTIONS=3 docker compose run --rm gaia-agent

# Full test with specific model
FRAMEWORK=langgraph MODEL_IDX=1 NUM_QUESTIONS=20 docker compose run --rm gaia-agent
```

### Compare All Frameworks:
```bash
TEST_MODE=compare NUM_QUESTIONS=10 docker compose run --rm gaia-agent
```

### Use Helper Scripts:
```bash
# Test all combinations
./run_test.sh 5

# Compare frameworks
./run_comparison.sh 10
```

## 🛠️ Adding New Frameworks

### 1. Create agent class inheriting from `BaseAgent`:
```python
# agents/your_framework_agent.py
from agents.base_agent import BaseAgent, AgentResponse
import time

class YourFrameworkAgent(BaseAgent):
    def __init__(self, model_config, verbose=False):
        super().__init__(model_config, verbose)
        # Initialize your framework
    
    @property
    def name(self) -> str:
        return f"YourFramework-{self.model_config['model']}"
    
    def run(self, question, file_paths=None) -> AgentResponse:
        start = time.time()
        # Your implementation
        return AgentResponse(
            answer="...",
            execution_time=time.time() - start
        )
```

### 2. Register in `gaia_tester.py`:
```python
try:
    from agents.your_framework_agent import YourFrameworkAgent
    AGENT_REGISTRY['yourframework'] = YourFrameworkAgent
except ImportError as e:
    print(f"Warning: YourFramework not available - {e}")
```

### 3. Add dependencies to `requirements.txt` and rebuild:
```bash
docker compose build
FRAMEWORK=yourframework NUM_QUESTIONS=3 docker compose run --rm gaia-agent
```

## 📊 Results

Results saved as JSON in `output/`:
- `{framework}_{model}_{timestamp}.json` - Single test
- `comparison_{timestamp}.json` - Multi-framework comparison

```bash
# View results
ls -lh output/
cat output/comparison_*.json | jq '.crewai.summary'
```

## 🏗️ Architecture

### BaseAgent Abstract Class
All agents must inherit from `BaseAgent` and implement:
- `name` property: Returns agent identification string
- `run(question, file_paths)` method: Executes agent and returns `AgentResponse`

This ensures consistency and catches errors at import time.

### Shared Tools
All frameworks use the same tool implementations from `agents/tools/shared_tools.py`:
- `web_search()` - DuckDuckGo search
- `read_webpage()` - BeautifulSoup scraping
- `inspect_file()` - PDF/text file reading
- `python_interpreter()` - Code execution with numpy/pandas/json

## ⚠️ Important Notes

1. **First Run**: Downloads GAIA dataset (~2-5 minutes, cached afterwards)
2. **Complete Isolation**: All execution happens in container, no host file access
3. **Dataset Cache**: Persists in `gaia-storage` Docker volume
4. **HF_TOKEN Required**: Get access at https://huggingface.co/datasets/gaia-benchmark/GAIA
5. **LLM Endpoint**: Uses `host.docker.internal` to access localhost LLMs from container
6. **Resource Limits**: Adjust in `docker-compose.yml` if needed

## 📚 See Also

- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Detailed testing instructions
- [GAIA Benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) - Dataset information
