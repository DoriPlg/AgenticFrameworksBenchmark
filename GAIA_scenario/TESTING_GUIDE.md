# GAIA Testing Framework - Testing Guide

## 🚀 Quick Start

### 1. Single Framework Test
```bash
# Test CrewAI on 3 questions
FRAMEWORK=crewai NUM_QUESTIONS=3 docker compose run --rm gaia-agent

# Test LangGraph with specific model
FRAMEWORK=langgraph MODEL_IDX=1 NUM_QUESTIONS=10 docker compose run --rm gaia-agent

# Test specific question range
FRAMEWORK=crewai START_IDX=10 NUM_QUESTIONS=5 docker compose run --rm gaia-agent
```

### 2. Compare All Frameworks
```bash
# Compare on same 10 questions
TEST_MODE=compare NUM_QUESTIONS=10 docker compose run --rm gaia-agent

# Compare with specific model
TEST_MODE=compare MODEL_IDX=0 NUM_QUESTIONS=20 docker compose run --rm gaia-agent
```

### 3. Use Helper Scripts
```bash
# Rebuild container first
docker compose build

# Test all combinations (5 questions default)
./run_test.sh

# Test with 20 questions per combination
./run_test.sh 20

# Compare all frameworks on 15 questions
./run_comparison.sh 15
```

### 4. Custom Iterations
```bash
# Test multiple models
for model_idx in 0 1 2; do
    FRAMEWORK=crewai MODEL_IDX=$model_idx NUM_QUESTIONS=5 docker compose run --rm gaia-agent
done

# Test all frameworks
for fw in crewai langgraph; do
    FRAMEWORK=$fw NUM_QUESTIONS=10 docker compose run --rm gaia-agent
done
```

## 📁 Architecture

```
GAIA_scenario/
├── gaia_tester.py              # Main test runner
├── agents/
│   ├── base_agent.py           # Abstract base class
│   ├── crewai_agent.py         # CrewAI implementation
│   └── langgraph_agent.py      # LangGraph implementation
├── shared_tools.py             # Common tool implementations
├── run_test.sh                 # Iteration script
├── run_comparison.sh           # Framework comparison script
└── output/                     # Results (JSON files)
```

### Agent Registry Pattern
Frameworks are registered in `AGENT_REGISTRY` dictionary:
```python
AGENT_REGISTRY: Dict[str, type[BaseAgent]] = {
    'crewai': CrewAIAgent,
    'langgraph': LangGraphAgent,
}
```

## 📊 Results Format

### Single Test Result
`output/{framework}_{model}_{timestamp}.json`:
```json
{
  "agent_info": {
    "name": "CrewAI-Meta-Llama-3.3-70B-Instruct",
    "framework": "crewai",
    "model": "Meta-Llama-3.3-70B-Instruct"
  },
  "test_config": {
    "num_questions": 5,
    "start_idx": 0,
    "timestamp": "2025-12-28T15:30:00"
  },
  "summary": {
    "total_questions": 5,
    "successful_runs": 5,
    "failed_runs": 0,
    "avg_execution_time": 12.3,
    "total_time": 61.5
  },
  "questions": [
    {
      "idx": 0,
      "question": "...",
      "correct_answer": "...",
      "agent_answer": "...",
      "execution_time": 10.5,
      "success": true
    }
  ]
}
```

### Comparison Result
`output/comparison_{timestamp}.json`:
```json
{
  "crewai": { /* same as single result */ },
  "langgraph": { /* same as single result */ }
}
```

## 🔧 Adding New Frameworks

### 1. Create Agent Class (inherit from BaseAgent)
`agents/your_framework_agent.py`:
```python
from agents.base_agent import BaseAgent, AgentResponse
import time
from typing import Dict, Any, Optional, List

class YourFrameworkAgent(BaseAgent):
    """Your framework implementation."""
    
    def __init__(self, model_config: Dict[str, Any], verbose: bool = False):
        super().__init__(model_config, verbose)
        # Initialize your framework with model_config
        self.agent = YourFramework(
            model=model_config['model'],
            api_key=model_config['api_key'],
            base_url=model_config['base_url']
        )
    
    @property
    def name(self) -> str:
        """Required: return agent name."""
        return f"YourFramework-{self.model_config['model']}"
    
    def run(self, question: str, file_paths: Optional[List[str]] = None) -> AgentResponse:
        """Required: execute agent and return response."""
        start = time.time()
        
        # Your implementation here
        result = self.agent.execute(question)
        
        return AgentResponse(
            answer=result,
            execution_time=time.time() - start,
            metadata={"framework": "yourframework"}
        )
```

### 2. Register in `gaia_tester.py`
```python
try:
    from agents.your_framework_agent import YourFrameworkAgent
    AGENT_REGISTRY['yourframework'] = YourFrameworkAgent
except ImportError as e:
    print(f"Warning: YourFramework not available - {e}")
```

### 3. Add Dependencies to `requirements.txt`
```bash
your-framework>=1.0.0
```

### 4. Rebuild and Test
```bash
docker compose build
FRAMEWORK=yourframework NUM_QUESTIONS=3 docker compose run --rm gaia-agent
```

## 🎯 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAMEWORK` | `crewai` | Framework to test: `crewai`, `langgraph` |
| `MODEL_IDX` | `0` | Model index from `llmforall.py` (0=first, 1=second, etc.) |
| `NUM_QUESTIONS` | `5` | Number of GAIA questions to test |
| `START_IDX` | `0` | Starting index in dataset (0-92 for test set) |
| `TEST_MODE` | `single` | `single` or `compare` (all frameworks) |
| `OUTPUT_DIR` | `/app/output` | Results directory (inside container) |

## ✅ Typical Workflow

### Initial Setup
```bash
# 1. Add API keys to .env
cp .env.example .env
vim .env  # Add HF_TOKEN, OPENAI_API_KEY, etc.

# 2. Build container (first time or after code changes)
docker compose build
```

### Development Testing
```bash
# Quick validation (3 questions)
FRAMEWORK=crewai NUM_QUESTIONS=3 docker compose run --rm gaia-agent

# Test specific question that's failing
START_IDX=42 NUM_QUESTIONS=1 FRAMEWORK=langgraph docker compose run --rm gaia-agent
```

### Benchmark Runs
```bash
# Single framework, full test
FRAMEWORK=crewai NUM_QUESTIONS=93 docker compose run --rm gaia-agent

# Compare all frameworks on subset
TEST_MODE=compare NUM_QUESTIONS=20 docker compose run --rm gaia-agent

# Automated comparison across models
./run_comparison.sh 50
```

### View Results
```bash
# List results
ls -lht output/ | head

# Summary of single test
cat output/crewai_*.json | jq '.summary'

# Compare frameworks
cat output/comparison_*.json | jq 'to_entries | map({framework: .key, success: .value.summary.successful_runs, avg_time: .value.summary.avg_execution_time})'

# Extract specific question
cat output/crewai_*.json | jq '.questions[] | select(.idx == 0)'
```

## 🐛 Debugging

### Check if framework is registered
```bash
docker compose run --rm gaia-agent python3 -c "from gaia_tester import AGENT_REGISTRY; print(list(AGENT_REGISTRY.keys()))"
```

### Enter container for debugging
```bash
docker compose run --rm --entrypoint bash gaia-agent
# Inside container:
python3 -c "from agents.crewai_agent import CrewAIAgent; print('OK')"
```

### Check GAIA dataset
```bash
docker compose run --rm gaia-agent python3 -c "from datasets import load_dataset; print(len(load_dataset('...', split='test')))"
```
