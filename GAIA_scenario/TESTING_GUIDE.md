# GAIA Testing Framework - Quick Start

## 🚀 Usage

### 1. Single Framework Test (environment variables)
```bash
# Test CrewAI with large model on 3 questions
FRAMEWORK=crewai MODEL_SIZE=large NUM_QUESTIONS=3 docker-compose run --rm gaia-agent

# Test with small model on 10 questions
FRAMEWORK=crewai MODEL_SIZE=small NUM_QUESTIONS=10 docker-compose run --rm gaia-agent
```

### 2. Use Iteration Script
```bash
# Rebuild first
docker-compose build

# Test all combinations (default 5 questions)
./run_test.sh

# Test with 20 questions per combination
./run_test.sh 20
```

### 3. Compare Frameworks
```bash
# Compare all frameworks on 10 questions with large model
./run_comparison.sh 10

# Compare with small model on 5 questions
./run_comparison.sh 5 small
```

### 4. Custom Combinations
```bash
# Test specific scenarios
for model in large small; do
    FRAMEWORK=crewai MODEL_SIZE=$model NUM_QUESTIONS=10 docker-compose run --rm gaia-agent
done
```

## 📁 Structure
```
GAIA_scenario/
├── gaia_tester.py              # Main test runner (env-driven)
├── agents/
│   ├── __init__.py
│   └── crewai_agent.py         # CrewAI implementation
├── run_test.sh                 # Iterate through combinations
├── run_comparison.sh           # Compare all frameworks
└── output/                     # Results (JSON files)
```

## 📊 Results

Results saved in `output/` as:
- `{framework}_{model}_{timestamp}.json` - Single test results
- `comparison_{timestamp}.json` - Framework comparison

Example result:
```json
{
  "agent_info": {
    "name": "CrewAI-gpt-4",
    "framework": "crewai",
    "model": "gpt-4"
  },
  "summary": {
    "total_questions": 5,
    "successful_runs": 5,
    "avg_execution_time": 12.3,
    "total_time": 61.5
  },
  "questions": [...]
}
```

## 🔧 Adding New Frameworks

### 1. Create agent file: `agents/your_framework_agent.py`
```python
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import time

@dataclass
class AgentResponse:
    answer: str
    execution_time: float
    tools_used: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class YourFrameworkAgent:
    def __init__(self, model_config: Dict[str, Any], verbose: bool = False):
        self.model_config = model_config
        # Initialize your framework
    
    @property
    def name(self) -> str:
        return f"YourFramework-{self.model_config['model']}"
    
    def run(self, question: str, file_paths: Optional[List[str]] = None) -> AgentResponse:
        start = time.time()
        # Your implementation
        answer = "..."
        return AgentResponse(
            answer=answer,
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

### 3. Update `run_test.sh`:
```bash
FRAMEWORKS=("crewai" "yourframework" "langgraph")
```

### 4. Test it:
```bash
docker-compose build
FRAMEWORK=yourframework NUM_QUESTIONS=3 docker-compose run --rm gaia-agent
```

## 🎯 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAMEWORK` | `crewai` | Agent framework to use |
| `MODEL_SIZE` | `large` | `large` (idx 0) or `small` (idx 1) from llmforall.py |
| `NUM_QUESTIONS` | `5` | Number of GAIA questions to test |
| `START_IDX` | `0` | Starting index in dataset |
| `TEST_MODE` | `single` | `single` or `compare` |
| `OUTPUT_DIR` | `/app/output` | Results directory |

## ✅ Testing Workflow

1. **Build container:**
   ```bash
   docker-compose build
   ```

2. **Quick test (3 questions):**
   ```bash
   FRAMEWORK=crewai NUM_QUESTIONS=3 docker-compose run --rm gaia-agent
   ```

3. **Iterate all combinations:**
   ```bash
   ./run_test.sh 10
   ```

4. **Compare frameworks:**
   ```bash
   ./run_comparison.sh 20
   ```

5. **View results:**
   ```bash
   ls -lh output/
   cat output/comparison_*.json | jq '.[] | {framework: .agent_info.framework, success: .summary.successful_runs}'
   ```
