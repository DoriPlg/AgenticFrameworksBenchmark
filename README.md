# Agentic Frameworks Benchmark

A comprehensive benchmarking suite for evaluating and comparing different AI agent frameworks (CrewAI, LangGraph, LangChain, OpenAI Agents) across multiple scenarios and difficulty levels.

## 📂 Project Structure

### **GAIA_scenario/**
The main testing framework for evaluating agents on the GAIA (General AI Assistants) benchmark - a challenging dataset designed to test real-world AI assistant capabilities.

**Key Components:**
- **gaia_agents/**: Modular agent implementations
  - `base_agent.py`: Abstract base class defining agent interface
  - `crewai_agent.py`, `langgraph_agent.py`, `langchain_agent.py`, `openai_agent.py`: Framework-specific implementations
  - `tools/`: Shared tools (web search, file inspection, Python interpreter, web scraping)
- **gaia_tester.py**: Main test runner supporting single framework or comparison mode
- **grade_pipeline.py**: Automated grading system for agent responses
- **grader.py**: Answer evaluation and scoring logic
- **llmforall.py**: Model configuration and API endpoints
- **plot_results.py**: Visualization tools for benchmark results
- **data/**: GAIA dataset management and caching
- **output/**: Test results organized by level (lvl1/lvl2/lvl3) and temperature
  - `graded/`: Scored comparison results
  - `summaries/`: Performance summaries and statistics
- **docker-compose.yml** + **Dockerfile**: Containerized testing environment for isolation and reproducibility

**Features:**
- Dockerized execution with complete isolation
- Automatic dataset caching
- Multi-framework comparison mode
- Configurable via environment variables (model, temperature, question count)
- Langfuse tracing integration for observability

### **vacation_scenario/**
A simpler demonstration scenario focused on vacation planning tasks, useful for rapid prototyping and testing framework-specific features.

**Files:**
- `crewai_vacation.py`: CrewAI implementation for vacation planning
- `langgraph_vacation.py`: LangGraph implementation
- `openai_vacation.py`: OpenAI Agents implementation
- `hybrid_vacation.py`: Mixed/hybrid approach demonstration
- `shared_tools.py`: Common tools for vacation research and planning

**Purpose:** 
Quick iteration and feature testing without the complexity of GAIA benchmark infrastructure.

## 🚀 Quick Start

### GAIA Benchmark
```bash
cd GAIA_scenario
cp .env.example .env  # Add API keys
docker compose build
TEST_MODE=compare NUM_QUESTIONS=10 MODEL_IDX=0 docker compose run --rm gaia-agent
```

### Vacation Scenario
```bash
cd vacation_scenario
cp .env.example .env  # Add API keys
python crewai_vacation.py  # or any other framework
```

## 📊 Output Files

- **Comparison JSONs**: Raw agent responses and execution metadata
- **Graded JSONs**: Scored results with correctness evaluation
- **Summaries**: Aggregated performance metrics per framework/model
- **Traces**: Langfuse trace exports for detailed analysis (JSON format)

## 🎯 Use Cases

- **Framework Evaluation**: Compare agent frameworks on standardized tasks
- **Model Benchmarking**: Test different LLMs across difficulty levels
- **Tool Usage Analysis**: Study how agents leverage tools (search, code execution, file handling)
- **Performance Optimization**: Identify bottlenecks through tracing and timing analysis
- **Research**: Dataset for studying agentic behavior and reasoning patterns

## 📝 Configuration

Key environment variables (see `.env` files in each scenario):
- `OPENAI_API_KEY` / model API keys
- `HF_TOKEN`: HuggingFace token for GAIA dataset access
- `LANGFUSE_*`: Optional tracing configuration
- Runtime: `MODEL_IDX`, `TEMPERATURE`, `NUM_QUESTIONS`, `TEST_LEVEL`, `OUTPUT_DIR`

## 🛠️ Requirements

- Docker + Docker Compose (recommended for GAIA)
- Python 3.10+ with dependencies from `requirements.txt`
- API access to LLM providers (OpenAI, Azure, etc.)
- HuggingFace account for GAIA dataset

## 📚 Documentation

- **GAIA_scenario/README.md**: Detailed setup and usage guide
- **GAIA_scenario/TESTING_GUIDE.md**: Testing procedures and best practices
- **grader.py**: Scoring logic and evaluation criteria

---

*Built for rigorous evaluation of agentic AI systems across real-world tasks.*
