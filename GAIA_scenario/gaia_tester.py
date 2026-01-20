"""
GAIA Benchmark Test Runner - Environment Variable Driven
Usage: FRAMEWORK=crewai MODEL_SIZE=large NUM_QUESTIONS=5 python3 gaia_tester.py
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Disable LiteLLM's verbose error logging before any imports
logging.getLogger('LiteLLM').setLevel(logging.CRITICAL)
logging.getLogger('LiteLLM').propagate = False

from datasets import load_dataset

sys.path.append('..')
from llmforall import get_llm_config
from gaia_agents.base_agent import BaseAgent, AgentResponse

# Agent registry - add your agents here
AGENT_REGISTRY: Dict[str, type[BaseAgent]] = {}

def register_agent(name):
    """Decorator to register agents."""
    def decorator(cls):
        AGENT_REGISTRY[name] = cls
        return cls
    return decorator

# Import and register agents
try:
    from gaia_agents.crewai_agent import CrewAIAgent
    AGENT_REGISTRY['crewai'] = CrewAIAgent
except ImportError as e:
    print(f"Warning: CrewAI agent not available - {e}")

try:
    from gaia_agents.langgraph_agent import LangGraphAgent
    AGENT_REGISTRY['langgraph'] = LangGraphAgent
except ImportError as e:
    print(f"Warning: LangGraph agent not available - {e}")

try:
    from gaia_agents.langchain_agent import LangChainAgent
    AGENT_REGISTRY['langchain'] = LangChainAgent
except ImportError as e:
    print(f"Warning: LangChain agent not available - {e}")

try:
    from gaia_agents.openai_agent import OpenAIAgent
    AGENT_REGISTRY['openai'] = OpenAIAgent
except ImportError as e:
    print(f"Warning: OpenAI agent not available - {e}")


def load_gaia_dataset(lvl):
    """
    Load GAIA dataset from environment or cache.
    args:
        lvl (int): Level of the GAIA dataset to load.
    returns:
        dataset: Loaded dataset object.
        data_dir: Path to the dataset directory.
    """
    data_dir = os.getenv("GAIA_DATA_DIR")
    if not data_dir:
        # Find in cache
        cache_base = "/app/hf_cache/hub/datasets--gaia-benchmark--GAIA/snapshots"
        if os.path.exists(cache_base):
            snapshots = os.listdir(cache_base)
            if snapshots:
                data_dir = os.path.join(cache_base, snapshots[0])
    
    if not data_dir:
        raise RuntimeError("GAIA dataset not found. Run data_pull.py first.")
    
    dataset = load_dataset(data_dir, f"2023_level{lvl}", split="validation")
    return dataset, data_dir


def run_test():
    """Main test function driven by environment variables."""
    # Read configuration from environment
    framework = os.getenv("FRAMEWORK", "crewai")
    model_idx = int(os.getenv("MODEL_IDX", "0"))
    num_questions = int(os.getenv("NUM_QUESTIONS", "5"))
    start_idx = int(os.getenv("START_IDX", "0"))
    output_dir = os.getenv("OUTPUT_DIR", "/app/output")
    test_mode = os.getenv("TEST_MODE", "single")  # single or compare
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    
    print(f"\n{'='*80}")
    print(f"GAIA BENCHMARK TEST")
    print(f"{'='*80}")
    print(f"Framework: {framework}")
    print(f"Model Number: {model_idx}")
    print(f"Questions: {num_questions} (starting at {start_idx})")
    print(f"Test Mode: {test_mode}")
    print(f"Temperature: {temperature}")
    print(f"{'='*80}\n")
    
    # Load dataset
    print("Loading GAIA dataset...")
    dataset, data_dir = load_gaia_dataset(lvl=1)
    print(f"✓ Loaded {len(dataset)} test examples\n")
    
    # Get model config
    model_config = get_llm_config(model_idx)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    if test_mode == "compare":
        # Compare all available frameworks
        results = compare_frameworks(
            dataset, data_dir, model_config,
            num_questions, start_idx, output_dir, temperature
        )
    else:
        # Single framework test
        results = test_single_framework(
            framework, dataset, data_dir, model_config,
            num_questions, start_idx, output_dir, temperature
        )
    
    return results


def test_single_framework(framework, dataset, data_dir, model_config, 
                          num_questions, start_idx, output_dir, temperature=0.0):
    """Test a single framework."""
    if framework not in AGENT_REGISTRY:
        print(f"ERROR: Framework '{framework}' not available.")
        print(f"Available frameworks: {list(AGENT_REGISTRY.keys())}")
        sys.exit(1)
    
    # Create agent
    AgentClass = AGENT_REGISTRY[framework]
    agent = AgentClass(model_config, verbose=False, temperature=temperature)
    
    print(f"Testing: {agent.name}\n")
    
    # Run tests
    results = {
        "agent_info": {
            "name": agent.name,
            "framework": framework,
            "model": model_config["model"],
        },
        "test_config": {
            "num_questions": num_questions,
            "start_idx": start_idx,
            "timestamp": datetime.now().isoformat()
        },
        "questions": []
    }
    
    end_idx = min(start_idx + num_questions, len(dataset))
    
    for idx in range(start_idx, end_idx):
        example = dataset[idx]
        question = example["Question"]
        correct_answer = example.get("Final answer", "")
        
        print(f"\n[{idx+1}/{len(dataset)}] Question: {question[:100]}...")
        
        # Get file paths
        file_paths = None
        if "file_name" in example and example["file_name"]:
            file_paths = [f"{data_dir}/2023/test/{example['file_name']}"]
        
        # Run agent
        try:
            start_time = time.time()
            response = agent.run(question, file_paths)
            
            print(f"Answer: {response.answer[:200]}...")
            print(f"Time: {response.execution_time:.2f}s")
            
            question_result = {
                "idx": idx,
                "question": question,
                "correct_answer": correct_answer,
                "agent_answer": response.answer,
                "execution_time": response.execution_time,
                "success": True,
            }
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            question_result = {
                "idx": idx,
                "question": question,
                "correct_answer": correct_answer,
                "agent_answer": None,
                "error": str(e),
                "success": False
            }
        
        results["questions"].append(question_result)
    
    # Calculate summary
    successful = [q for q in results["questions"] if q["success"]]
    results["summary"] = {
        "total_questions": len(results["questions"]),
        "successful_runs": len(successful),
        "failed_runs": len(results["questions"]) - len(successful),
        "avg_execution_time": sum(q.get("execution_time", 0) for q in successful) / len(successful) if successful else 0,
        "total_time": sum(q.get("execution_time", 0) for q in successful)
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(output_dir) / f"{framework}_{model_config['model']}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {filename}")
    print(f"\nSummary: {results['summary']['successful_runs']}/{results['summary']['total_questions']} successful")
    print(f"Average time: {results['summary']['avg_execution_time']:.2f}s")
    
    return results


def compare_frameworks(dataset, data_dir, model_config, num_questions, start_idx, output_dir, temperature=0.0):
    """Compare all available frameworks."""
    print(f"Comparing {len(AGENT_REGISTRY)} frameworks...\n")
    
    all_results = {}
    
    for framework_name in AGENT_REGISTRY.keys():
        print(f"\n{'='*80}")
        print(f"Testing {framework_name.upper()}")
        print(f"{'='*80}")
        
        results = test_single_framework(
            framework_name, dataset, data_dir, model_config,
            num_questions, start_idx, output_dir, temperature
        )
        all_results[framework_name] = results
    
    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(output_dir) / f"comparison_{model_config['model']}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Framework':<20} {'Success':<15} {'Avg Time (s)':<15} {'Total Time (s)':<15}")
    print(f"{'-'*80}")
    
    for framework, results in all_results.items():
        summary = results["summary"]
        print(f"{framework:<20} "
              f"{summary['successful_runs']}/{summary['total_questions']:<15} "
              f"{summary['avg_execution_time']:<15.2f} "
              f"{summary['total_time']:<15.2f}")
    
    print(f"\n✓ Comparison saved to: {filename}\n")
    
    return all_results


if __name__ == "__main__":
    run_test()
