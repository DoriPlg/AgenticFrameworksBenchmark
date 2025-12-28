from __future__ import annotations

import os
from typing import List, Type
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel

# --- LLM & Tracing Setup ---
import sys
sys.path.append('..')
from llmforall import get_llm_config
from langfuse import get_client

from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

import shared_tools as st

CrewAIInstrumentor().instrument(skip_dep_check=True)
LiteLLMInstrumentor().instrument()

langfuse = get_client()
llm_config_big = get_llm_config(0)

llm_big = LLM(
    model=f"openai/{llm_config_big['model']}",
    base_url=llm_config_big['base_url'],
    api_key=llm_config_big['api_key']
)

llm_config_small = get_llm_config(1)

llm_small = LLM(
    model=f"openai/{llm_config_small['model']}",
    base_url=llm_config_small['base_url'],
    api_key=llm_config_small['api_key']
)

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = st.web_search.__doc__
    args_schema: Type[BaseModel] = st.SearchInput

    def _run(self, *args, **kwargs) -> str:
        return st.web_search(query=kwargs.get('query'))
    
class WebBrowserTool(BaseTool):
    name: str = "web_browser"
    description: str = st.read_webpage.__doc__
    args_schema: Type[BaseModel] = st.BrowserInput

    def _run(self, *args, **kwargs) -> str:
        return st.read_webpage(url=kwargs.get('url'))
    
class FileInspectorTool(BaseTool):
    name: str = "file_inspector"
    description: str = st.inspect_file.__doc__
    args_schema: Type[BaseModel] = st.FileToolInput

    def _run(self, *args, **kwargs) -> str:
        return st.inspect_file(file_path=kwargs.get('file_path'))
    
class PythonExecutorTool(BaseTool):
    name: str = "python_executor"
    description: str = st.python_interpreter.__doc__
    args_schema: Type[BaseModel] = st.PythonInput

    def _run(self, *args, **kwargs) -> str:
        return st.python_interpreter(code=kwargs.get('code'))


# --- Agent Definition ---
# Single generalist agent with access to all tools for GAIA benchmark

researcher_agent = Agent(
    role="Expert Research and Analysis Assistant",
    goal="Answer complex, multi-step questions accurately by gathering information, analyzing data, and performing computations",
    backstory=(
        "You are an elite research assistant with expertise in information retrieval, "
        "data analysis, and problem-solving. You excel at:"
        "\n- Breaking down complex questions into manageable steps"
        "\n- Gathering information from web sources and documents"
        "\n- Processing and analyzing structured data (CSV, Excel, PDFs)"
        "\n- Performing mathematical calculations and data manipulations"
        "\n- Synthesizing findings into clear, accurate answers"
        "\n\nYou approach each question methodically, using the right tool for each task."
    ),
    tools=[WebSearchTool(), WebBrowserTool(), FileInspectorTool(), PythonExecutorTool()],
    llm=llm_big,
    verbose=True,
    allow_delegation=False,  # Single agent doesn't need delegation
    max_iter=15,  # Allow enough iterations for multi-step reasoning
)


# --- Task Creation Function ---

def create_gaia_task(question: str, file_paths: List[str] = None) -> Task:
    """
    Creates a GAIA benchmark task with proper context and instructions.
    
    Args:
        question: The GAIA question to answer
        file_paths: Optional list of file paths referenced in the question
    """
    file_context = ""
    if file_paths:
        file_context = f"\n\nATTACHED FILES: {', '.join(file_paths)}\nYou MUST inspect these files if they're relevant to the question."
    
    return Task(
        description=(
            f"Answer this question accurately and concisely:\n\n"
            f"QUESTION: {question}{file_context}\n\n"
            f"APPROACH:\n"
            f"1. Analyze what information or computation is needed\n"
            f"2. Use appropriate tools systematically:\n"
            f"   - web_search: Find general information or current facts\n"
            f"   - web_browser: Read specific webpages found via search\n"
            f"   - file_inspector: Examine attached files (CSV, Excel, PDF, etc.)\n"
            f"   - python_executor: Perform calculations, data analysis, or complex logic\n"
            f"3. Verify findings and cross-check if needed\n"
            f"4. Provide a clear, direct answer\n\n"
            f"IMPORTANT:\n"
            f"- For ANY calculation or data processing, use python_executor\n"
            f"- After inspecting files, use python_executor to analyze the data\n"
            f"- Your final answer should be concise and directly answer the question\n"
            f"- If uncertain, gather more information before concluding\n"
        ),
        expected_output="A clear, factual, and concise answer to the question",
        agent=researcher_agent,
    )


# --- Main Execution Function ---

def run_gaia_question(question: str, file_paths: List[str] = None) -> str:
    """
    Runs a single GAIA question through the CrewAI agent.
    
    Args:
        question: The GAIA question to answer
        file_paths: Optional list of file paths referenced in the question
        
    Returns:
        The agent's answer
    """
    task = create_gaia_task(question, file_paths)
    
    crew = Crew(
        agents=[researcher_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
        memory=False,  # GAIA is single-shot Q&A, no need for memory
    )
    
    result = crew.kickoff()
    return result


# --- Example Usage ---
if __name__ == "__main__":
    # Example GAIA-style question
    test_question = "What is the square root of 144 multiplied by 5?"
    
    print("=" * 80)
    print("GAIA BENCHMARK - CrewAI Implementation")
    print("=" * 80)
    print(f"\nQuestion: {test_question}\n")
    
    answer = run_gaia_question(test_question)
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(answer)
    print("\n")
