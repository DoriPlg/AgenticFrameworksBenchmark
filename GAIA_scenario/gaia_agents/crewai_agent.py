"""CrewAI agent implementation."""
from typing import Dict, Any, Optional, List
import time
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel

from gaia_agents.tools import shared_tools as st
from gaia_agents.base_agent import BaseAgent, AgentResponse


# Tool wrappers for CrewAI
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = st.web_search.__doc__
    args_schema: type[BaseModel] = st.SearchInput
    def _run(self, *args, **kwargs) -> str:
        return st.web_search(query=kwargs.get('query'))


class WebBrowserTool(BaseTool):
    name: str = "web_browser"
    description: str = st.read_webpage.__doc__
    args_schema: type[BaseModel] = st.BrowserInput
    def _run(self, *args, **kwargs) -> str:
        return st.read_webpage(url=kwargs.get('url'))


class FileInspectorTool(BaseTool):
    name: str = "file_inspector"
    description: str = st.inspect_file.__doc__
    args_schema: type[BaseModel] = st.FileToolInput
    def _run(self, *args, **kwargs) -> str:
        return st.inspect_file(file_path=kwargs.get('file_path'))


class PythonExecutorTool(BaseTool):
    name: str = "python_executor"
    description: str = st.python_interpreter.__doc__
    args_schema: type[BaseModel] = st.PythonInput
    def _run(self, *args, **kwargs) -> str:
        return st.python_interpreter(code=kwargs.get('code'))


class CrewAIAgent(BaseAgent):
    """CrewAI-based agent implementation."""
    
    def __init__(self, model_config: Dict[str, Any], verbose: bool = False):
        super().__init__(model_config, verbose)
        
        self.llm = LLM(
            model=f"openai/{model_config['model']}",
            base_url=model_config['base_url'],
            api_key=model_config['api_key']
        )
        
        self.agent = Agent(
            role="Expert Research and Analysis Assistant",
            goal="Answer complex, multi-step questions accurately",
            backstory=(
                "You are an elite research assistant with expertise in information retrieval, "
                "data analysis, and problem-solving. You excel at breaking down complex questions, "
                "gathering information, and synthesizing accurate answers."
            ),
            tools=[WebSearchTool(), WebBrowserTool(), FileInspectorTool(), PythonExecutorTool()],
            llm=self.llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=15,
        )
    
    @property
    def name(self) -> str:
        return f"CrewAI-{self.model_config['model']}"
    
    def run(self, question: str, file_paths: Optional[List[str]] = None) -> AgentResponse:
        """Run agent on question."""
        start_time = time.time()
        
        file_context = ""
        if file_paths:
            file_context = f"\n\nATTACHED FILES: {', '.join(file_paths)}\nYou MUST inspect these files if relevant."
        
        task = Task(
            description=(
                f"Answer this question accurately and concisely:\n\n"
                f"QUESTION: {question}{file_context}\n\n"
                f"Use appropriate tools (web_search, file_inspector, python_executor) and provide a clear answer."
            ),
            expected_output="A clear, factual answer to the question",
            agent=self.agent,
        )
        
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
            memory=False,
        )
        
        result = crew.kickoff()
        
        return AgentResponse(
            answer=str(result),
            execution_time=time.time() - start_time,
            metadata={"framework": "crewai", "model": self.model_config["model"]}
        )
