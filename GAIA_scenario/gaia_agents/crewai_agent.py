"""CrewAI agent implementation."""
from typing import Dict, Any, Optional, List
import time
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel

from gaia_agents.tools import shared_tools as st
from gaia_agents.base_agent import BaseAgent, AgentResponse

# from langfuse import get_client

from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

CrewAIInstrumentor().instrument(skip_dep_check=True)
LiteLLMInstrumentor().instrument()

# langfuse = get_client()
# # Verify connection
# if langfuse.auth_check():
#     print("Langfuse client is authenticated and ready!")
# else:
#     print("Authentication failed. Please check your credentials and host.")



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
    
    def __init__(self, model_config: Dict[str, Any], verbose: bool = False, temperature: float = 0.0):
        super().__init__(model_config, verbose, temperature)
        self.llm = LLM(
            model=f"openai/{self.model_config['model']}",
            base_url=self.model_config['base_url'],
            api_key=self.model_config['api_key'],
            temperature=self.temperature,
        )
        self._name = f"CrewAI-{self.model_config['model']}"
        
        self.agent = Agent(
            role="Expert Research and Analysis Assistant",
            goal="Answer complex, multi-step questions accurately",
            backstory=(
                "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
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

        # with langfuse.start_as_current_observation(as_type= "span",
        #     name="CrewAI GAIA Attempt", 
        #     metadata={"framework": "crewai", "model": self.model_config["model"]},
        #     input= {"question": question}
        #     ) as observation:
        result = crew.kickoff()
            # observation.update(output= result)
        
        # langfuse.flush()
        return AgentResponse(
            answer=str(result),
            execution_time=time.time() - start_time,
            metadata={"framework": "crewai", "model": self.model_config["model"]}
        )
        
 