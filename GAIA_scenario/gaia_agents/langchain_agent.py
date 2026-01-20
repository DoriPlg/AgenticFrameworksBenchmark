"""LangChain agent implementation using AgentExecutor."""
from typing import Dict, Any, Optional, List
import time

from langgraph.graph import StateGraph
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langfuse.langchain import CallbackHandler

from gaia_agents.tools import shared_tools as st
from gaia_agents.base_agent import BaseAgent, AgentResponse

langfuse_handler = CallbackHandler()

@tool(description=st.web_search.__doc__, args_schema=st.SearchInput)
def web_search(query: str) -> str:
    """Perform a web search and return results."""
    return st.web_search(query)

@tool(description=st.read_webpage.__doc__, args_schema=st.BrowserInput)
def read_webpage(url: str) -> str:
    """Read a webpage and return its content."""
    return st.read_webpage(url)

@tool(description=st.inspect_file.__doc__, args_schema=st.FileToolInput)
def inspect_file(file_path: str, query: str) -> str:
    """Inspect a file and return its content."""
    return st.inspect_file(file_path, query)

@tool(description=st.python_interpreter.__doc__, args_schema=st.PythonInput)
def python_interpreter(code: str) -> str:
    """Execute Python code and return the output."""
    return st.python_interpreter(code)

class LangChainAgent(BaseAgent):
    """LangChain AgentExecutor-based agent implementation."""
    
    def __init__(self, model_config: Dict[str, Any], verbose: bool = False, temperature: float = 0.0):
        super().__init__(model_config, verbose)
        self.model_config = model_config
        
        self.llm = ChatOpenAI(
            model=model_config['model'],
            base_url=model_config['base_url'],
            api_key=model_config['api_key'],
            temperature=temperature
        )
        
        # Create tools
        self.tools = [web_search, read_webpage, inspect_file, python_interpreter]
        
        # Create the agent with prompt
        self.agent = self._build_agent()
    
    @property
    def name(self) -> str:
        return f"LangChain-{self.model_config['model']}"
    
    def _build_agent(self) -> StateGraph:
        """Build the LangChain agent with AgentExecutor."""
        prompt = \
                "You are a general AI assistant. I will ask you a question. "\
                "Report your thoughts, and finish your answer with the following template: "\
                "FINAL ANSWER: [YOUR FINAL ANSWER]. "\
                "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. "\
                "If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. "\
                "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. "\
                "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n\n"\
                "Use the available tools when needed to gather information and solve problems."\
           
        
        return create_agent(model= self.llm,tools= self.tools,system_prompt= prompt)



    def run(self, question: str, file_paths: Optional[List[str]] = None) -> AgentResponse:
        """Run agent on question."""        
        start_time = time.time()
        
        file_context = ""
        if file_paths:
            file_context = f"\n\nATTACHED FILES: {', '.join(file_paths)}\nYou MUST inspect these files if relevant."
        
        full_question = f"{question}{file_context}"
        
        # create_agent expects messages in the correct format
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=full_question)]},
            config={"callbacks": [langfuse_handler], "recursion_limit": 50}
        )
        
        # Extract answer from the messages
        messages = result.get("messages", [])
        if messages:
            answer = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        else:
            answer = str(result)
        
        return AgentResponse(
            answer=answer,
            execution_time=time.time() - start_time,
            metadata={"framework": "langchain", "model": self.model_config["model"]}
        )

