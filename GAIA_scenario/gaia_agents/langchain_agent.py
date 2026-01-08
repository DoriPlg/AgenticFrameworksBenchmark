"""LangChain agent implementation using AgentExecutor."""
from typing import Dict, Any, Optional, List
import time

from langgraph.graph import StateGraph
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langfuse.langchain import CallbackHandler

from gaia_agents.tools import shared_tools as st
from gaia_agents.base_agent import BaseAgent, AgentResponse

langfuse_handler = CallbackHandler()


class LangChainAgent(BaseAgent):
    """LangChain AgentExecutor-based agent implementation."""
    
    def __init__(self, model_config: Dict[str, Any], verbose: bool = False):
        super().__init__(model_config, verbose)
        self.model_config = model_config
        
        self.llm = ChatOpenAI(
            model=model_config['model'],
            base_url=model_config['base_url'],
            api_key=model_config['api_key']
        )
        
        # Create tools
        self.tools = [
            StructuredTool.from_function(
                func=st.web_search,
                description=st.web_search.__doc__,
                args_schema=st.SearchInput
            ),
            StructuredTool.from_function(
                func=st.read_webpage,
                description=st.read_webpage.__doc__,
                args_schema=st.BrowserInput
            ),
            StructuredTool.from_function(
                func=st.inspect_file,
                description=st.inspect_file.__doc__,
                args_schema=st.FileToolInput
            ),
            StructuredTool.from_function(
                func=st.python_interpreter,
                description=st.python_interpreter.__doc__,
                args_schema=st.PythonInput
            ),
        ]
        
        # Create the agent with prompt
        self.agent = self._build_agent()
    
    @property
    def name(self) -> str:
        return f"LangChain-{self.model_config['model']}"
    
    def _build_agent(self) -> StateGraph:
        """Build the LangChain agent with AgentExecutor."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a general AI assistant. I will ask you a question. "
                "Report your thoughts, and finish your answer with the following template: "
                "FINAL ANSWER: [YOUR FINAL ANSWER]. "
                "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. "
                "If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. "
                "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. "
                "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n\n"
                "Use the available tools when needed to gather information and solve problems."
            )),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return create_agent(model= self.llm,tools= self.tools,system_prompt= prompt)



    def run(self, question: str, file_paths: Optional[List[str]] = None) -> AgentResponse:
        """Run agent on question."""
        start_time = time.time()
        
        file_context = ""
        if file_paths:
            file_context = f"\n\nATTACHED FILES: {', '.join(file_paths)}\nYou MUST inspect these files if relevant."
        
        full_question = f"{question}{file_context}"
        
        result = self.agent.invoke(
            {"input": full_question},
            config={"callbacks": [langfuse_handler], "max_recursion_depth": 50}
        )
        
        answer = result.get("output", str(result))
        
        return AgentResponse(
            answer=answer,
            execution_time=time.time() - start_time,
            metadata={"framework": "langchain", "model": self.model_config["model"]}
        )

