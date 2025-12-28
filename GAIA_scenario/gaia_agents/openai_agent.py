"""OpenAI Agents framework implementation."""
from typing import Dict, Any, Optional, List
import time
import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, SQLiteSession
from agents.run import RunContextWrapper

from gaia_agents.tools import shared_tools as st
from gaia_agents.base_agent import BaseAgent, AgentResponse


class OpenAIAgent(BaseAgent):
    """OpenAI Agents framework-based agent implementation."""
    
    def __init__(self, model_config: Dict[str, Any], verbose: bool = False):
        super().__init__(model_config, verbose)
        
        self.model = OpenAIChatCompletionsModel(
            model=model_config['model'],
            openai_client=AsyncOpenAI(
                base_url=model_config['base_url'],
                api_key=model_config['api_key']
            )
        )
        
        # Create tools using function_tool decorator
        self.tools = [
            function_tool(st.web_search),
            function_tool(st.read_webpage),
            function_tool(st.inspect_file),
            function_tool(st.python_interpreter),
        ]
        
        self.agent = self._build_agent()
        self.session = SQLiteSession("OpenAIAgentSession")
    
    @property
    def name(self) -> str:
        return f"OpenAI-{self.model_config['model']}"
    
    def _build_agent(self) -> Agent:
        """Build the OpenAI Agent."""
        system_prompt = (
            "You are an elite research assistant with expertise in information retrieval, "
            "data analysis, and problem-solving. You excel at breaking down complex questions, "
            "gathering information, and synthesizing accurate answers.\n\n"
            "Use the available tools when needed and provide clear, factual answers."
        )
        
        agent = Agent(
            model=self.model,
            name="research_agent",
            instructions=system_prompt,
            tools=self.tools,
        )
        
        return agent
    
    def run(self, question: str, file_paths: Optional[List[str]] = None) -> AgentResponse:
        """Run agent on question."""
        start_time = time.time()
        
        file_context = ""
        if file_paths:
            file_context = f"\n\nATTACHED FILES: {', '.join(file_paths)}\nYou MUST inspect these files if relevant."
        
        full_question = f"{question}{file_context}"
        
        # Run the agent asynchronously
        result = asyncio.run(self._run_async(full_question))
        
        # Extract answer from result
        answer = self._extract_answer(result)
        
        return AgentResponse(
            answer=answer,
            execution_time=time.time() - start_time,
            metadata={"framework": "openai_agents", "model": self.model_config["model"]}
        )
    
    async def _run_async(self, question: str) -> Any:
        """Run the agent asynchronously."""
        result = await Runner.run(
            self.agent,
            input=question,
            max_turns=50,
            session=self.session
        )
        return result
    
    def _extract_answer(self, result: Any) -> str:
        """Extract the answer from the agent result."""
        if hasattr(result, 'content'):
            return result.content
        elif hasattr(result, 'text'):
            return result.text
        elif isinstance(result, str):
            return result
        else:
            return str(result)
