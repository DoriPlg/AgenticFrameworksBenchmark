"""OpenAI Agents framework implementation."""
from typing import Dict, Any, Optional, List
import time
import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, SQLiteSession, ModelSettings
from agents.run import RunContextWrapper

import nest_asyncio
# from langfuse import get_client
# from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

from gaia_agents.tools import shared_tools as st
from gaia_agents.base_agent import BaseAgent, AgentResponse

nest_asyncio.apply()
# OpenAIAgentsInstrumentor().instrument()
# langfuse = get_client()

class OpenAIAgent(BaseAgent):
    """OpenAI Agents framework-based agent implementation."""
    
    def __init__(self, model_config: Dict[str, Any], verbose: bool = False, temperature: float = 0.0):
        super().__init__(model_config, verbose, temperature)
        
        self.model = OpenAIChatCompletionsModel(
            model=self.model_config['model'],
            openai_client=AsyncOpenAI(
                base_url=self.model_config['base_url'],
                api_key=self.model_config['api_key']
            )
        )
        self.model_settings = ModelSettings(
            temperature=self.temperature
        )
        self._name = f"OpenAI-{self.model_config['model']}"
        
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
            "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
        )
        
        agent = Agent(
            model=self.model,
            name="research_agent",
            instructions=system_prompt,
            tools=self.tools,
            model_settings=self.model_settings,
        )
        
        return agent
    
    def run(self, question: str, file_paths: Optional[List[str]] = None) -> AgentResponse:
        """Run agent on question."""
        start_time = time.time()
        
        file_context = f"\n\nATTACHED FILES: {', '.join(file_paths)}\nYou MUST inspect these files if relevant." \
            if file_paths else ""
        
        full_question = f"{question}{file_context}"
        
        # Run the agent asynchronously
        # with langfuse.start_as_current_observation(
        #     name="OpenAI GAIA Attempt", 
        #     metadata={"framework": "openai_agents", "model": self.model_config["model"]},
        #     input=full_question
        # ) as observation:
        result = asyncio.run(self._run_async(full_question))
        #     observation.update(output=result)
        # langfuse.flush()
        
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
        return result.final_output
    
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
