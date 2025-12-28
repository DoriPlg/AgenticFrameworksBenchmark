import sys
from typing import Dict, Any, Optional, List
import time
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence, Literal
import shared_tools as st
from agents.base_agent import BaseAgent, AgentResponse

"""LangGraph agent implementation."""

sys.path.append('..')


# Graph State
class AgentGraphState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class LangGraphAgent(BaseAgent):
    """LangGraph-based agent implementation."""
    
    def __init__(self, model_config: Dict[str, Any], verbose: bool = False):
        super().__init__(model_config, verbose)
        
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
        
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()
    
    @property
    def name(self) -> str:
        return f"LangGraph-{self.model_config['model']}"
    
    def _agent_node(self, state: AgentGraphState) -> AgentGraphState:
        """Agent reasoning node."""
        messages = state.get("messages", [])
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: AgentGraphState) -> Literal["tools", "end"]:
        """Router: check if agent wants to use tools."""
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "end"
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph ReAct workflow."""
        workflow = StateGraph(AgentGraphState)
        
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue, {"tools": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def run(self, question: str, file_paths: Optional[List[str]] = None) -> AgentResponse:
        """Run agent on question."""
        start_time = time.time()
        
        file_context = ""
        if file_paths:
            file_context = f"\n\nATTACHED FILES: {', '.join(file_paths)}\nYou MUST inspect these files if relevant."
        
        system_prompt = (
            "You are an elite research assistant with expertise in information retrieval, "
            "data analysis, and problem-solving. You excel at breaking down complex questions, "
            "gathering information, and synthesizing accurate answers.\n\n"
            "Use the available tools when needed and provide clear, factual answers."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{question}{file_context}")
        ]
        
        result = self.graph.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )
        
        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        return AgentResponse(
            answer=answer,
            execution_time=time.time() - start_time,
            metadata={"framework": "langgraph", "model": self.model_config["model"]}
        )