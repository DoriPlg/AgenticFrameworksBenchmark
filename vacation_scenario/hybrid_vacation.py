from __future__ import annotations


from typing import List, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


from typing import List, Type
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool

# --- Pydantic & LangChain ---
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_community.utilities import GoogleSerperAPIWrapper


# --- LLM & Tracing Setup ---
from GAIA_scenario.llmforall import get_llm_config
from langfuse.langchain import CallbackHandler

from shared_tools import *
from langfuse import get_client

from openinference.instrumentation.crewai import CrewAIInstrumentor
# from openinference.instrumentation.litellm import LiteLLMInstrumentor
from shared_tools import *
 
CrewAIInstrumentor().instrument(skip_dep_check=True)
# LiteLLMInstrumentor().instrument()

 
# langfuse = get_client()
llm_config_big = get_llm_config(0)
llm_config_small = get_llm_config(1)
    
# --- Setup Langfuse ---
langfuse_handler = CallbackHandler()

# --- Setup LLM access for both frameworks ---
llm_small_lang = ChatOpenAI(
    model=llm_config_small['model'],
    base_url=llm_config_small['base_url'],
    api_key=llm_config_small['api_key']
)

llm_big_lang = ChatOpenAI(
    model=llm_config_big['model'],
    base_url=llm_config_big['base_url'],
    api_key=llm_config_big['api_key']
)

llm_big_crew = LLM(
    model=f"openai/{llm_config_big['model']}",
    base_url=llm_config_big['base_url'],
    api_key=llm_config_big['api_key']
)


llm_small_crew = LLM(
    model=f"openai/{llm_config_small['model']}",
    base_url=llm_config_small['base_url'],
    api_key=llm_config_small['api_key']
)


# =========================================
# Tools
# =========================================
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search online for information."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str, max_results: int) -> List[WebSearchResult]:
        return web_search(query=query, max_results=max_results)

class QueryDatabaseTool(BaseTool):
    name: str = "query_database"
    description: str = """Query the travel database for flights, hotels, attractions, and bookings.
    
    Available tables:
    - flights
    - hotels
    - attractions
    - bookings

    Whenever you first access a table, always run "SELECT * FROM [table_name] LIMIT 3" WITHOUT ANY conditions in order to understand the table schema.
    Write a SELECT query to retrieve the information needed."""
    args_schema: Type[BaseModel] = SQLQueryInput

    def _run(self, query: str) -> SQLQueryResult:
        return query_database(query=query)

create_booking_tool = StructuredTool.from_function(
    func=create_booking,
    description="Create a new booking in the database.",
    args_schema=BookingInput
)

class DictionaryOutput(BaseModel):
    output: Dict[str, Any]


# =========================================
# Graph State
# =========================================

class VacationGraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str  # <-- Add this to track routing decision

# ===============================================
#  Structured Output for Task Delegation
# ===============================================

class TaskDelegation(BaseModel):
    """Structured output for task delegation decisions"""
    next_agent: Literal["search", "booker", "end"] = Field(
        description="Which agent to delegate to: 'search' for searching, 'booker' for booking, 'end' when complete"
    )
    instruction: str = Field(
        description="Clear instruction/subtask for the chosen agent"
    )
    reasoning: str = Field(
        description="Brief explanation of why this delegation was chosen"
    )


# ===============================================
#  Agents (LLMs with bound tools for ReAct)
# ===============================================

task_llm = llm_big_lang.with_structured_output(TaskDelegation)

searcher_agent = Agent(
    role="SQL Searcher",
    goal="Search the database for information.",
    backstory=(
        """
        You work at a travel agency. Your task is to search for a vacation package for your customer.
        
        You have access to the following tools:
        - web_search: Search online for information. Each time you access a table for the first time, always run "SELECT * FROM [table_name] LIMIT 3" WITHOUT ANY conditions in order to understand the table schema.
        - query_database: Query the your travel database for flights, hotels, attractions, and bookings.

        IMPORTANT: 
        - Always prefer to use "query_database" over "web_search" when possible.
        - Always respond with the complete result upon completion to whoever tasked you, including the item's ID.
    """),
    tools=[QueryDatabaseTool(), WebSearchTool()],
    verbose=True,
    llm=llm_big_crew,
    allow_delegation=False,
    memory=True,
)

booking_tools = [create_booking_tool]
booking_llm = llm_small_lang.bind_tools(booking_tools)
booking_tool_node = ToolNode(booking_tools)

# ===============================================
#  Nodes (LLMs with bound tools for ReAct)
# ===============================================

TASKER_INIT_PROMPT = """
    **Persona:** Act as an experienced travel agency team leader responsible for fulfilling client vacation requests.

    **Context:** You are managing a team of specialized agents to plan and book a vacation package for a client as specified in messages[0].

    **Task:** Break down the client's request into a series of small, manageable subtasks and delegate them to your team members one at a time.  Coordinate the team to efficiently fulfill the request.

    **Team Members:**

    *   **search:**  Specializes in finding flights, hotels, and attractions.  Instructions *must* include the word "search" and be highly specific.
    *   **booker:**  Responsible for creating bookings in the database. Instructions *must* include the word "book" and be highly specific.
    *   **end:** Signals completion of all tasks and readiness to provide a final answer.

    **Instructions:**

    1.  Delegate *only one* subtask at a time.
    2.  Wait for a response from the agent before delegating the next subtask.
    3.  Provide clear, precise, and complete instructions for each subtask. Avoid ambiguous or single-word commands.
    4.  Delegate to 'end' *only* when all tasks are completed and you are ready to present the final trip summary or an error message.

    **Output Format:**

    Present the final output as a concise trip summary including: flight details, hotel information, booked attractions (if any), total cost, and a confirmation number. If an error occurred during the process, provide a clear error message explaining the issue.

    **Goal:** To efficiently and accurately plan and book a vacation package based on the client's request.

    **Example Delegation Pattern (for reference):**

    1.  Delegate to 'search' to find the cheapest round-trip flights for the specified dates and destination.
    2.  Wait for response.
    3.  Delegate to 'search' to find hotels that meet the client's criteria (e.g., price range, star rating, amenities).
    4.  Wait for response.
    5.  Delegate to 'booker' to book the selected flights and hotel.
    6.  Finally, delegate to 'end' with the complete trip summary or an error message.
    """

TASKER_CONT_PROMPT = """
    You are an experienced travel agency team leader. Your task is to find a vacation package for your customer.

    Review the conversation history and the most recent response from your team member.
    Decide on the NEXT subtask to delegate, or if all tasks are complete.
    """

def task_node(state: VacationGraphState) -> VacationGraphState:
    """
    Task node - Orchestrator that delegates subtasks to appropriate agents.
    Uses structured output to determine next agent and instruction.
    """
    messages = state.get("messages", [])
    
    # Use different prompt for initial vs continuation
    if len(messages) <= 1:  # Initial request
        system_prompt = TASKER_INIT_PROMPT
    else:  # Continuation after agent response
        system_prompt = TASKER_CONT_PROMPT
    
    messages_with_prompt = [SystemMessage(content=system_prompt)] + messages
    
    # Get structured delegation decision
    delegation = task_llm.invoke(messages_with_prompt)
    
    # Store delegation decision in state for routing
    # Create a message that includes the instruction for the next agent
    instruction_message = AIMessage(
        content=delegation.instruction,
        name="task_orchestrator"  # Mark this as coming from orchestrator
    )
    
    return {
        "messages": [instruction_message],
        "next_agent": delegation.next_agent  # Store for router
    }


def search_node(state: VacationGraphState) -> VacationGraphState:
    """Search node - ReAct: Reasons and decides whether and how to search"""
    messages = state.get("messages", [])
    if len(messages) == 0:
        raise ValueError("No messages found")
    print("The task is: ", messages[-1].content)
    response = searcher_agent.kickoff(messages[-1].content, response_format=DictionaryOutput)
    print(response.raw)
    return {"messages": [AIMessage(content= response.raw, name="searcher")]} 

BOOKER_PROMPT = """
    You are a travel agent. Your task is to book the requested vacation package for your customer.
    
    You have access to the following tools:
    - create_booking_tool: Create a new booking in the database.
    """

def booker_node(state: VacationGraphState) -> VacationGraphState:
    """Booker node - ReAct: Uses the create_booking_tool to create a new booking in the database"""
    messages = state.get("messages", [])
    last_speakers = [message.name for message in messages[-8:]]
    if "task_orchestrator" not in last_speakers:
        return {"messages": [AIMessage(
            content="I couldn't complete the given task, please rethink and try again", 
            name="booker")]} 
    messages_with_prompt = [SystemMessage(content=BOOKER_PROMPT)] + messages
    response = booking_llm.invoke(messages_with_prompt)
    try:
        calls = response.tool_calls
    except:
        return {"messages": [response]}
    if len(calls) == 0:
        return {"messages": [HumanMessage(content="No tool calls found")]}
    if len(calls) > 1:
        response.tool_calls = calls[0]
        response.content += "\n\nFYI only the first tool call was used. The model accepts only one tool call at a time."
    return {"messages": [response]}

# ===============================================
#  Routing
# ===============================================

def should_use_tools(state: VacationGraphState) -> bool:
    """Routing function: check if agent wants to use tools"""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return True
    return False

def task_router(state: VacationGraphState) -> Literal["search_node", "booker_node", END]:
    """Routing function: check if vacation was booked"""
    next_agent = state.get("next_agent", "end")
    
    if next_agent == "search":
        return "search"
    elif next_agent == "booker":
        return "booker"
    else: 
        return END

def build_graph():
    """Builds the LangGraph ReAct state machine.
    
    Flow demonstrates true ReAct pattern:
    - Agent reasons and decides to use tools OR respond
    - ToolNode automatically executes tools when requested
    - Agent observes results and continues
    """
    workflow = StateGraph(VacationGraphState)
    workflow.add_node("task", task_node)
    workflow.add_node("search", search_node)
    workflow.add_node("booker", booker_node)
    workflow.add_node("booker_tools", booking_tool_node)

    workflow.set_entry_point("task")

    workflow.add_conditional_edges("task", task_router, ["search", "booker", END])

    workflow.add_edge("search", "task")

    workflow.add_conditional_edges("booker", should_use_tools, 
        {True: "booker_tools", False: "task"})
    workflow.add_edge("booker_tools", "booker")

    return workflow.compile()
        
# ===============================================
#  Run
# ===============================================

def main():
    """Main function demonstrating LangGraph ReAct framework with multiple agents."""
    question="""I need to find the cheapest vacation package to LA for a family of 4 living in New York, we are flexible with the dates and the destination.
    Book the cheapest vacation package for us, including flights there and back and hotel stay.
    After that add 2 attractions to the vacation package."""
    try:
        app = build_graph()

        state = 0
        for _ in app.stream({"messages": [HumanMessage(content = question)]},
                    config={"callbacks": [langfuse_handler], "recursion_limit": 50}):
            state+=1
            print(state)
    except Exception as e:
        error_msg = f"Error during execution: {str(e)}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

