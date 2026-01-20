from __future__ import annotations

import os
import json
from typing import List, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- Pydantic & LangChain ---
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_community.utilities import GoogleSerperAPIWrapper


# --- LLM & Tracing Setup ---
from GAIA_scenario.llmforall import get_llm_config
from langfuse.langchain import CallbackHandler

from shared_tools import *


# --- Setup Langfuse ---
langfuse_handler = CallbackHandler()

# --- Setup LLM ---
llm_config_small = get_llm_config(1)
llm_small = ChatOpenAI(
    model=llm_config_small['model'],
    base_url=llm_config_small['base_url'],
    api_key=llm_config_small['api_key']
)

llm_config_big = get_llm_config(0)
llm_big = ChatOpenAI(
    model=llm_config_big['model'],
    base_url=llm_config_big['base_url'],
    api_key=llm_config_big['api_key']
)

# =========================================
# Tools
# =========================================

web_search_tool = StructuredTool.from_function(
    func=web_search,
    description="Search online for information.",
    args_schema=WebSearchInput
)

query_database_tool = StructuredTool.from_function(
    func=query_database,
    description="""Query the travel database for flights, hotels, attractions, and bookings.
    
    Available tables and columns:
    - flights: flight_id, airline, flight_number, origin_airport, destination_airport, 
    departure_date, departure_time, arrival_time, duration_minutes, base_price, 
    cabin_class, available_seats, aircraft_type
    
    - hotels: hotel_id, name, city, address, rating, price_per_night, 
    distance_to_center_km, available_rooms, has_wifi, has_pool, has_gym, 
    has_parking, has_spa, has_restaurant, has_bar, has_breakfast_included
    
    - attractions: attraction_id, name, city, category, rating, description, 
    average_visit_hours, entry_fee, website
    
    - bookings: booking_id, booking_type, item_id, customer_name, customer_email, 
    booking_date, status, confirmation_number, total_price, special_requests

    Write a SELECT query to retrieve the information needed.""",
    args_schema=SQLQueryInput
)

create_booking_tool = StructuredTool.from_function(
    func=create_booking,
    description="Create a new booking in the database.",
    args_schema=BookingInput
)


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
    verdict: str = Field(
        default="",
        description="Final verdict of the task, describing the hotel and flight booked."
    )


# ===============================================
#  Agents (LLMs with bound tools for ReAct)
# ===============================================

task_llm = llm_big.with_structured_output(TaskDelegation)

search_tools = [web_search_tool, query_database_tool]
search_llm = llm_big.bind_tools(search_tools)
search_tool_node = ToolNode(search_tools)

booking_tools = [create_booking_tool]
booking_llm = llm_small.bind_tools(booking_tools)
booking_tool_node = ToolNode(booking_tools)

# ===============================================
#  Nodes (LLMs with bound tools for ReAct)
# ===============================================

TASKER_INIT_PROMPT = """
    You are an experienced travel agency team leader. Your task is to find a vacation package for your customer.

    The client will provide you with a request, and you must break it down into small simple subtasks.
    Then you will delegate these subtasks to your team members ONE AT A TIME.

    Your team members are:
    - search: Can search online and in the database for flights, hotels, and attractions
    - booker: Can create new bookings in the database
    - end: Use this when all tasks are complete and you're ready to provide a final answer

    IMPORTANT: 
    - Delegate only ONE subtask at a time
    - Wait for the agent to complete before delegating the next subtask
    - Provide clear, specific instructions for each subtask

    Example delegation pattern:
    1. First: Delegate to 'search' to find cheapest flights
    2. Wait for response
    3. Then: Delegate to 'search' to find hotels
    4. Wait for response
    5. Then: Delegate to 'booker' to book the package
    6. Finally: Delegate to 'end' with final trip summary or an error message
    """

TASKER_CONT_PROMPT = """
    You are an experienced travel agency team leader. Your task is to find a vacation package for your customer.

    Review the conversation history and the most recent response from your team member.
    Decide on the NEXT subtask to delegate, or if all tasks are complete.

    Your team members are:
    - search: Can search online and in the database for flights, hotels, and attractions
    - booker: Can create new bookings in the database
    - end: Use this when all tasks are complete and you're ready to provide a final answer

    IMPORTANT: 
    - Delegate only ONE subtask at a time
    - Consider what information you now have and what you still need
    - Provide clear, specific instructions for the next subtask
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
    instruction_message = HumanMessage(
        content=delegation.instruction,
        name="task_orchestrator"  # Mark this as coming from orchestrator
    )
    
    return {
        "messages": [instruction_message],
        "next_agent": delegation.next_agent  # Store for router
    }

SEARCH_PROMPT = """
    You are a travel agent. Your task is to find a vacation package for your customer.
    
    You have access to the following tools:
    - web_search_tool: Search online for information.
    - query_database_tool: Query the travel database for flights, hotels, attractions, and bookings.
    Each time you access a table for the first time, always run "SELECT * FROM [table_name] LIMIT 3" WITHOUT ANY conditions in order to understand the table schema.
    Always prefer to use "query_database" over "web_search" when possible.
    """

RETURN_WEB_SEARCH_PROMPT = """
    You are a travel agent. Your task is to find a vacation package for your customer.
    
    Having just used the web_search_tool, you now have a list of relevant web pages.
    Go over them and provide an answer to the original task you were given.
    
    Respond with clean English text.
    """

RETURN_QUERY_DATABASE_PROMPT = """
    You are a travel agent. Your task is to find a vacation package for your customer.
    
    Having just used the query_database_tool, you now have a list of relevant database entries.
    Go over them and provide an answer to the original task you were given.
    
    Respond with clean English text.
    """


def search_node(state: VacationGraphState) -> VacationGraphState:
    """Search node - ReAct: Reasons and decides whether and how to search"""
    messages = state.get("messages", [])

    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]

    if any(msg.name == "web_search" for msg in tool_messages): # maybe need to check only last messages to avoid checking only old messages
        prompt = RETURN_WEB_SEARCH_PROMPT
    elif any(msg.name == "query_database" for msg in tool_messages):
        prompt = RETURN_QUERY_DATABASE_PROMPT
    else:
        prompt = SEARCH_PROMPT

    messages_with_prompt = [SystemMessage(content=prompt)] + messages
    response = search_llm.invoke(messages_with_prompt)

    return {"messages": [response]}

BOOKER_PROMPT = """
    You are a travel agent. Your task is to book the requested vacation package for your customer.
    
    You have access to the following tools:
    - create_booking_tool: Create a new booking in the database.
    """

def booker_node(state: VacationGraphState) -> VacationGraphState:
    """Booker node - ReAct: Uses the create_booking_tool to create a new booking in the database"""
    messages = state.get("messages", [])
    messages_with_prompt = [SystemMessage(content=BOOKER_PROMPT)] + messages
    response = booking_llm.invoke(messages_with_prompt)

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
    workflow.add_node("search_tools", search_tool_node)
    workflow.add_node("booker_tools", booking_tool_node)

    workflow.set_entry_point("task")

    workflow.add_conditional_edges("task", task_router, ["search", "booker", END])

    workflow.add_conditional_edges("search", should_use_tools, 
        {True: "search_tools", False: "task"})
    workflow.add_edge("search_tools", "search")

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

        