from __future__ import annotations

import os
from typing import List, Type
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
# --- LLM & Tracing Setup ---
from llmforall import get_llm_config
from langfuse import get_client

from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from shared_tools import *
 
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

if not os.getenv("SERPER_API_KEY"):
    raise EnvironmentError(
        "SERPER_API_KEY not found in .env file. "
        "Please get a key from serper.dev and add it to src/.env"
    )

# ==================================================
# Tools
# ==================================================

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search online for information."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str, max_results: int) -> List[WebSearchResult]:
        return web_search(query=query, max_results=max_results)

class QueryDatabaseTool(BaseTool):
    name: str = "query_database"
    description: str = """Query the travel database for flights, hotels, attractions, and bookings.
    
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

    Write a SELECT query to retrieve the information needed."""
    args_schema: Type[BaseModel] = SQLQueryInput

    def _run(self, query: str) -> SQLQueryResult:
        return query_database(query=query)

class CreateBookingTool(BaseTool):
    name: str = "create_booking"
    description: str = "Create a new booking in the database."
    args_schema: Type[BaseModel] = BookingInput

    def _run(self, booking_type: Literal["flight", "hotel"], item_id: str, 
        customer_name: str, customer_email: str, 
        special_requests: Optional[str] = None) -> Dict[str, Any]:
        return create_booking(booking_type=booking_type, item_id=item_id, customer_name=customer_name,
                                            customer_email=customer_email, special_requests=special_requests)

# ==================================================
# Agents
# ==================================================
sql_searcher_agent = Agent(
    role="SQL Searcher",
    goal="Search the database for information.",
    backstory=(
        """You are a SQL searcher, you can use the query_database tool to get the information requested of you.
        To better understand the databse ALWAYS start by running "SELECT * FROM [table_name] LIMIT 10" .
        Don't forget to return the IDs of the items you found."""
    ),
    tools=[QueryDatabaseTool()],
    verbose=True,
    llm=llm_big,
    allow_delegation=False,
    memory=True,
)

web_searcher_agent = Agent(
    role="Web Searcher",
    goal="Search online for information.",
    backstory=(
        "You are a web searcher, you can use the web_search tool to get the information."
    ),
    tools=[WebSearchTool()],
    verbose=True,
    llm=llm_small,
    allow_delegation=False,
    memory=True,
)

booking_agent = Agent(
    role="Booking Agent",
    goal="Create a new booking in the database.",
    backstory=(
        "You are a booking agent, you can use the create_booking tool to create a new booking."
    ),
    tools=[CreateBookingTool()],
    verbose=True,
    llm=llm_small,
    allow_delegation=False,
    memory=True,
)

orchestrator_agent = Agent(
    role="Orchestrator",
    goal="Manage the crew to complete the user's tasks.",
    backstory=(
        """You are the orchestrator, you manage the crew.
        At your disposal are the following agents:
        - SQL Searcher: Who has access to a database of flights and hotels.
        - Web Searcher: Who can search the web for information about attractions and activities.
        - Booking Agent: Who can create a new booking in the database, don't forget to pass him the id of the items you wish to book.
        """
    ),
    tools=[],
    verbose=True,
    llm=llm_big,
    allow_delegation=True,
    memory=True,
)

task = Task(
    description="""I need to find the cheapest vacation package to LA for a family of 4 living in New York, we are flexible with the dates and the destination.
    Book the cheapest vacation package for us, including flights there and back and hotel stay.
    After that add 2 attractions to the vacation package.""",
    expected_output="""A simple plan for a vacation."""
)

crew = Crew(
    agents=[sql_searcher_agent, web_searcher_agent, booking_agent],
    tasks=[task],
    manager_agent=orchestrator_agent,
    process=Process.hierarchical,
    verbose=False,
)

if __name__ == "__main__":
    with langfuse.start_as_current_observation(
        name="VacationPlannerHierarchical",
        metadata={"crew_type": "hierarchical", "goal": "Find the cheapest vacation package"},
    ) as obs:
        try:
            print("Kicking off the Vacation Planner Crew...")
            result = crew.kickoff()
            print(result)
            
        except Exception as e:
            print(f"Error: {e}")
