import os
import asyncio

from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, memory, trace, SQLiteSession, RunContextWrapper
from llmforall import get_llm_config


# --- Langfuse Setup ---
from langfuse import get_client
import nest_asyncio
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
 
nest_asyncio.apply()
OpenAIAgentsInstrumentor().instrument()
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

def create_models():
    llm_config_small = get_llm_config(model_choice=1)
    small_model = OpenAIChatCompletionsModel( 
        model=llm_config_small["model"],
        openai_client=AsyncOpenAI(base_url=llm_config_small["base_url"], api_key=llm_config_small["api_key"]),
    )
    llm_config_big = get_llm_config(model_choice=4)
    large_model = OpenAIChatCompletionsModel( 
        model=llm_config_big["model"],
        openai_client=AsyncOpenAI(base_url=llm_config_big["base_url"], api_key=llm_config_big["api_key"]),
    )
    return {"small_model": small_model, "large_model": large_model}

def prepare_function_tools():
    from shared_tools import web_search, query_database, create_booking
    web_search = function_tool(web_search)
    query_database = function_tool(query_database)
    create_booking = function_tool(create_booking)
    return {"web_search": web_search, "query_database": query_database, "create_booking": create_booking}

SEARCH_PROMPT="""
    You work at a travel agency. Your task is to search for a vacation package for your customer.
    
    You have access to the following tools:
    - web_search: Search online for information. Each time you access a table for the first time, always run "SELECT * FROM [table_name] LIMIT 3" WITHOUT ANY conditions in order to understand the table schema. Then you may query it as you see fit.
    - query_database: Query the your travel database for flights, hotels, attractions, and bookings.

    IMPORTANT: 
    - Always prefer to use "query_database" over "web_search" when possible.
    - Always respond with the complete result upon completion to whoever tasked you, including the item's ID.
    """

BOOKER_PROMPT="""
    You work at a travel agency. Your task is to book the requested vacation package for your customer.
    
    You have access to the following tools:
    - create_booking: Create a new booking in the database.

    When you have booked the vacation package respond with it to whoever tasked you.
    """

LEADER_PROMPT="""
    **Persona:** Act as an experienced travel agency executive specializing in personalized vacation planning.

    **Context:** You are interacting with a customer to fulfill their vacation request.

    **Task:** A client has provided a vacation request (detailed below). Your task is to break down the request into a series of small, simple subtasks, then execute those subtasks using the available tools *one at a time*.  
    
    **Do not fabricate any information.** If you need data, use `searcher_agent` to retrieve it.

    **Tools:** 
    - web_search: Search online for information. Each time you access a table for the first time, always run "SELECT * FROM [table_name] LIMIT 3" WITHOUT ANY conditions in order to understand the table schema.
    - query_database: Query the your travel database for flights, hotels, attractions, and bookings.
    - create_booking: Allows you to write a new booking in the database

    **Goal:** To efficiently and accurately plan a vacation package based on the client's request, utilizing the provided tools and avoiding fabricated data.
"""
#     You are an experienced travel agency executive. Your task is to find a vacation package for your customer.

#     The client will provide you with a request, and you must break it down into small simple subtasks.
#     Then you will apply your tools in order to complete the task, one at a time.

#     Your tools are:
#     - searcher_agent: Allows you to search online and in the database for flights, hotels, and attractions
#     - create_booking: Allows you to write a new booking in the database

#     IMPORTANT: 
#     - Do not make up any data not given to you, if you do not have the data, retrieve it using "searcher_agent".

# """


def prepare_agents(models):
    tools = prepare_function_tools()
    
    searcher_agent = Agent(
        model= models["small_model"],
        name="searcher_agent",
        instructions=SEARCH_PROMPT,
        tools=[tools["web_search"], tools["query_database"]],
    )

    query_session = SQLiteSession("QuerySession")
    
    # booker_agent = Agent(
    #     model= models["small_model"],
    #     name="booker_agent",
    #     instructions=BOOKER_PROMPT,
    #     tools=[tools["create_booking"]],
    # )

    leader_agent = Agent(
        model= models["large_model"],
        name="leader_agent",
        instructions=LEADER_PROMPT,
        tools=list(tools.values()),
        # tools=[
        #     searcher_agent.as_tool(
        #         tool_name="searcher_agent",
        #         tool_description="Can search online and in the database for flights, hotels, and attractions",
        #         session = query_session
        #     ),
        #     tools["create_booking"]
        # ]
    )

    return leader_agent

async def main(question):
    models = create_models()
    leader_agent = prepare_agents(models)
    session = SQLiteSession("QuerySession")
    with langfuse.start_as_current_observation(
        name="OpenAI Vacation Agent",
    ):
        result = await Runner.run(leader_agent, input=question, max_turns=40, session=session)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Result: ", result)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(await Runner.run(leader_agent, input=input(), max_turns=40, session=session))

details = """
    Our names are: Alan, Bella, Chris, and David.        My adress is Alan@domain.com        Book the dates as described in the vacation package.        We will stay in the cheapest hotel you can find for that duration.
    """

if __name__ == "__main__":
    question="""I need to find the *cheapest* vacation package to LA for a family of 4 living in New York, we are flexible with the dates and the destination.
    Book the cheapest vacation package for us, including flights there and back and hotel stay.
    After that add 2 attractions to the vacation package."""
    asyncio.run(main(question))

    