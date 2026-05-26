import os
import operator
from typing import TypedDict, Annotated

import psycopg
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain_groq import ChatGroq

from tools.tavily_tools import tavily_search
from tools.flight_tools import search_flights

load_dotenv()

# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

DATABASE_URL = os.getenv("DATABASE_URL")

# =========================================================
# DESTINATION → AIRPORT MAPPING
# =========================================================

DESTINATION_AIRPORTS = {
    "tungnath": "DED",
    "chopta": "DED",
    "kedarnath": "DED",
    "badrinath": "DED",
    "rishikesh": "DED",
    "dehradun": "DED",
    "manali": "KUU",
    "shimla": "SLV",
    "goa": "GOI",
    "mumbai": "BOM",
    "delhi": "DEL"
}

# =========================================================
# STATE
# =========================================================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

    user_query: str

    destination: str
    nearest_airport: str

    flight_results: str
    hotel_results: str
    itinerary: str

    llm_calls: int

# =========================================================
# DESTINATION AGENT
# =========================================================

def destination_agent(state: TravelState):

    query = state["user_query"].lower()

    destination = "unknown"
    airport = "N/A"

    for place, code in DESTINATION_AIRPORTS.items():
        if place in query:
            destination = place.title()
            airport = code
            break

    return {
        "destination": destination,
        "nearest_airport": airport,
        "messages": [
            AIMessage(
                content=f"Destination identified: {destination}"
            )
        ]
    }

# =========================================================
# FLIGHT AGENT
# =========================================================

def flight_agent(state: TravelState):

    airport_code = state["nearest_airport"]

    # No airport found
    if airport_code == "N/A":

        result = """
No direct airport available.

Recommended:
- Reach nearest railway station
- Continue by road transport
"""

    else:
        try:
            result = search_flights(airport_code)

        except Exception as e:
            result = f"Flight search failed: {str(e)}"

    return {
        "flight_results": result,
        "messages": [
            AIMessage(content="Flight information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =========================================================
# HOTEL AGENT
# =========================================================

def hotel_agent(state: TravelState):

    destination = state["destination"]

    query = f"""
    Best hotels in {destination}
    with price, ratings, and location
    """

    try:
        hotels = tavily_search(query)

    except Exception as e:
        hotels = f"Hotel search failed: {str(e)}"

    return {
        "hotel_results": hotels,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =========================================================
# ITINERARY AGENT
# =========================================================

def itinerary_agent(state: TravelState):

    prompt = f"""
You are an expert AI travel planner.

Create a detailed itinerary.

USER REQUEST:
{state['user_query']}

DESTINATION:
{state['destination']}

NEAREST AIRPORT:
{state['nearest_airport']}

FLIGHT INFORMATION:
{state['flight_results']}

HOTELS:
{state['hotel_results']}

Requirements:
- Create day-wise itinerary
- Mention travel time
- Mention food recommendations
- Mention local sightseeing
- Mention trekking difficulty if applicable
- Keep response clean and structured
"""

    response = llm.invoke([
        SystemMessage(
            content="You are a professional travel planner."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =========================================================
# FINAL AGENT
# =========================================================

def final_agent(state: TravelState):

    prompt = f"""
Generate final polished travel response.

DESTINATION:
{state['destination']}

NEAREST AIRPORT:
{state['nearest_airport']}

FLIGHTS:
{state['flight_results']}

HOTELS:
{state['hotel_results']}

ITINERARY:
{state['itinerary']}

Format:
- Trip Summary
- Flights
- Hotels
- Day-wise itinerary
- Travel tips
- Budget estimate
"""

    response = llm.invoke([
        SystemMessage(
            content="Generate a polished markdown travel plan."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =========================================================
# GRAPH
# =========================================================

graph = StateGraph(TravelState)

graph.add_node("destination_agent", destination_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "destination_agent")

graph.add_edge("destination_agent", "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")

graph.add_edge("final_agent", END)

# =========================================================
# CHECKPOINTER
# =========================================================

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app = graph.compile(
    checkpointer=checkpointer
)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "user_aarohi"
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],

            "user_query": user_input,

            "destination": "",
            "nearest_airport": "",

            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",

            "llm_calls": 0
        },

        config=config
    )

    print("\n================ FINAL RESPONSE ================\n")

    for msg in result["messages"]:
        print(msg.content)

    print("\n================================================")
    print(f"LLM Calls: {result['llm_calls']}")
