# backend.py

import os
from typing import TypedDict, Annotated
import operator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

# ---------------------------------------------------
# ENV
# ---------------------------------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL missing")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing")

# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------

api = FastAPI(
    title="AI Travel Booking API",
    version="1.0.0"
)

# ---------------------------------------------------
# LLM
# ---------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# ---------------------------------------------------
# STATE
# ---------------------------------------------------

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

# ---------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------

class TravelRequest(BaseModel):
    user_query: str
    thread_id: str = "default_user"

# ---------------------------------------------------
# AGENTS
# ---------------------------------------------------

def flight_agent(state: TravelState):

    query = state["user_query"]

    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def hotel_agent(state: TravelState):

    query = f"Best hotels for {state['user_query']}"

    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def itinerary_agent(state: TravelState):

    prompt = f"""
    Create a detailed travel itinerary.

    USER QUERY:
    {state['user_query']}

    FLIGHT RESULTS:
    {state['flight_results']}

    HOTEL RESULTS:
    {state['hotel_results']}
    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert AI travel planner."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def final_agent(state: TravelState):

    final_prompt = f"""
    Create the final polished travel plan.

    FLIGHTS:
    {state['flight_results']}

    HOTELS:
    {state['hotel_results']}

    ITINERARY:
    {state['itinerary']}
    """

    response = llm.invoke([
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# ---------------------------------------------------
# GRAPH
# ---------------------------------------------------

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

# ---------------------------------------------------
# POSTGRES CHECKPOINTER
# ---------------------------------------------------

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True
)

checkpointer = PostgresSaver(_conn)

checkpointer.setup()

app_graph = graph.compile(
    checkpointer=checkpointer
)

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------

@api.get("/")
def health_check():

    return {
        "status": "running",
        "service": "AI Travel Booking API"
    }


@api.post("/travel")
def generate_travel_plan(request: TravelRequest):

    try:

        config = {
            "configurable": {
                "thread_id": request.thread_id
            }
        }

        result = app_graph.invoke(
            {
                "messages": [
                    HumanMessage(content=request.user_query)
                ],
                "user_query": request.user_query,
                "flight_results": "",
                "hotel_results": "",
                "itinerary": "",
                "llm_calls": 0
            },
            config=config
        )

        final_response = ""

        if result.get("messages"):
            final_response = result["messages"][-1].content

        return {
            "success": True,
            "user_query": request.user_query,
            "flight_results": result.get("flight_results", ""),
            "hotel_results": result.get("hotel_results", ""),
            "itinerary": result.get("itinerary", ""),
            "final_response": final_response,
            "llm_calls": result.get("llm_calls", 0)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )