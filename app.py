import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from main import app as langgraph_app
from fastapi.middleware.cors import CORSMiddleware



# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Travel Booking System",
    description="Multi-Agent AI Travel Planner using LangGraph",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# ENV VALIDATION
# ─────────────────────────────────────────────────────────────
required_env = [
    "DATABASE_URL",
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
    "AVIATIONSTACK_API_KEY"
]

missing = [key for key in required_env if not os.getenv(key)]

if missing:
    raise Exception(
        f"Missing environment variables: {', '.join(missing)}"
    )

# ─────────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────────
class TravelRequest(BaseModel):
    user_query: str
    thread_id: str = "default_user"


# ─────────────────────────────────────────────────────────────
# ROOT ROUTE
# ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "✈️ AI Travel Booking API Running",
        "status": "success"
    }


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ─────────────────────────────────────────────────────────────
# GENERATE TRAVEL PLAN
# ─────────────────────────────────────────────────────────────
@app.post("/generate-trip")
async def generate_trip(request: TravelRequest):

    if not request.user_query.strip():
        raise HTTPException(
            status_code=400,
            detail="User query cannot be empty"
        )

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    collected = {
        "flight_results": "",
        "hotel_results": "",
        "itinerary": "",
        "final_response": "",
        "llm_calls": 0
    }

    try:

        for chunk in langgraph_app.stream(
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
            config=config,
            stream_mode="updates"
        ):

            for node_name, state_update in chunk.items():

                if node_name == "flight_agent":

                    collected["flight_results"] = state_update.get(
                        "flight_results",
                        ""
                    )

                elif node_name == "hotel_agent":

                    collected["hotel_results"] = state_update.get(
                        "hotel_results",
                        ""
                    )

                elif node_name == "itinerary_agent":

                    collected["itinerary"] = state_update.get(
                        "itinerary",
                        ""
                    )

                elif node_name == "final_agent":

                    msgs = state_update.get("messages", [])

                    if msgs:
                        collected["final_response"] = msgs[-1].content

                collected["llm_calls"] = state_update.get(
                    "llm_calls",
                    collected["llm_calls"]
                )

        # ─────────────────────────────────────────────────────
        # SAVE MARKDOWN FILE
        # ─────────────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"travel_plan_{timestamp}.md"

        save_dir = os.path.join(
            os.path.dirname(__file__),
            "travel_plans"
        )

        os.makedirs(save_dir, exist_ok=True)

        file_content = f"""
# Travel Plan

## Query
{request.user_query}

## Generated
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## User ID
{request.thread_id}

---

# Flight Information

{collected['flight_results']}

---

# Hotel Information

{collected['hotel_results']}

---

# Itinerary

{collected['itinerary']}

---

# Final Travel Plan

{collected['final_response']}

---

LLM Calls: {collected['llm_calls']}
"""

        filepath = os.path.join(save_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(file_content)

        return JSONResponse(
            content={
                "status": "success",
                "query": request.user_query,
                "thread_id": request.thread_id,
                "flight_results": collected["flight_results"],
                "hotel_results": collected["hotel_results"],
                "itinerary": collected["itinerary"],
                "final_response": collected["final_response"],
                "llm_calls": collected["llm_calls"],
                "saved_file": filename
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ─────────────────────────────────────────────────────────────
# DOWNLOAD GENERATED FILE
# ─────────────────────────────────────────────────────────────
@app.get("/download/{filename}")
async def download_file(filename: str):

    file_path = os.path.join(
        os.path.dirname(__file__),
        "travel_plans",
        filename
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/markdown"
    )


# ─────────────────────────────────────────────────────────────
#───────────────────────────────