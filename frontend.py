# Corrected `app.py` (Streamlit Frontend)


import os
import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage
from main import app

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Booking System",
    page_icon="✈️",
    layout="wide"
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
    st.error(f"Missing environment variables: {', '.join(missing)}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: #080d14;
        color: white;
        font-family: Inter, sans-serif;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 3rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #1e40af;
    }

    .hero h1 {
        color: white;
        font-size: 3rem;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.1rem;
    }

    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }

    .final-box {
        background: #0f172a;
        border-left: 5px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 12px;
        color: #dbeafe;
        line-height: 1.8;
    }

    .stTextArea textarea {
        background: #0f172a !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid #1e293b !important;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        height: 3.2rem;
        font-weight: bold;
        width: 100%;
    }

    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 AI Travel Planner")

    thread_id = st.text_input(
        "User Session ID",
        value="user_aarohi"
    )

    st.markdown("---")

    st.subheader("⚙️ Tech Stack")

    techs = [
        "LangGraph",
        "Groq Llama 3.3 70B",
        "PostgreSQL",
        "Tavily Search",
        "AviationStack"
    ]

    for tech in techs:
        st.markdown(f"- {tech}")

    st.markdown("---")

    st.subheader("🤖 Agent Workflow")

    workflow = [
        "Flight Agent",
        "Hotel Agent",
        "Itinerary Agent",
        "Final Agent"
    ]

    for step in workflow:
        st.markdown(f"✅ {step}")

# ─────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>✈️ AI Travel Booking System</h1>
        <p>
            Multi-agent travel planning using LangGraph, Groq, Tavily,
            PostgreSQL memory and live agent orchestration.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# QUICK PROMPTS
# ─────────────────────────────────────────────────────────────
quick_prompts = [
    "Plan a 5 day Dubai trip",
    "Tokyo trip under ₹2 lakhs",
    "Bali backpacking itinerary",
    "Paris honeymoon trip"
]

cols = st.columns(len(quick_prompts))
selected_prompt = ""

for col, prompt in zip(cols, quick_prompts):
    with col:
        if st.button(prompt):
            selected_prompt = prompt

# ─────────────────────────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────────────────────────
user_query = st.text_area(
    "Describe your travel plan",
    value=selected_prompt,
    placeholder="Example: Plan a complete 7-day Japan trip including flights, hotels and sightseeing under ₹2 lakhs",
    height=120
)

# ─────────────────────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────────────────────
generate = st.button("🚀 Generate Travel Plan")

# ─────────────────────────────────────────────────────────────
# AGENT LABELS
# ─────────────────────────────────────────────────────────────
AGENT_META = {
    "flight_agent": "✈️ Flight Agent",
    "hotel_agent": "🏨 Hotel Agent",
    "itinerary_agent": "🗓️ Itinerary Agent",
    "final_agent": "🧠 Final Agent"
}

# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────
if generate:

    if not user_query.strip():
        st.warning("Please enter your travel request.")
        st.stop()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    collected = {
        "flight_results": "",
        "hotel_results": "",
        "itinerary": "",
        "final_response": "",
        "llm_calls": 0
    }

    st.markdown("---")
    st.subheader("🤖 Live Agent Execution")

    try:

        with st.spinner("Agents are planning your trip..."):

            for chunk in app.stream(
                {
                    "messages": [HumanMessage(content=user_query)],
                    "user_query": user_query,
                    "flight_results": "",
                    "hotel_results": "",
                    "itinerary": "",
                    "llm_calls": 0
                },
                config=config,
                stream_mode="updates"
            ):

                for node_name, state_update in chunk.items():

                    label = AGENT_META.get(node_name, node_name)

                    with st.status(label, expanded=True):

                        if node_name == "flight_agent":

                            text = state_update.get(
                                "flight_results",
                                "No flight data found"
                            )

                            collected["flight_results"] = text

                            st.markdown(text)

                        elif node_name == "hotel_agent":

                            text = state_update.get(
                                "hotel_results",
                                "No hotel data found"
                            )

                            collected["hotel_results"] = text

                            st.markdown(text)

                        elif node_name == "itinerary_agent":

                            text = state_update.get(
                                "itinerary",
                                "No itinerary generated"
                            )

                            collected["itinerary"] = text

                            st.markdown(text)

                        elif node_name == "final_agent":

                            msgs = state_update.get("messages", [])

                            if msgs:
                                text = msgs[-1].content
                            else:
                                text = "No final response generated"

                            collected["final_response"] = text

                            st.markdown(text)

                        collected["llm_calls"] = state_update.get(
                            "llm_calls",
                            collected["llm_calls"]
                        )

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.stop()

    # ─────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <h2>4</h2>
                <p>Agents Executed</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h2>{collected['llm_calls']}</h2>
                <p>LLM Calls</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <h2>✅</h2>
                <p>Status</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ─────────────────────────────────────────────────────────
    # FINAL RESPONSE
    # ─────────────────────────────────────────────────────────
    if collected["final_response"]:

        st.markdown("---")
        st.subheader("🧠 Final Travel Plan")

        st.markdown(
            f"""
            <div class="final-box">
                {collected['final_response']}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ─────────────────────────────────────────────────────────
    # SAVE FILE
    # ─────────────────────────────────────────────────────────
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
{user_query}

## Generated
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## User ID
{thread_id}

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

    with open(
        os.path.join(save_dir, filename),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(file_content)

    st.download_button(
        "⬇️ Download Travel Plan",
        data=file_content,
        file_name=filename,
        mime="text/markdown"
    )

    st.success(
        f"Travel plan saved successfully in travel_plans/{filename}"
    )
