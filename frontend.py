import os
import requests
import streamlit as st
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Booking System",
    page_icon="✈️",
    layout="wide"
)

API_URL = "https://multi-agent-travel-plan-langgraph.onrender.com/generate-trip"

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
        value="aditya_singh"
    )

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>✈️ AI Travel Booking System</h1>
        <p>
            Multi-agent travel planning using LangGraph + Groq
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────────────────────────
user_query = st.text_area(
    "Describe your travel plan",
    placeholder="Example: Plan a 5-day Bali trip under ₹1 lakh",
    height=120
)

generate = st.button("🚀 Generate Travel Plan")

# ─────────────────────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────────────────────
if generate:

    if not user_query.strip():
        st.warning("Please enter your travel request.")
        st.stop()

    payload = {
        "user_query": user_query,
        "thread_id": thread_id
    }

    try:

        with st.spinner("Generating AI travel plan..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=300
            )

        if response.status_code != 200:
            st.error(f"API Error: {response.text}")
            st.stop()

        data = response.json()

        # ─────────────────────────────────────────────────────
        # METRICS
        # ─────────────────────────────────────────────────────
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
                    <h2>{data['llm_calls']}</h2>
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

        # ─────────────────────────────────────────────────────
        # FLIGHTS
        # ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("✈️ Flight Information")
        st.markdown(data["flight_results"])

        # ─────────────────────────────────────────────────────
        # HOTELS
        # ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏨 Hotel Information")
        st.markdown(data["hotel_results"])

        # ─────────────────────────────────────────────────────
        # ITINERARY
        # ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🗓️ Itinerary")
        st.markdown(data["itinerary"])

        # ─────────────────────────────────────────────────────
        # FINAL RESPONSE
        # ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🧠 Final Travel Plan")

        st.markdown(
            f"""
            <div class="final-box">
                {data['final_response']}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ─────────────────────────────────────────────────────
        # DOWNLOAD
        # ─────────────────────────────────────────────────────
        st.download_button(
            "⬇️ Download Travel Plan",
            data=data["final_response"],
            file_name=data["saved_file"],
            mime="text/markdown"
        )

    except Exception as e:
        st.error(f"Application Error: {str(e)}")