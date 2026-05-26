import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

BASE_URL = "http://api.aviationstack.com/v1/flights"

# ======================================================
# DESTINATION → AIRPORT MAP
# ======================================================

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

# ======================================================
# SEARCH FLIGHTS
# ======================================================

def search_flights(query):

    query = query.lower()

    arrival_airport = None

    # Find matching airport
    for place, airport_code in DESTINATION_AIRPORTS.items():

        if place in query:
            arrival_airport = airport_code
            break

    # No airport found
    if not arrival_airport:

        return """
No nearby airport found.

Recommended:
- Use train or road transport
"""

    # Special handling for trekking destinations
    if arrival_airport == "DED":

        return f"""
Nearest Airport:
Jolly Grant Airport, Dehradun (DED)

Recommended Route:
Flight → Dehradun
Road → Chopta/Tungnath (6–7 hrs)

Nearest Railway Stations:
- Haridwar
- Rishikesh

Best Transport:
Taxi or shared cab from Dehradun/Rishikesh
"""

    params = {
        "access_key": API_KEY,
        "arr_iata": arrival_airport,
        "limit": 5
    }

    try:

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=10
        )

        data = response.json()

        flights = []

        if "data" not in data:
            return "No flight data available."

        for flight in data["data"][:5]:

            airline = (
                flight.get("airline", {})
                .get("name", "Unknown")
            )

            departure = (
                flight.get("departure", {})
                .get("airport", "Unknown")
            )

            arrival = (
                flight.get("arrival", {})
                .get("airport", "Unknown")
            )

            status = flight.get(
                "flight_status",
                "Unknown"
            )

            flights.append(
                f"""
Airline: {airline}
Departure: {departure}
Arrival: {arrival}
Status: {status}
"""
            )

        return "\n".join(flights)

    except Exception as e:

        return f"Flight API Error: {str(e)}"