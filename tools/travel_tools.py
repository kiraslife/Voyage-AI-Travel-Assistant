import json
import os
from datetime import datetime
import requests
from langchain_core.tools import tool

# Path to our data folder
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_json(filename: str) -> list:
    """Helper function to safely load a JSON file from the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []


# =====================================================================
# TOOL 1: FLIGHT SEARCH TOOL
# =====================================================================
@tool
def search_flights(source: str, destination: str, sort_by: str = "cheapest") -> str:
    """
    Searches for flights between a source city and a destination city.
    sort_by can be 'cheapest' (lowest price) or 'fastest' (shortest duration).
    Use this tool to find flight options for the user's trip.
    """
    flights = load_json("flights.json")
    
    # Filter flights matching 'from' and 'to' (case-insensitive)
    matching_flights = [
        f for f in flights
        if f.get("from", "").lower() == source.strip().lower()
        and f.get("to", "").lower() == destination.strip().lower()
    ]
    
    if not matching_flights:
        # Check what cities actually exist to help the agent if a city isn't found
        available_routes = set((f.get("from"), f.get("to")) for f in flights)
        return f"No flights found from {source} to {destination}. Available routes sample: {list(available_routes)[:5]}"

    # Calculate flight duration in hours for each flight using departure and arrival times
    for f in matching_flights:
        try:
            dep = datetime.fromisoformat(f["departure_time"])
            arr = datetime.fromisoformat(f["arrival_time"])
            duration_hours = round((arr - dep).total_seconds() / 3600, 2)
            f["duration_hours"] = duration_hours
        except Exception:
            f["duration_hours"] = 99.0

    # Sort based on user/agent preference (cheapest vs fastest)
    if sort_by.lower() == "fastest":
        matching_flights.sort(key=lambda x: (x["duration_hours"], x["price"]))
    else:
        matching_flights.sort(key=lambda x: (x["price"], x["duration_hours"]))

    # Return the top 3 best options as a JSON string for the LLM to analyze
    return json.dumps({
        "criteria_used": sort_by,
        "total_matches": len(matching_flights),
        "recommended_flights": matching_flights[:3]
    }, indent=2)


# =====================================================================
# TOOL 2: HOTEL RECOMMENDATION TOOL
# =====================================================================
@tool
def search_hotels(city: str, min_stars: int = 3, max_price: int = 10000) -> str:
    """
    Searches for hotels in a destination city filtered by minimum star rating (1-5)
    and maximum price per night. Sorts by highest star rating and lowest price.
    """
    hotels = load_json("hotels.json")
    
    # Step 1: Filter by city first
    city_hotels = [
        h for h in hotels
        if h.get("city", "").lower() == city.strip().lower()
    ]
    
    if not city_hotels:
        available_cities = list(set(h.get("city") for h in hotels))
        return f"No hotels found in {city}. Available cities: {available_cities}"

    # Step 2: Filter by stars and max_price
    filtered_hotels = [
        h for h in city_hotels
        if h.get("stars", 0) >= min_stars and h.get("price_per_night", 999999) <= max_price
    ]
    
    # Fallback: If strict filters return 0 results, return all hotels in that city
    if not filtered_hotels:
        filtered_hotels = city_hotels

    # Sort by highest stars first (-stars), then cheapest price_per_night
    filtered_hotels.sort(key=lambda x: (-x.get("stars", 0), x.get("price_per_night", 0)))

    return json.dumps({
        "city": city,
        "recommended_hotels": filtered_hotels[:3]
    }, indent=2)


# =====================================================================
# TOOL 3: PLACES / POIs DISCOVERY TOOL
# =====================================================================
@tool
def search_places(city: str, place_type: str = "all") -> str:
    """
    Discovers tourist attractions and places to visit in a city.
    place_type is optional (e.g., 'all', 'beach', 'temple', 'fort', 'lake', 'market', 'park').
    Returns top-rated attractions to build a day-wise itinerary.
    """
    places = load_json("places.json")
    
    city_places = [
        p for p in places
        if p.get("city", "").lower() == city.strip().lower()
    ]
    
    if not city_places:
        available_cities = list(set(p.get("city") for p in places))
        return f"No places found in {city}. Available cities: {available_cities}"

    # If the agent requested a specific type (and not 'all'), filter by type
    if place_type.lower() != "all":
        typed_places = [
            p for p in city_places
            if place_type.strip().lower() in p.get("type", "").lower()
        ]
        # Only overwrite if matches exist; otherwise keep all city places
        if typed_places:
            city_places = typed_places

    # Sort by highest rating first
    city_places.sort(key=lambda x: -x.get("rating", 0.0))

    return json.dumps({
        "city": city,
        "top_attractions": city_places[:10]
    }, indent=2)


# =====================================================================
# TOOL 4: LIVE WEATHER LOOKUP TOOL (Open-Meteo Free API)
# =====================================================================
@tool
def get_weather_forecast(city: str, num_days: int = 3) -> str:
    """
    Calls the live Open-Meteo API (no API key required) to get daily weather
    forecasts for a given city for the next num_days (1 to 7 days).
    """
    try:
        # Step 1: Convert City Name to Latitude & Longitude using Open-Meteo Geocoding API
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geo_url, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find geographic coordinates for city: {city}"

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        # Step 2: Fetch Daily Weather Forecast using coordinates
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,weathercode"
            f"&timezone=auto"
        )
        weather_response = requests.get(weather_url, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        daily = weather_data.get("daily", {})
        dates = daily.get("time", [])[:num_days]
        max_temps = daily.get("temperature_2m_max", [])[:num_days]
        min_temps = daily.get("temperature_2m_min", [])[:num_days]
        codes = daily.get("weathercode", [])[:num_days]

        # WMO Weather interpretation codes dictionary
        wmo_codes = {
            0: "Sunny / Clear Sky",
            1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
            45: "Foggy", 48: "Rime Fog",
            51: "Light Drizzle", 53: "Moderate Drizzle", 55: "Dense Drizzle",
            61: "Slight Rain", 63: "Moderate Rain", 65: "Heavy Rain",
            80: "Light Rain Showers", 81: "Moderate Rain Showers", 82: "Violent Rain Showers",
            95: "Thunderstorm"
        }

        forecast_list = []
        for i in range(len(dates)):
            condition = wmo_codes.get(codes[i], "Pleasant / Partly Cloudy")
            forecast_list.append({
                "day": f"Day {i + 1} ({dates[i]})",
                "condition": condition,
                "max_temp_c": max_temps[i],
                "min_temp_c": min_temps[i]
            })

        return json.dumps({
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "daily_forecast": forecast_list
        }, indent=2)

    except requests.exceptions.RequestException as e:
        return f"Weather API network error: {e}"
    except Exception as e:
        return f"Error fetching weather data: {e}"


# =====================================================================
# TOOL 5: BUDGET ESTIMATION TOOL
# =====================================================================
@tool
def calculate_trip_budget(
    flight_price: int,
    hotel_price_per_night: int,
    num_days: int,
    daily_local_expense: int = 1000
) -> str:
    """
    Calculates the total estimated trip budget including flight cost,
    hotel cost (price_per_night * num_days), and local food/travel expenses.
    """
    try:
        # Calculate number of hotel nights (typically num_days or num_days - 1; project example uses 2 nights for 3 days or per night)
        # Note: In the project doc example, Goa 3-day trip has ₹3200/night -> ₹6400 (2 nights) or num_days.
        # Let's compute both or standard nights = max(1, num_days - 1) and also show full days so it's crystal clear.
        nights = max(1, num_days - 1)
        hotel_total = hotel_price_per_night * nights
        food_and_local_total = daily_local_expense * num_days
        total_cost = flight_price + hotel_total + food_and_local_total

        return json.dumps({
            "flight_cost_inr": flight_price,
            "hotel_nights": nights,
            "hotel_price_per_night_inr": hotel_price_per_night,
            "hotel_total_cost_inr": hotel_total,
            "food_and_local_travel_inr": food_and_local_total,
            "estimated_total_budget_inr": total_cost
        }, indent=2)
    except Exception as e:
        return f"Error calculating budget: {e}"


# Export list of all tools so the Agent can import them easily
ALL_TRAVEL_TOOLS = [
    search_flights,
    search_hotels,
    search_places,
    get_weather_forecast,
    calculate_trip_budget
]