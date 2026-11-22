import json
import re
from typing import Any, Dict, List, Optional, Tuple
import requests
from agent.travel_agent import _discover_keys


def geocode_city(city: str) -> Tuple[float, float]:
    """Resolves real (lat, lon) for any searched city via free Open-Meteo Geocoding API."""
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city.strip(), "count": 1, "language": "en", "format": "json"},
            timeout=6,
        )
        if r.status_code == 200:
            results = r.json().get("results") or []
            if results:
                return float(results[0]["latitude"]), float(results[0]["longitude"])
    except Exception:
        pass
    return 22.5726, 88.3639


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _call_llm_json(system_prompt: str, user_prompt: str) -> Tuple[Dict[str, Any], str, str]:
    """Calls Groq -> OpenAI -> Gemini for structured JSON trip output."""
    keys = _discover_keys()
    errors: List[str] = []

    if keys.get("Groq"):
        for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {keys['Groq']}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.3,
                        "max_completion_tokens": 2200,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=18,
                )
                if resp.status_code == 200:
                    parsed = _extract_json_block(resp.json()["choices"][0]["message"].get("content", ""))
                    if parsed:
                        return parsed, "Groq", model
                errors.append(f"Groq/{model}: {resp.status_code}")
            except Exception as e:
                errors.append(f"Groq/{model}: {e}")

    if keys.get("OpenAI"):
        try:
            resp = requests.post(
                "https://api.openai.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {keys['OpenAI']}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2200,
                    "response_format": {"type": "json_object"},
                },
                timeout=18,
            )
            if resp.status_code == 200:
                parsed = _extract_json_block(resp.json()["choices"][0]["message"].get("content", ""))
                if parsed:
                    return parsed, "OpenAI", "gpt-4o-mini"
            errors.append(f"OpenAI: {resp.status_code}")
        except Exception as e:
            errors.append(f"OpenAI: {e}")

    if keys.get("Gemini"):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={keys['Gemini']}"
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser request: {user_prompt}"}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2200, "responseMimeType": "application/json"},
                },
                timeout=18,
            )
            if resp.status_code == 200:
                txt = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                parsed = _extract_json_block(txt)
                if parsed:
                    return parsed, "Gemini", "gemini-2.5-flash"
            errors.append(f"Gemini: {resp.status_code}")
        except Exception as e:
            errors.append(f"Gemini: {e}")

    raise RuntimeError(" | ".join(errors) or "No API key configured")


def generate_dynamic_trip(
    search_query: str,
    default_source: str = "Delhi",
    default_days: int = 4,
    travelers: int = 2,
    max_hotel_price: int = 6000,
) -> Dict[str, Any]:
    """Generates a real, destination-specific trip object for any search query."""
    system_prompt = (
        "You are Voyage AI, a travel planning engine. Return ONLY valid JSON with real, famous, accurate landmarks, "
        "hotels, restaurants, and day-by-day plans for the user's searched destination. "
        "NEVER use generic placeholder names like 'Famous Fort', 'Historic Temple', 'Scenic Museum', or 'Popular Lake'. "
        "JSON schema:\n"
        "{\n"
        '  "title": "Short trip title",\n'
        '  "source": "Origin city",\n'
        '  "destination": "Searched destination city",\n'
        '  "days": 4,\n'
        '  "travelers": 2,\n'
        '  "summary": "2 concise sentences tailored to what the user searched.",\n'
        '  "ai_insight": "1-2 sentence insider travel tip specific to this destination.",\n'
        '  "flight": {"airline": "IndiGo", "flight_id": "6E-204", "price": 5200, "duration": "2h 15m"},\n'
        '  "hotel": {"name": "Real well-known hotel in destination", "price_per_night": 4200, "stars": 4, "rating": 4.5, "area": "Central area"},\n'
        '  "budget": {"flights": 10400, "hotels": 16800, "food": 8000, "transport": 3200, "total": 38400},\n'
        '  "attractions": [\n'
        '    {"name": "Real Landmark", "category": "Heritage", "rating": 4.8, "description": "Short 1-line highlight", "lat_offset": 0.01, "lon_offset": -0.01}\n'
        '  ],\n'
        '  "day_plans": [\n'
        '    {\n'
        '      "day": 1,\n'
        '      "theme": "Day 1 Theme",\n'
        '      "items": [\n'
        '        {"time": "09:00 AM", "activity": "Real Place/Activity", "category": "Sightseeing", "note": "Short tip"},\n'
        '        {"time": "01:00 PM", "activity": "Real Local Restaurant/Dish", "category": "Food", "note": "What to order"},\n'
        '        {"time": "05:00 PM", "activity": "Real Evening/Sunset Spot", "category": "Leisure", "note": "Best view"}\n'
        '      ]\n'
        '    }\n'
        '  ],\n'
        '  "packing_essentials": ["Destination-specific item 1", "Item 2", "Item 3"]\n'
        "}\n"
        "Include exactly 5 real attractions and a complete day_plans entry for every single day (Day 1 through Day N)."
    )

    user_prompt = (
        f"User query: {search_query}\n"
        f"Defaults (override if specified in query): source={default_source}, days={default_days}, "
        f"travelers={travelers}, max_hotel_price_inr={max_hotel_price}."
    )

    data, provider, model = _call_llm_json(system_prompt, user_prompt)
    dest = str(data.get("destination") or search_query).strip().title()
    lat, lon = geocode_city(dest)

    attractions = []
    for idx, a in enumerate((data.get("attractions") or [])[:5]):
        d_lat = float(a.get("lat_offset", (idx - 2) * 0.012))
        d_lon = float(a.get("lon_offset", ((idx % 3) - 1) * 0.015))
        p_lat = round(lat + max(-0.06, min(0.06, d_lat)), 5)
        p_lon = round(lon + max(-0.06, min(0.06, d_lon)), 5)
        attractions.append({
            "name": a.get("name", f"{dest} Highlight {idx + 1}"),
            "city": dest,
            "category": a.get("category", "Landmark"),
            "type": a.get("category", "Landmark"),
            "rating": float(a.get("rating", 4.6)),
            "description": a.get("description", ""),
            "lat": p_lat,
            "lon": p_lon,
            "latitude": p_lat,
            "longitude": p_lon,
        })

    day_plans = data.get("day_plans") or []
    md_lines = [f"### {data.get('title', dest + ' Itinerary')}", f"{data.get('summary', '')}\n"]
    for dp in day_plans:
        md_lines.append(f"**Day {dp.get('day', '')}: {dp.get('theme', '')}**")
        for it in dp.get("items") or []:
            md_lines.append(f"- **{it.get('time', '')}** — {it.get('activity', '')} *({it.get('note', '')})*")
        md_lines.append("")

    data["destination"] = dest
    data["lat"] = lat
    data["lon"] = lon
    data["attractions"] = attractions
    data["day_plans"] = day_plans
    data["full_itinerary"] = "\n".join(md_lines)
    data["_provider"] = provider
    data["_model"] = model
    return data
