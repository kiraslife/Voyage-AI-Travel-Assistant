import json
import re
from typing import Any, Dict, List, Optional, Tuple
import requests
from agent.travel_agent import _discover_keys


KNOWN_CITY_COORDS = {
    "darjeeling": (27.0410, 88.2663),
    "kolkata": (22.5726, 88.3639),
    "goa": (15.2993, 74.1240),
    "delhi": (28.6139, 77.2090),
    "new delhi": (28.6139, 77.2090),
    "jaipur": (26.9124, 75.7873),
    "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "bengaluru": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "chennai": (13.0827, 80.2707),
    "manali": (32.2432, 77.1892),
    "shimla": (31.1048, 77.1734),
    "gangtok": (27.3389, 88.6065),
    "leh": (34.1526, 77.5771),
    "ladakh": (34.1526, 77.5771),
    "srinagar": (34.0837, 74.7973),
    "udaipur": (24.5854, 73.7125),
    "varanasi": (25.3176, 82.9739),
    "rishikesh": (30.0869, 78.2676),
    "ooty": (11.4102, 76.6950),
    "munnar": (10.0889, 77.0595),
    "kochi": (9.9312, 76.2673),
    "kerala": (9.9312, 76.2673),
    "agra": (27.1767, 78.0081),
    "amritsar": (31.6340, 74.8723),
    "shillong": (25.5788, 91.8933),
    "pondicherry": (11.9416, 79.8083),
    "durgapur": (23.5204, 87.3119),
    "pune": (18.5204, 73.8567),
    "coorg": (12.3375, 75.8069),
    "nainital": (29.3919, 79.4542),
    "mussoorie": (30.4598, 78.0644),
}

HERO_THEMES = {
    "mountain": "https://images.unsplash.com/photo-1544735716-392fe2489ffa?auto=format&fit=crop&w=1200&q=80",
    "beach": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=1200&q=80",
    "heritage": "https://images.unsplash.com/photo-1524492412937-b28074a5d7da?auto=format&fit=crop&w=1200&q=80",
    "city": "https://images.unsplash.com/photo-1570168007204-dfb528c6958f?auto=format&fit=crop&w=1200&q=80",
}


def pick_hero_image(destination: str, summary: str = "") -> str:
    txt = f"{destination} {summary}".lower()
    if any(k in txt for k in ["darjeeling", "manali", "shimla", "gangtok", "leh", "ladakh", "srinagar", "ooty", "munnar", "shillong", "nainital", "mussoorie", "hill", "mountain", "himalaya", "tea", "snow"]):
        return HERO_THEMES["mountain"]
    if any(k in txt for k in ["goa", "beach", "island", "andaman", "pondicherry", "coastal", "kerala", "kochi", "gokarna", "varkala"]):
        return HERO_THEMES["beach"]
    if any(k in txt for k in ["kolkata", "jaipur", "udaipur", "delhi", "agra", "varanasi", "amritsar", "fort", "palace", "heritage", "temple"]):
        return HERO_THEMES["heritage"]
    return HERO_THEMES["city"]


def geocode_city(city: str) -> Tuple[float, float]:
    """Resolves accurate (lat, lon) using verified city table first, then Open-Meteo Geocoding."""
    key = city.strip().lower()
    if key in KNOWN_CITY_COORDS:
        return KNOWN_CITY_COORDS[key]
    for k, coords in KNOWN_CITY_COORDS.items():
        if k in key:
            return coords
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city.strip(), "count": 5, "language": "en", "format": "json"},
            timeout=6,
        )
        if r.status_code == 200:
            results = r.json().get("results") or []
            in_matches = [x for x in results if str(x.get("country_code", "")).upper() == "IN"]
            best = in_matches[0] if in_matches else (results[0] if results else None)
            if best:
                return float(best["latitude"]), float(best["longitude"])
    except Exception:
        pass
    return 22.5726, 88.3639


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            candidate = re.sub(r",\s*([}\]])", r"\1", m.group(0))
            try:
                return json.loads(candidate)
            except Exception:
                return None
    return None


def _discover_groq_models(api_key: str) -> List[str]:
    """Fetches currently active chat models from the user's Groq account."""
    preferred = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]
    blocked = ("whisper", "guard", "orpheus", "tts", "embed", "safeguard", "compound")
    live_models: List[str] = []
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=6,
        )
        if resp.status_code == 200:
            for item in resp.json().get("data", []):
                mid = str(item.get("id", "")).strip()
                if mid and not any(b in mid.lower() for b in blocked):
                    live_models.append(mid)
    except Exception:
        pass

    if not live_models:
        return preferred

    ordered = [m for m in preferred if m in live_models]
    for m in live_models:
        if m not in ordered:
            ordered.append(m)
    return ordered[:6]


def _call_llm_json(system_prompt: str, user_prompt: str) -> Tuple[Dict[str, Any], str, str]:
    """Calls live Groq models -> Gemini (thinking disabled for clean JSON) -> OpenAI."""
    keys = _discover_keys()
    errors: List[str] = []

    # 1. Groq (dynamically discovered active models)
    if keys.get("Groq"):
        headers = {"Authorization": f"Bearer {keys['Groq']}", "Content-Type": "application/json"}
        for model in _discover_groq_models(keys["Groq"]):
            try:
                payload: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.25,
                    "max_completion_tokens": 4096,
                    "response_format": {"type": "json_object"},
                }
                if "gpt-oss" in model.lower():
                    payload["reasoning_effort"] = "low"

                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=20,
                )
                # Retry without response_format if a specific model rejects json_object
                if resp.status_code == 400 and "response_format" in payload:
                    payload.pop("response_format", None)
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=20,
                    )
                if resp.status_code == 200:
                    raw = resp.json()["choices"][0]["message"].get("content", "")
                    parsed = _extract_json_block(raw)
                    if parsed:
                        return parsed, "Groq", model
                    errors.append(f"Groq/{model}: invalid JSON ({len(raw)} chars)")
                else:
                    errors.append(f"Groq/{model}: {resp.status_code}")
            except Exception as e:
                errors.append(f"Groq/{model}: {e}")

    # 2. Gemini (disable thinking budget so full 4096 tokens go to valid JSON)
    if keys.get("Gemini"):
        for g_model in ["gemini-2.5-flash", "gemini-2.0-flash"]:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={keys['Gemini']}"
                gen_cfg: Dict[str, Any] = {
                    "temperature": 0.25,
                    "maxOutputTokens": 4096,
                    "responseMimeType": "application/json",
                }
                if "2.5" in g_model:
                    gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}

                resp = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser request: {user_prompt}"}]}],
                        "generationConfig": gen_cfg,
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    parts = resp.json()["candidates"][0]["content"].get("parts", [])
                    txt = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    parsed = _extract_json_block(txt)
                    if parsed:
                        return parsed, "Gemini", g_model
                    errors.append(f"Gemini/{g_model}: invalid JSON ({len(txt)} chars)")
                else:
                    errors.append(f"Gemini/{g_model}: {resp.status_code}")
            except Exception as e:
                errors.append(f"Gemini/{g_model}: {e}")

    # 3. OpenAI fallback
    if keys.get("OpenAI"):
        for o_model in ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o"]:
            try:
                resp = requests.post(
                    "https://api.openai.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {keys['OpenAI']}", "Content-Type": "application/json"},
                    json={
                        "model": o_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.25,
                        "max_tokens": 4096,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    raw = resp.json()["choices"][0]["message"].get("content", "")
                    parsed = _extract_json_block(raw)
                    if parsed:
                        return parsed, "OpenAI", o_model
                    errors.append(f"OpenAI/{o_model}: invalid JSON")
                else:
                    errors.append(f"OpenAI/{o_model}: {resp.status_code}")
            except Exception as e:
                errors.append(f"OpenAI/{o_model}: {e}")

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
        "You are Voyage AI, a travel planning engine. Return ONLY a single valid JSON object (no markdown) "
        "with real, famous, accurate landmarks, hotels, restaurants, and day-by-day plans for the searched destination. "
        "NEVER use generic placeholder names like 'Famous Fort', 'Historic Temple', 'Scenic Museum', or 'Popular Lake'. "
        "Required JSON schema:\n"
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
    data["hero_image"] = pick_hero_image(dest, str(data.get("summary", "")))
    data["lat"] = lat
    data["lon"] = lon
    data["attractions"] = attractions
    data["day_plans"] = day_plans
    data["full_itinerary"] = "\n".join(md_lines)
    data["_provider"] = provider
    data["_model"] = model
    return data
