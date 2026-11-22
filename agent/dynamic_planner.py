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
    data["lat"] = lat
    data["lon"] = lon
    data["attractions"] = attractions
    data["day_plans"] = day_plans
    data["full_itinerary"] = "\n".join(md_lines)
    data["_provider"] = provider
    data["_model"] = model
    return data
