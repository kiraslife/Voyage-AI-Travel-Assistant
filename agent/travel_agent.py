# agent/travel_agent.py
import os
import json
import logging
import requests
from enum import Enum
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger("VoyageAI.Agent")


class LLMErrorCategory(str, Enum):
    DAILY_QUOTA_EXCEEDED = "Daily Request Quota Exceeded"
    RPM_RATE_LIMIT = "Requests-Per-Minute Limit Exceeded"
    TOKEN_LIMIT_EXCEEDED = "Input/Output Token Limit Exceeded"
    INVALID_API_KEY = "Invalid or Missing API Key"
    MODEL_UNAVAILABLE = "Model Unavailable or Overloaded"
    NETWORK_ERROR = "Network Connectivity Error"
    PROVIDER_ERROR = "Provider Execution Error"


def _discover_keys() -> Dict[str, str]:
    """Finds Gemini, Groq, and OpenAI keys by variable name OR key prefix (gsk_, sk-, AIza)."""
    load_dotenv(override=True)
    keys = {"Gemini": "", "Groq": "", "OpenAI": ""}
    try:
        import streamlit as st
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass

    # 1. Check common variable names (including spelling variations)
    for k in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_KEY"]:
        v = os.getenv(k, "").strip().strip('"').strip("'")
        if len(v) > 10 and not v.startswith("your_"):
            keys["Gemini"] = v
            break

    for k in ["GROQ_API_KEY", "GROUQ_API_KEY", "GROK_API_KEY", "GROQ_KEY"]:
        v = os.getenv(k, "").strip().strip('"').strip("'")
        if len(v) > 10 and not v.startswith("your_"):
            keys["Groq"] = v
            break

    for k in ["OPENAI_API_KEY", "CHATGPT_API_KEY", "OPENAI_KEY", "CHATGPT_KEY"]:
        v = os.getenv(k, "").strip().strip('"').strip("'")
        if len(v) > 10 and not v.startswith("your_"):
            keys["OpenAI"] = v
            break

    # 2. Scan all .env variables by prefix so no key is ever missed
    for env_k, env_v in os.environ.items():
        val = env_v.strip().strip('"').strip("'")
        if val.startswith("gsk_") and not keys["Groq"]:
            keys["Groq"] = val
        elif val.startswith("sk-") and not keys["OpenAI"]:
            keys["OpenAI"] = val
        elif val.startswith("AIza") and not keys["Gemini"]:
            keys["Gemini"] = val

    return keys


def classify_llm_exception(exc: Exception) -> Dict[str, Any]:
    raw_msg = str(exc)
    lower_msg = raw_msg.lower()

    if "429" in lower_msg or "resource_exhausted" in lower_msg or "rate_limit" in lower_msg or "quota" in lower_msg:
        return {
            "category": LLMErrorCategory.DAILY_QUOTA_EXCEEDED.value,
            "is_retryable": False,
            "retry_after": None,
            "suggested_action": "Automatic failover to secondary provider triggered.",
            "raw_detail": raw_msg,
        }
    if any(k in lower_msg for k in ["api_key", "apikey", "401", "403", "unauthorized", "invalid_api_key"]):
        return {
            "category": LLMErrorCategory.INVALID_API_KEY.value,
            "is_retryable": False,
            "retry_after": None,
            "suggested_action": "Verify the API key in your .env file.",
            "raw_detail": raw_msg,
        }
    return {
        "category": LLMErrorCategory.MODEL_UNAVAILABLE.value,
        "is_retryable": False,
        "retry_after": None,
        "suggested_action": "Switched to next available model.",
        "raw_detail": raw_msg,
    }


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    keys = _discover_keys()
    return {
        "Groq": {
            "configured": bool(keys["Groq"]),
            "installed": True,
            "model": "openai/gpt-oss-20b",
            "env_var": "GROQ_API_KEY",
        },
        "OpenAI": {
            "configured": bool(keys["OpenAI"]),
            "installed": True,
            "model": "gpt-4o-mini",
            "env_var": "OPENAI_API_KEY",
        },
        "Gemini": {
            "configured": bool(keys["Gemini"]),
            "installed": True,
            "model": "gemini-2.5-flash",
            "env_var": "GOOGLE_API_KEY",
        },
    }


def _gather_local_dataset_context(user_query: str) -> tuple[str, List[Dict[str, Any]]]:
    steps: List[Dict[str, Any]] = []
    q_lower = user_query.lower()
    cities = ["goa", "delhi", "jaipur", "mumbai", "bangalore", "kolkata", "hyderabad", "chennai"]
    matched = [c.title() for c in cities if c in q_lower]
    dest = matched[-1] if matched else "Goa"
    src = matched[0] if len(matched) > 1 else "Delhi"

    def _read_json(fname: str) -> list:
        path = os.path.join("data", fname)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    return d if isinstance(d, list) else []
            except Exception:
                return []
        return []

    flights = [f for f in _read_json("flights.json") if str(f.get("to", f.get("destination", ""))).lower() == dest.lower()][:3]
    hotels = [h for h in _read_json("hotels.json") if str(h.get("city", h.get("location", ""))).lower() == dest.lower()][:3]
    places = [p for p in _read_json("places.json") if str(p.get("city", p.get("location", ""))).lower() == dest.lower()][:5]

    steps.append({"order": 1, "tool": "search_flights", "tool_input": {"source": src, "destination": dest}, "status": "Success", "observation": flights})
    steps.append({"order": 2, "tool": "search_hotels", "tool_input": {"city": dest}, "status": "Success", "observation": hotels})
    steps.append({"order": 3, "tool": "discover_places", "tool_input": {"city": dest}, "status": "Success", "observation": places})

    ctx = f"Dataset Context ({src} -> {dest}): Flights={json.dumps(flights)} | Hotels={json.dumps(hotels)} | Places={json.dumps(places)}"
    return ctx, steps


def _try_groq(api_key: str, system_prompt: str, user_query: str, steps: list) -> Dict[str, Any]:
    """Tries current Groq production models and dynamically queries /v1/models if needed."""
    candidate_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Fetch live model list from Groq account to append any newly enabled chat models
    try:
        m_resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
        if m_resp.status_code == 200:
            for item in m_resp.json().get("data", []):
                mid = item.get("id", "")
                if mid and not any(x in mid.lower() for x in ["whisper", "guard", "orpheus", "tts", "embed"]):
                    if mid not in candidate_models:
                        candidate_models.append(mid)
    except Exception:
        pass

    last_err = "Groq call failed"
    for model in candidate_models:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                    ],
                    "temperature": 0.25,
                    "max_tokens": 450,
                },
                timeout=12,
            )
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"]
                return {"success": True, "provider": "Groq", "model": model, "output": text, "intermediate_steps": steps, "error": None}
            last_err = f"{resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(last_err)


def _try_openai(api_key: str, system_prompt: str, user_query: str, steps: list) -> Dict[str, Any]:
    """Tries OpenAI (ChatGPT) models in order."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err = "OpenAI call failed"
    for model in ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-3.5-turbo"]:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                    ],
                    "temperature": 0.25,
                    "max_tokens": 450,
                },
                timeout=12,
            )
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"]
                return {"success": True, "provider": "OpenAI", "model": model, "output": text, "intermediate_steps": steps, "error": None}
            last_err = f"{resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(last_err)


def _try_gemini(api_key: str, system_prompt: str, user_query: str, steps: list) -> Dict[str, Any]:
    """Tries Gemini models in order."""
    last_err = "Gemini call failed"
    for model in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser Request: {user_query}"}]}],
                "generationConfig": {"temperature": 0.25, "maxOutputTokens": 450},
            }
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                return {"success": True, "provider": "Gemini", "model": model, "output": text, "intermediate_steps": steps, "error": None}
            last_err = f"{resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(last_err)


def generate_itinerary(
    user_query: str,
    preferred_provider: str = "Groq",
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Automatically cascades across Groq -> OpenAI -> Gemini (starting with preferred_provider)
    so if one provider or model fails, the next working one answers immediately.
    """
    keys = _discover_keys()
    context_str, steps = _gather_local_dataset_context(user_query)
    system_prompt = (
        "You are Voyage AI, an autonomous travel planning assistant. "
        "Use the verified dataset records below to answer the user's request concisely (3-5 sentences) "
        "with concrete times, real hotel/attraction names from the dataset, and budget advice.\n\n"
        + context_str
    )

    # Always attempt all configured providers in cascade order
    all_providers = ["Groq", "OpenAI", "Gemini"]
    order = [preferred_provider] + [p for p in all_providers if p != preferred_provider]
    errors = []

    for idx, provider in enumerate(order):
        api_key = keys.get(provider, "")
        if not api_key:
            continue
        try:
            if provider == "Groq":
                res = _try_groq(api_key, system_prompt, user_query, steps)
            elif provider == "OpenAI":
                res = _try_openai(api_key, system_prompt, user_query, steps)
            else:
                res = _try_gemini(api_key, system_prompt, user_query, steps)

            res["fallback_used"] = idx > 0
            return res
        except Exception as exc:
            err = classify_llm_exception(exc)
            err["provider"] = provider
            err["model"] = provider
            errors.append(err)

    primary_err = errors[0] if errors else {
        "provider": "None",
        "model": "N/A",
        "category": LLMErrorCategory.INVALID_API_KEY.value,
        "suggested_action": "Ensure GROQ_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY is saved in .env",
    }
    primary_err["fallback_status"] = " | ".join(f"{e['provider']}: {e['category']}" for e in errors)
    return {
        "success": False,
        "provider": primary_err["provider"],
        "model": primary_err["model"],
        "output": None,
        "intermediate_steps": steps,
        "fallback_used": False,
        "error": primary_err,
    }