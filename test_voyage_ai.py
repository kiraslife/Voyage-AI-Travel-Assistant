# test_voyage_ai.py
import unittest
from unittest.mock import patch
import plotly.graph_objects as go
from agent.travel_agent import (
    classify_llm_exception,
    LLMErrorCategory,
    generate_itinerary,
)


class TestVoyageAIVerification(unittest.TestCase):
    def test_1_plotly_scattermap_available(self):
        """Verifies that go.Scattermap exists and constructs a valid MapLibre trace without Scattermapbox."""
        self.assertTrue(hasattr(go, "Scattermap"), "Plotly must expose go.Scattermap")
        fig = go.Figure(go.Scattermap(lat=[15.2993], lon=[74.1240], mode="markers"))
        fig.update_layout(map=dict(style="open-street-map", center=dict(lat=15.2993, lon=74.1240), zoom=10))
        self.assertEqual(fig.layout.map.style, "open-street-map")

    def test_2_classify_daily_quota_429(self):
        """Verifies that daily 429 quota errors are marked non-retryable."""
        err = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for metric: GenerateRequestsPerDayPerProjectPerModel")
        info = classify_llm_exception(err)
        self.assertEqual(info["category"], LLMErrorCategory.DAILY_QUOTA_EXCEEDED.value)
        self.assertFalse(info["is_retryable"])

    def test_3_classify_rpm_rate_limit_429(self):
        """Verifies that transient RPM 429 limits are marked retryable and parse retry delay."""
        err = Exception("429 Resource Exhausted: rate_limit exceeded. Please retry in 12.5s.")
        info = classify_llm_exception(err)
        self.assertEqual(info["category"], LLMErrorCategory.RPM_RATE_LIMIT.value)
        self.assertTrue(info["is_retryable"])
        self.assertEqual(info["retry_after"], 12.5)

    def test_4_classify_invalid_api_key(self):
        """Verifies 401/403 API key errors are cleanly classified."""
        err = Exception("403 PERMISSION_DENIED: API key not valid. Please pass a valid API_KEY.")
        info = classify_llm_exception(err)
        self.assertEqual(info["category"], LLMErrorCategory.INVALID_API_KEY.value)
        self.assertFalse(info["is_retryable"])

    @patch("agent.travel_agent._execute_agent_with_backoff")
    @patch("agent.travel_agent.get_available_providers")
    def test_5_no_fake_itinerary_on_quota_failure(self, mock_providers, mock_exec):
        """Ensures failed LLM requests never fabricate an itinerary and accurately report error details."""
        mock_providers.return_value = {
            "Gemini": {"configured": True, "installed": True, "model": "gemini-2.5-flash", "env_var": "GOOGLE_API_KEY"},
            "Groq": {"configured": False, "installed": False, "model": "llama-3.3-70b-versatile", "env_var": "GROQ_API_KEY"},
        }
        mock_exec.return_value = {
            "success": False,
            "provider": "Gemini",
            "model": "gemini-2.5-flash",
            "output": None,
            "intermediate_steps": [],
            "error": {
                "provider": "Gemini",
                "model": "gemini-2.5-flash",
                "category": LLMErrorCategory.DAILY_QUOTA_EXCEEDED.value,
                "is_retryable": False,
                "suggested_action": "Switch providers.",
            },
        }
        res = generate_itinerary("Plan a 4 day trip to Goa")
        self.assertFalse(res["success"])
        self.assertIsNone(res["output"])
        self.assertEqual(res["error"]["category"], LLMErrorCategory.DAILY_QUOTA_EXCEEDED.value)


if __name__ == "__main__":
    unittest.main(verbosity=2)