"""Optional live check against the real Groq API.

Skipped by default. It makes one small, real API call and needs a valid key:

    AGENTCODE_LIVE_GROQ=1 python -m pytest -q tests/test_live_groq.py
"""

import os

import pytest

from models import groq_llm

pytestmark = pytest.mark.skipif(
    os.getenv("AGENTCODE_LIVE_GROQ") != "1" or not os.getenv("GROQ_API_KEY"),
    reason="live Groq test: set AGENTCODE_LIVE_GROQ=1 and GROQ_API_KEY to run it",
)


def test_live_groq_returns_text():
    answer = groq_llm.chat("Reply with exactly the word: pong")
    assert "pong" in answer.lower()
