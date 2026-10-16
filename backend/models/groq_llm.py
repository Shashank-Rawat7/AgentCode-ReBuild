
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Load the backend's environment file regardless of the
# directory from which the application is launched.
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

MODEL_NAME = "gemini-2.5-flash"


def chat(prompt: str) -> str:
    """Generate a text response using the Gemini API."""
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_GENAI_API_KEY is not configured. "
            "Add it to backend/.env."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if response.text is None:
        raise RuntimeError("Gemini returned no text response.")

    return response.text.strip()
