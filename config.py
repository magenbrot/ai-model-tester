"""
Configuration for the AI Model Availability Tester.

Values can be specified:
1. In a `.env` file in the same directory (recommended)
2. Directly in this file as default values
3. Via system environment variables
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from this script's directory or the current working directory
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()  # Fallback to default search path

# ------------------------------------------------------------------------------
# 1. API Connection Settings
# ------------------------------------------------------------------------------
# Base URL of the OpenAI-compatible API endpoint
# e.g.:
# - https://api.openai.com/v1
# - https://openrouter.ai/api/v1
# - http://localhost:11434/v1 (Ollama)
# - http://localhost:8000/v1 (vLLM)
API_BASE_URL: str = (
    os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
)

# Authentication token / API key
API_TOKEN: str = os.getenv("API_TOKEN", "").strip()

# ------------------------------------------------------------------------------
# 2. Model Filter (Blacklist)
# ------------------------------------------------------------------------------
# Model IDs to exclude from testing.
# Can be set in .env as comma-separated list ("model1,model2") or here as a list.
_env_ignored = os.getenv("IGNORED_MODELS")
if _env_ignored is not None:
    # Explicitly configured in .env / environment (e.g. IGNORED_MODELS="" or "none")
    _cleaned = _env_ignored.strip().strip('"').strip("'")
    if _cleaned.lower() in ("", "none", "null", "[]"):
        IGNORED_MODELS: list[str] = []
    else:
        IGNORED_MODELS: list[str] = [
            m.strip() for m in _cleaned.split(",") if m.strip()
        ]
else:
    # Default list if not specified in .env:
    IGNORED_MODELS: list[str] = [
        "whisper-1",
        "dall-e-2",
        "dall-e-3",
        "tts-1",
        "tts-1-hd",
        "text-moderation-latest",
        "text-moderation-stable",
    ]

# ------------------------------------------------------------------------------
# 3. Test Run Settings
# ------------------------------------------------------------------------------
# HTTP request timeout in seconds
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "15"))

# Maximum concurrent requests
MAX_CONCURRENT: int = int(os.getenv("MAX_CONCURRENT", "3"))

# Test prompt for chat/text completion models
TEST_PROMPT: str = os.getenv("TEST_PROMPT", "Reply with 'OK' only.")


def mask_token(token: str) -> str:
    """Masks the API token for secure screen output."""
    if not token:
        return "[red]<NOT SET>[/red]"
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}...{token[-4:]}"
