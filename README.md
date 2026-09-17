# 🧪 AI Model Availability Tester (Health Checker)

[![CI](https://github.com/magenbrot/ai-model-tester/actions/workflows/ci.yml/badge.svg)](https://github.com/magenbrot/ai-model-tester/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-orange.svg)](#)
[![UI: Rich](https://img.shields.io/badge/UI-Rich-magenta.svg)](https://github.com/Textualize/rich)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A fast, multithreaded Python CLI tool to automate health-checking, availability testing, latency benchmarking, and capability discovery for AI models across any OpenAI-compatible API endpoint (e.g., OpenAI, OpenRouter, Ollama, vLLM, LiteLLM, Groq, Mistral, LM Studio).

Ideal for CI/CD pipelines, synthetic monitoring (Nagios/Icinga), or daily health checks of your self-hosted LLM infrastructure.

---

## 🚀 Features & Highlights

- 🔍 **Automatic Model Discovery**: Queries the endpoint (`/v1/models`) and catalogs all served models.
- 🎯 **Capability & Modality Detection**: Automatically parses model features (🛠️ Tools/Function Calling, 🧠 Reasoning/Chain-of-Thought, 📋 Structured Outputs/JSON, 🖼️ Vision, 🎥 Video, ⚡ Streaming) and maximum context length (e.g., 128k, 1M).
- 🔄 **Type-Specific Testing**: Dispatches the correct payloads and endpoints for Chat, Completion, Embedding, and Reranker models.
- ⚡ **Health & Latency Benchmarking**: Sends live test requests and records exact response time in milliseconds with color-coded latency indicators.
- 📊 **Rich Terminal Dashboard**: Visual status badges (`✔ OK`, `✖ ERROR`, `⊘ IGNORED`), capability indicators, and clean tabular summaries powered by `rich`.
- 🚀 **Concurrent Multi-Threading**: Validates dozens of models in parallel within seconds.
- 🚫 **Model Blacklist**: Exclude specific model IDs from testing (e.g. audio/TTS, moderation, or image generation models).
- ⚙️ **Flexible Configuration**: Configurable via `.env` file, `config.py`, or directly through CLI flags.
- 🤖 **CI/CD & Monitoring Ready**: Standardized shell exit codes and optional `--json` export for automated workflows and alerting.

---

## 📦 Installation & Requirements

Python 3.10+ is required. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

*(Dependencies: `requests`, `python-dotenv`, `rich`)*

---

## ⚙️ Configuration

Copy `.env.example` to create your own `.env`:

```bash
cp .env.example .env
```

Adjust the values in `.env` (or directly in `config.py`):

```ini
# Base URL of your API endpoint
API_BASE_URL=https://api.openai.com/v1

# API token / secret key
API_TOKEN=sk-your-token-here

# Comma-separated list of model IDs to ignore
IGNORED_MODELS=whisper-1,dall-e-3,tts-1,text-moderation-latest

# Optional settings
REQUEST_TIMEOUT=15
MAX_CONCURRENT=3
TEST_PROMPT=Reply with 'OK' only.
```

### Configuration via `config.py`

Alternatively, you can modify default parameters directly in [config.py](file:///home/ovoelker/software/OVTEC/ai-model-tester/config.py):

```python
API_BASE_URL = "https://api.openai.com/v1"
API_TOKEN = "sk-..."
IGNORED_MODELS = ["whisper-1", "dall-e-3"]
```

---

## 🖥️ Usage

> [!NOTE]
> **Token Usage & API Costs**: Live tests execute a minimal request per model (typically ~5–10 input tokens and up to 10 max output tokens, e.g., *"Reply with 'OK' only."*). Depending on your API provider and pricing tiers, this will consume tokens and may incur small API costs. Use `--dry-run` to inspect models and capabilities without sending test queries or incurring costs.

### 1. Default Run
Reads configuration from `.env` / `config.py` and runs the suite:

```bash
python3 model_tester.py
```

### 2. Dry Run (List Models Without Sending Test Queries)

```bash
python3 model_tester.py --dry-run
```

### 3. Override Settings via CLI Arguments

```bash
# Custom endpoint or token
python3 model_tester.py --url http://localhost:11434/v1 --token dummy

# Ignore additional models on the fly
python3 model_tester.py --ignore gpt-4o gpt-3.5-turbo

# Filter models by capability (e.g. vision, tools, reasoning, json, embedding)
python3 model_tester.py --filter-cap=vision
python3 model_tester.py --filter-cap tools reasoning

# Adjust concurrency and timeout
python3 model_tester.py --parallel 5 --timeout 10

# Export results as JSON for automation or monitoring
python3 model_tester.py --json
```

### 4. Recognized Capabilities & Badges

The console dashboard highlights model abilities with visual tags:
- 🛠️ **Tools**: Function Calling / Tool Use
- 🧠 **Reason**: Reasoning / Thinking / Chain-of-Thought
- 📋 **JSON**: Structured Outputs / JSON Schema
- 🖼️ **Vision**: Image processing capabilities
- 🎥 **Video**: Video processing capabilities
- ⚡ **Stream**: Streaming support
- **Context**: Maximum context window token length (e.g. `128k`, `1M`)

---

## 🛑 Exit Codes

The tool returns standard POSIX exit codes, making it seamless to integrate with cronjobs, CI/CD pipelines, or monitoring agents (Nagios, Icinga, Zabbix):

- `0`: All tested models are available and answered successfully (or were ignored).
- `1`: At least one model returned an error (e.g. 4xx, 5xx, timeout).
- `2`: Unable to reach the endpoint or failed to fetch the model list.

---

## 🧪 Running Tests

Run the test suite locally using Python's built-in `unittest` (no extra dependencies required):

```bash
python3 -m unittest discover -s tests
```

Or install development dependencies and run via `pytest`:

```bash
pip install -r requirements-dev.txt
pytest -v
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
