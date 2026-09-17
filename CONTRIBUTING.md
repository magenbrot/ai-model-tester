# Contributing to AI Model Availability Tester 🧪

Thank you for your interest in contributing to **AI Model Availability Tester**! This document outlines guidelines and best practices for contributing to this project.

---

## 📋 Table of Contents

1. [Code of Conduct](#-code-of-conduct)
2. [How Can I Contribute?](#-how-can-i-contribute)
   - [Reporting Bugs](#reporting-bugs)
   - [Suggesting Enhancements](#suggesting-enhancements)
   - [Pull Requests](#pull-requests)
3. [Local Development Setup](#-local-development-setup)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running Tests](#running-tests)
4. [Coding Guidelines & Quality](#-coding-guidelines--quality)
   - [Code Style](#code-style)
   - [Type Annotations](#type-annotations)
5. [Pull Request Process](#-pull-request-process)

---

## 🤝 Code of Conduct

Please help us maintain a friendly, welcoming, and respectful environment for all contributors.

---

## 💡 How Can I Contribute?

### Reporting Bugs
If you find a bug:
1. Search existing issues to verify it hasn't been reported already.
2. Open a new issue with a clear title and description.
3. Include relevant details: Python version, operating system, target API provider (e.g. OpenAI, Ollama, vLLM), and output/error logs.

### Suggesting Enhancements
Feature requests are always welcome! Please describe:
- The problem you are trying to solve.
- The proposed solution or behavior.
- Any alternative solutions you considered.

---

## 🛠️ Local Development Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Installation
```bash
git clone https://github.com/<your-username>/ai-model-tester.git
cd ai-model-tester

# Optional: Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running Tests
Run the unit test suite locally before pushing:

```bash
# Using unittest (no extra dependencies)
python3 -m unittest discover -s tests

# Or using pytest
pytest -v
```

---

## 📏 Coding Guidelines & Quality

- **Linting & Formatting**: We use [Ruff](https://github.com/astral-sh/ruff) for fast linting and formatting. Run:
  ```bash
  ruff check .
  ruff format --check .
  ```
- **Type Hints**: Use Python 3.10+ type annotations (`int | None`, `list[str]`, etc.).
- **Minimal Dependencies**: Keep runtime dependencies minimal (`requests`, `python-dotenv`, `rich`).
- **OpenAI Standard Compatibility**: Ensure changes maintain compatibility with major providers (OpenAI, Ollama, vLLM, LiteLLM, OpenRouter).

---

## 🚀 Pull Request Process

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
2. Commit your changes with clear, descriptive commit messages.
3. Verify that all tests pass (`pytest` or `python3 -m unittest discover -s tests`).
4. Push to your branch and open a Pull Request against `main`.
