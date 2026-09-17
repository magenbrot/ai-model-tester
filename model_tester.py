#!/usr/bin/env python3
"""
AI Model Availability Tester
============================

Tests OpenAI-compatible API endpoints:
1. Discovers available models (/v1/models) with capabilities & modalities
2. Filters out configured ignored model IDs
3. Tests each remaining model with a type-specific request (Chat, Embedding, Rerank, Completion)
4. Displays a visual results dashboard in the terminal
"""

import argparse
import concurrent.futures
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

import config

console = Console()


@dataclass
class ModelCapabilities:
    modalities: list[str] = field(default_factory=list)  # ["text", "image", "video", "audio"]
    features: list[str] = field(default_factory=list)  # ["tools", "reasoning", "structured_outputs", "streaming"]
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    task_type: str = "Chat"  # "Chat", "Completion", "Embedding", "Reranker", "Unknown"


@dataclass
class ModelInfo:
    id: str
    name: str = ""
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    model_id: str
    status: str  # "OK", "ERROR", "IGNORED"
    model_type: str  # "Chat", "Completion", "Embedding", "Reranker", "Unknown"
    latency_ms: float
    status_code: Optional[int]
    message: str
    capabilities: Optional[ModelCapabilities] = None


def resolve_api_endpoints(base_url: str) -> dict[str, str]:
    """
    Resolves the correct endpoint URLs based on the base URL.
    Supports base URLs both with and without a trailing '/v1'.
    """
    clean_base = base_url.rstrip("/")
    if clean_base.endswith("/v1"):
        root = clean_base
        base_without_v1 = clean_base[:-3]
    else:
        root = f"{clean_base}/v1"
        base_without_v1 = clean_base

    return {
        "models": f"{root}/models",
        "chat": f"{root}/chat/completions",
        "completions": f"{root}/completions",
        "embeddings": f"{root}/embeddings",
        "rerank": f"{root}/rerank",
        "alt_rerank": f"{base_without_v1}/rerank",
        "alt_models": f"{base_without_v1}/models",
    }


def get_headers(token: str) -> dict[str, str]:
    """Generates standard HTTP request headers."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AI-Model-Tester/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_model_info(item: Any) -> ModelInfo:
    """Extracts model metadata and capabilities from API response payload."""
    if isinstance(item, str):
        item = {"id": item}
    elif not isinstance(item, dict):
        item = {"id": str(item)}

    model_id = str(item.get("id") or item.get("name") or "").strip()
    name = str(item.get("name") or model_id).strip()

    modalities: list[str] = []
    features: list[str] = []
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    task_type = "Chat"

    # 1. Input modalities & context length
    in_mods = item.get("input_modalities", [])
    if isinstance(in_mods, list):
        for m in in_mods:
            if isinstance(m, dict):
                m_type = m.get("type")
                if m_type and str(m_type).lower() not in modalities:
                    modalities.append(str(m_type).lower())
                ctx = (
                    m.get("supported_inputs", {})
                    .get("max_context_length", {})
                    .get("value")
                )
                if ctx and not context_length:
                    try:
                        context_length = int(ctx)
                    except (ValueError, TypeError):
                        pass
            elif isinstance(m, str) and m.lower() not in modalities:
                modalities.append(m.lower())

    # 2. Output modalities & parameters (tools, reasoning, structured outputs)
    out_mods = item.get("output_modalities", [])
    if isinstance(out_mods, list):
        for m in out_mods:
            if isinstance(m, dict):
                otype = str(m.get("type", "")).lower()
                if otype in ("embeddings", "embedding"):
                    task_type = "Embedding"
                elif otype in ("rerank", "reranker"):
                    task_type = "Reranker"
                elif otype in ("text", "completion", "chat") and task_type not in ("Embedding", "Reranker"):
                    task_type = "Chat"

                params = m.get("supported_parameters", {})
                if isinstance(params, dict):
                    if "tools" in params and "tools" not in features:
                        features.append("tools")
                    if "reasoning" in params and "reasoning" not in features:
                        features.append("reasoning")
                    if ("structured_outputs" in params or "response_format" in params) and "structured_outputs" not in features:
                        features.append("structured_outputs")
                elif isinstance(params, list):
                    for p in params:
                        p_str = str(p).lower()
                        if "tool" in p_str and "tools" not in features:
                            features.append("tools")
                        elif "reason" in p_str and "reasoning" not in features:
                            features.append("reasoning")
                        elif ("struct" in p_str or "json" in p_str) and "structured_outputs" not in features:
                            features.append("structured_outputs")

                if m.get("streaming") and "streaming" not in features:
                    features.append("streaming")

                max_len = m.get("max_length", {}).get("value")
                if max_len and not max_output_tokens:
                    try:
                        max_output_tokens = int(max_len)
                    except (ValueError, TypeError):
                        pass

    # 3. Fallback / Alternative Schemas (e.g. OpenRouter, vLLM, LiteLLM)
    if not context_length:
        for k in ("context_length", "context_window", "max_model_len", "max_context_length"):
            if k in item and isinstance(item[k], (int, float)):
                context_length = int(item[k])
                break

    or_arch = item.get("architecture", {})
    if isinstance(or_arch, dict):
        modality_str = str(or_arch.get("modality", "")).lower()
        if "multimodal" in modality_str or "image" in modality_str:
            if "image" not in modalities:
                modalities.append("image")

    or_params = item.get("supported_parameters", [])
    if isinstance(or_params, list):
        for p in or_params:
            p_str = str(p).lower()
            if "tool" in p_str and "tools" not in features:
                features.append("tools")
            elif "reason" in p_str and "reasoning" not in features:
                features.append("reasoning")
            elif "structured_outputs" in p_str and "structured_outputs" not in features:
                features.append("structured_outputs")

    # 4. Heuristics based on model ID
    mid_lower = model_id.lower()
    if "rerank" in mid_lower:
        task_type = "Reranker"
    elif any(k in mid_lower for k in ("embed", "bge-", "e5-", "text-similarity")):
        task_type = "Embedding"

    if any(k in mid_lower for k in ("vision", "-vl", "llava", "pixtral", "4o", "gemini-1.5", "gemini-2.0", "claude-3")):
        if "image" not in modalities:
            modalities.append("image")

    if any(k in mid_lower for k in ("deepseek-r1", "o1-", "o3-", "-reasoning", "-thinking")):
        if "reasoning" not in features:
            features.append("reasoning")

    caps = ModelCapabilities(
        modalities=modalities,
        features=features,
        context_length=context_length,
        max_output_tokens=max_output_tokens,
        task_type=task_type,
    )
    return ModelInfo(id=model_id, name=name, capabilities=caps, raw_data=item)


def fetch_models(base_url: str, token: str, timeout: int) -> list[ModelInfo]:
    """
    Fetches the list of available models from the server.
    Supports standard OpenAI format (`data: [...]`) and Ollama format (`models: [...]`).
    Returns a sorted list of ModelInfo objects with parsed capabilities.
    """
    endpoints = resolve_api_endpoints(base_url)
    headers = get_headers(token)

    urls_to_try = [endpoints["models"]]
    if endpoints["models"] != endpoints["alt_models"]:
        urls_to_try.append(endpoints["alt_models"])

    last_error = None

    for url in urls_to_try:
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                raw_items = []
                # 1. OpenAI format: {"data": [{...}, ...]}
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                    raw_items = data["data"]
                # 2. Ollama format: {"models": [{...}, ...]}
                elif isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
                    raw_items = data["models"]
                # 3. Direct array: [{...}, ...] or ["model1", ...]
                elif isinstance(data, list):
                    raw_items = data

                model_infos: dict[str, ModelInfo] = {}
                for item in raw_items:
                    info = parse_model_info(item)
                    if info.id:
                        model_infos[info.id] = info

                if model_infos:
                    return [model_infos[k] for k in sorted(model_infos.keys())]
                else:
                    return []
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:120]}"
        except requests.exceptions.RequestException as e:
            last_error = str(e)

    raise RuntimeError(f"Could not fetch models from '{base_url}'. Details: {last_error}")


def is_model_ignored(model_id: str, ignored_models: list[str]) -> bool:
    """Checks whether a model should be ignored (exact or case-insensitive match)."""
    clean_id = model_id.strip().lower()
    for ignored in ignored_models:
        if clean_id == ignored.strip().lower():
            return True
    return False


def normalize_capability_name(cap: str) -> str:
    """Normalizes capability aliases to canonical keys."""
    c = cap.strip().lower()
    mapping = {
        "vision": "image",
        "image": "image",
        "video": "video",
        "audio": "audio",
        "tools": "tools",
        "tool": "tools",
        "function": "tools",
        "functions": "tools",
        "function_calling": "tools",
        "reasoning": "reasoning",
        "reason": "reasoning",
        "thinking": "reasoning",
        "json": "structured_outputs",
        "structured": "structured_outputs",
        "structured_outputs": "structured_outputs",
        "streaming": "streaming",
        "stream": "streaming",
        "chat": "chat",
        "embedding": "embedding",
        "embed": "embedding",
        "embeddings": "embedding",
        "rerank": "reranker",
        "reranker": "reranker",
        "completion": "completion",
    }
    return mapping.get(c, c)


def model_matches_capability(caps: ModelCapabilities, normalized_cap: str) -> bool:
    """Checks if a model's capabilities match a normalized capability key."""
    if normalized_cap == "multimodal":
        return bool("image" in caps.modalities or "video" in caps.modalities or "audio" in caps.modalities)
    if normalized_cap in caps.modalities:
        return True
    if normalized_cap in caps.features:
        return True
    if normalized_cap == caps.task_type.lower():
        return True
    return False


def filter_models_by_capabilities(models: list[ModelInfo], required_caps: list[str]) -> list[ModelInfo]:
    """Filters a list of models to only those matching all required capabilities."""
    if not required_caps:
        return models

    normalized_required = [normalize_capability_name(c) for c in required_caps]
    return [
        m for m in models
        if all(model_matches_capability(m.capabilities, req) for req in normalized_required)
    ]


def test_single_model(
    model_info: ModelInfo,
    base_url: str,
    token: str,
    test_prompt: str,
    timeout: int,
    ignored_models: list[str],
) -> TestResult:
    """Tests an individual model with a type-specific request and measures latency."""
    model_id = model_info.id
    caps = model_info.capabilities

    if is_model_ignored(model_id, ignored_models):
        return TestResult(
            model_id=model_id,
            status="IGNORED",
            model_type=caps.task_type,
            latency_ms=0.0,
            status_code=None,
            message="Ignored in configuration",
            capabilities=caps,
        )

    endpoints = resolve_api_endpoints(base_url)
    headers = get_headers(token)
    task_type = caps.task_type

    if task_type == "Reranker":
        endpoint_url = endpoints["rerank"]
        payload: dict[str, Any] = {
            "model": model_id,
            "query": "healthcheck",
            "documents": [{"text": "healthcheck document"}],
        }
        model_type = "Reranker"
    elif task_type == "Embedding":
        endpoint_url = endpoints["embeddings"]
        payload = {"model": model_id, "input": "healthcheck"}
        model_type = "Embedding"
    else:
        endpoint_url = endpoints["chat"]
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": test_prompt}],
            "max_tokens": 10,
        }
        model_type = "Chat"

    start_time = time.perf_counter()
    try:
        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Fallbacks:
        # If Rerank returns 404, try alt_rerank
        if response.status_code == 404 and model_type == "Reranker" and endpoints["rerank"] != endpoints["alt_rerank"]:
            rerank_alt_start = time.perf_counter()
            response = requests.post(endpoints["alt_rerank"], headers=headers, json=payload, timeout=timeout)
            latency_ms = (time.perf_counter() - rerank_alt_start) * 1000

        # If Chat returns 400/404, optionally fall back to legacy completions or embedding
        if response.status_code in (400, 404) and model_type == "Chat":
            err_text = response.text.lower()
            if "embedding" in err_text:
                # Automatic fallback to embedding endpoint
                emb_start = time.perf_counter()
                response = requests.post(
                    endpoints["embeddings"],
                    headers=headers,
                    json={"model": model_id, "input": "healthcheck"},
                    timeout=timeout,
                )
                latency_ms = (time.perf_counter() - emb_start) * 1000
                model_type = "Embedding"
            elif "not support chat" in err_text or response.status_code == 404:
                comp_start = time.perf_counter()
                response = requests.post(
                    endpoints["completions"],
                    headers=headers,
                    json={"model": model_id, "prompt": test_prompt, "max_tokens": 10},
                    timeout=timeout,
                )
                latency_ms = (time.perf_counter() - comp_start) * 1000
                model_type = "Completion"

        if response.status_code in (200, 201):
            try:
                res_data = response.json()
                if model_type == "Chat":
                    snippet = (
                        res_data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                elif model_type == "Completion":
                    snippet = res_data.get("choices", [{}])[0].get("text", "").strip()
                elif model_type == "Embedding":
                    dims = len(res_data.get("data", [{}])[0].get("embedding", []))
                    snippet = f"Embedding generated ({dims} dims)"
                elif model_type == "Reranker":
                    res_list = res_data.get("results", [])
                    if res_list and isinstance(res_list, list):
                        score = res_list[0].get("relevance_score")
                        if score is not None:
                            snippet = f"Rerank OK (Score: {float(score):.2f})"
                        else:
                            snippet = "Rerank OK"
                    else:
                        snippet = "Rerank OK"
                else:
                    snippet = "Response received"

                clean_snippet = " ".join(snippet.split())
                if len(clean_snippet) > 60:
                    clean_snippet = clean_snippet[:57] + "..."
                if not clean_snippet:
                    clean_snippet = "Empty response (OK)"

                return TestResult(
                    model_id=model_id,
                    status="OK",
                    model_type=model_type,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                    message=clean_snippet,
                    capabilities=caps,
                )
            except Exception:
                return TestResult(
                    model_id=model_id,
                    status="OK",
                    model_type=model_type,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                    message="HTTP 200 received (response not parseable)",
                    capabilities=caps,
                )
        else:
            # Error HTTP response
            try:
                err_json = response.json()
                if "error" in err_json:
                    err_val = err_json["error"]
                    if isinstance(err_val, dict):
                        err_msg = err_val.get("message", str(err_val))
                    else:
                        err_msg = str(err_val)
                else:
                    err_msg = response.text
            except Exception:
                err_msg = response.text

            clean_err = " ".join(err_msg.split())
            if len(clean_err) > 70:
                clean_err = clean_err[:67] + "..."

            return TestResult(
                model_id=model_id,
                status="ERROR",
                model_type=model_type,
                latency_ms=latency_ms,
                status_code=response.status_code,
                message=clean_err or f"HTTP {response.status_code}",
                capabilities=caps,
            )

    except requests.exceptions.Timeout:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return TestResult(
            model_id=model_id,
            status="ERROR",
            model_type=model_type,
            latency_ms=latency_ms,
            status_code=None,
            message=f"Timeout after {timeout}s",
            capabilities=caps,
        )
    except requests.exceptions.RequestException as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return TestResult(
            model_id=model_id,
            status="ERROR",
            model_type=model_type,
            latency_ms=latency_ms,
            status_code=None,
            message=f"Connection error: {str(e)[:50]}",
            capabilities=caps,
        )


def format_context_length(tokens: Optional[int]) -> Text:
    """Formats context length into a compact k/M string."""
    if not tokens or tokens <= 0:
        return Text("-", style="dim")
    if tokens >= 1_000_000:
        val = tokens / 1_000_000
        text = f"{val:.1f}M".replace(".0M", "M")
        return Text(text, style="bold cyan")
    elif tokens >= 1_000:
        val = tokens / 1_000
        text = f"{val:.0f}k"
        return Text(text, style="cyan")
    else:
        return Text(str(tokens), style="dim")


def format_capabilities(caps: Optional[ModelCapabilities]) -> Text:
    """Formats capabilities into concise badges."""
    if not caps:
        return Text("-", style="dim")

    parts: list[Text] = []
    # Modalities
    if "image" in caps.modalities:
        parts.append(Text("🖼️ Vision", style="bold yellow"))
    if "video" in caps.modalities:
        parts.append(Text("🎥 Video", style="bold blue"))
    if "audio" in caps.modalities:
        parts.append(Text("🎙️ Audio", style="bold magenta"))

    # Features
    if "tools" in caps.features:
        parts.append(Text("🛠️ Tools", style="bold green"))
    if "reasoning" in caps.features:
        parts.append(Text("🧠 Reason", style="bold magenta"))
    if "structured_outputs" in caps.features:
        parts.append(Text("📋 JSON", style="bold bright_cyan"))

    if not parts:
        if "streaming" in caps.features:
            return Text("⚡ Stream", style="dim")
        return Text("-", style="dim")

    res = Text()
    for i, p in enumerate(parts):
        if i > 0:
            res.append(" ")
        res.append_text(p)
    return res


def format_latency(latency_ms: float, status: str) -> Text:
    """Formats latency with colored badges based on threshold."""
    if status == "IGNORED":
        return Text("-", style="dim")
    if latency_ms <= 0:
        return Text("-", style="dim")

    if latency_ms < 1000:
        return Text(f"{latency_ms:6.0f} ms", style="bold green")
    elif latency_ms < 3000:
        return Text(f"{latency_ms:6.0f} ms", style="bold yellow")
    else:
        return Text(f"{latency_ms / 1000:5.2f} s", style="bold red")


def display_header(
    base_url: str,
    token: str,
    ignored: list[str],
    max_concurrency: int,
    timeout: int,
    filter_caps: Optional[list[str]] = None,
):
    """Displays an informative header panel before running tests."""
    masked = config.mask_token(token)
    ignored_str = ", ".join(ignored) if ignored else "[dim]None[/dim]"

    content = Text()
    content.append("API Base URL : ", style="bold cyan")
    content.append(f"{base_url}\n")
    content.append("API Token    : ", style="bold cyan")
    content.append(f"{masked}\n")
    content.append("Parallel Req : ", style="bold cyan")
    content.append(f"{max_concurrency} Workers | Timeout: {timeout}s\n")
    if filter_caps:
        content.append("Cap Filter   : ", style="bold cyan")
        content.append(f"{', '.join(filter_caps)}\n")
    content.append("Ignored IDs  : ", style="bold cyan")
    content.append(f"{ignored_str}")

    panel = Panel(
        content,
        title="[bold blue]🤖 AI Model Availability Checker[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


def display_results_table(results: list[TestResult]):
    """Creates the Rich table showing all results including capabilities."""
    table = Table(
        title="AI Model Test Results",
        title_style="bold white",
        header_style="bold cyan",
        show_header=True,
        border_style="bright_black",
        expand=True,
    )

    table.add_column("Status", justify="center", width=12)
    table.add_column("Model ID", style="bold white", min_width=24)
    table.add_column("Type", justify="center", width=10)
    table.add_column("Context", justify="right", width=9)
    table.add_column("Capabilities", min_width=22)
    table.add_column("Latency", justify="right", width=10)
    table.add_column("Code", justify="center", width=6)
    table.add_column("Response / Error Message", overflow="ellipsis")

    for r in results:
        if r.status == "OK":
            status_text = Text("✔ OK", style="bold green")
            code_text = Text(str(r.status_code or 200), style="green")
            msg_text = Text(r.message, style="green")
        elif r.status == "IGNORED":
            status_text = Text("⊘ IGNORED", style="bold yellow")
            code_text = Text("-", style="dim")
            msg_text = Text(r.message, style="yellow dim")
        else:  # ERROR
            status_text = Text("✖ ERROR", style="bold red")
            code_text = Text(str(r.status_code or "ERR"), style="bold red")
            msg_text = Text(r.message, style="red")

        if r.model_type == "Chat":
            type_style = "cyan"
        elif r.model_type == "Embedding":
            type_style = "magenta"
        elif r.model_type == "Reranker":
            type_style = "yellow"
        elif r.model_type == "Completion":
            type_style = "bright_black"
        else:
            type_style = "dim"

        type_text = Text(r.model_type, style=type_style)
        ctx_tokens = r.capabilities.context_length if r.capabilities else None
        ctx_text = format_context_length(ctx_tokens)
        caps_text = format_capabilities(r.capabilities)
        latency_text = format_latency(r.latency_ms, r.status)

        table.add_row(
            status_text,
            r.model_id,
            type_text,
            ctx_text,
            caps_text,
            latency_text,
            code_text,
            msg_text,
        )

    console.print(table)


def display_summary(results: list[TestResult], elapsed_seconds: float):
    """Displays a summary statistic panel of all tests."""
    total = len(results)
    ok_count = sum(1 for r in results if r.status == "OK")
    error_count = sum(1 for r in results if r.status == "ERROR")
    ignored_count = sum(1 for r in results if r.status == "IGNORED")
    tested_count = ok_count + error_count

    avg_latency = 0.0
    ok_latencies = [r.latency_ms for r in results if r.status == "OK"]
    if ok_latencies:
        avg_latency = sum(ok_latencies) / len(ok_latencies)

    summary_text = Text()
    summary_text.append(f"Models found: {total}  |  ", style="bold")
    summary_text.append(f"Tested: {tested_count}  |  ", style="bold cyan")
    summary_text.append(f"Passed: {ok_count}  |  ", style="bold green")
    summary_text.append(f"Failed: {error_count}  |  ", style="bold red" if error_count > 0 else "dim")
    summary_text.append(f"Ignored: {ignored_count}\n", style="bold yellow" if ignored_count > 0 else "dim")
    summary_text.append(f"Total duration: {elapsed_seconds:.2f}s", style="dim")
    if ok_latencies:
        summary_text.append(f"  |  Avg Latency (OK): {avg_latency:.0f}ms", style="dim green")

    border_color = "green" if error_count == 0 else "red"
    title_status = "[bold green]All tests passed[/bold green]" if error_count == 0 else f"[bold red]{error_count} model(s) failed[/bold red]"

    panel = Panel(
        summary_text,
        title=f"Summary: {title_status}",
        border_style=border_color,
        padding=(0, 2),
    )
    console.print(panel)


def main():
    parser = argparse.ArgumentParser(
        description="Tests availability and health of AI models on OpenAI-compatible APIs."
    )
    parser.add_argument("--url", default=config.API_BASE_URL, help="API Base URL (overrides .env)")
    parser.add_argument("--token", default=config.API_TOKEN, help="API Token (overrides .env)")
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="Additional model IDs to ignore (space or comma separated)",
    )
    parser.add_argument(
        "--filter-cap",
        nargs="*",
        default=[],
        help="Filter models by capability/modality (e.g. vision, tools, reasoning, json, streaming, audio, video). Multiple filters can be space- or comma-separated.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=config.REQUEST_TIMEOUT,
        help=f"Request timeout in seconds (default: {config.REQUEST_TIMEOUT})",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=config.MAX_CONCURRENT,
        help=f"Maximum concurrent requests (default: {config.MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and list models only, without sending test requests",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output test results as JSON to stdout",
    )

    args = parser.parse_args()

    # Merge ignored models
    ignored_models = list(config.IGNORED_MODELS)
    if args.ignore:
        for ig in args.ignore:
            parts = [p.strip() for p in ig.split(",") if p.strip()]
            ignored_models.extend(parts)
    ignored_models = list(dict.fromkeys(ignored_models))

    # Parse capability filter
    filter_caps: list[str] = []
    if args.filter_cap:
        for fc in args.filter_cap:
            parts = [p.strip().lower() for p in fc.split(",") if p.strip()]
            filter_caps.extend(parts)
    filter_caps = list(dict.fromkeys(filter_caps))

    # Display initial header
    display_header(
        base_url=args.url,
        token=args.token,
        ignored=ignored_models,
        max_concurrency=args.parallel,
        timeout=args.timeout,
        filter_caps=filter_caps,
    )

    # 1. Fetch models
    with console.status("[bold cyan]Fetching model list from server...", spinner="dots"):
        try:
            models = fetch_models(base_url=args.url, token=args.token, timeout=args.timeout)
        except Exception as e:
            console.print(f"\n[bold red]Error fetching models:[/bold red] {e}")
            sys.exit(2)

    if not models:
        console.print("[bold yellow]No models found at endpoint![/bold yellow]")
        sys.exit(0)

    console.print(f"[bold green]✔ {len(models)} model(s) successfully found.[/bold green]\n")

    # Filter models by capabilities if requested
    if filter_caps:
        caps_label = ", ".join(filter_caps)
        models = filter_models_by_capabilities(models, filter_caps)
        if not models:
            console.print(
                f"[bold yellow]No models match capability filter: '{caps_label}'[/bold yellow]"
            )
            sys.exit(0)
        console.print(
            f"[bold cyan]ℹ Filtered by capability '{caps_label}': {len(models)} model(s) matching.[/bold cyan]\n"
        )

    if args.dry_run:
        console.print("[yellow]Dry-run enabled: Test requests skipped.[/yellow]")
        results = [
            TestResult(
                model_id=m.id,
                status="IGNORED" if is_model_ignored(m.id, ignored_models) else "OK",
                model_type=m.capabilities.task_type,
                latency_ms=0.0,
                status_code=None,
                message="Ignored in configuration" if is_model_ignored(m.id, ignored_models) else "Found (Dry-Run)",
                capabilities=m.capabilities,
            )
            for m in models
        ]
        display_results_table(results)

        if args.json:
            json_output = [
                {
                    "model_id": r.model_id,
                    "status": r.status,
                    "model_type": r.model_type,
                    "context_length": r.capabilities.context_length if r.capabilities else None,
                    "max_output_tokens": r.capabilities.max_output_tokens if r.capabilities else None,
                    "modalities": r.capabilities.modalities if r.capabilities else [],
                    "capabilities": r.capabilities.features if r.capabilities else [],
                    "latency_ms": round(r.latency_ms, 2),
                    "status_code": r.status_code,
                    "message": r.message,
                }
                for r in results
            ]
            console.print("\n[bold]JSON Output:[/bold]")
            console.print_json(data=json_output)

        sys.exit(0)

    # 2. Test models concurrently
    results_map: dict[str, TestResult] = {}
    model_map: dict[str, ModelInfo] = {m.id: m for m in models}
    overall_start = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Testing AI models...", total=len(models))

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_model = {
                executor.submit(
                    test_single_model,
                    model_info=m,
                    base_url=args.url,
                    token=args.token,
                    test_prompt=config.TEST_PROMPT,
                    timeout=args.timeout,
                    ignored_models=ignored_models,
                ): m.id
                for m in models
            }

            for future in concurrent.futures.as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    res = future.result()
                except Exception as exc:
                    minfo = model_map.get(model_id)
                    res = TestResult(
                        model_id=model_id,
                        status="ERROR",
                        model_type=minfo.capabilities.task_type if minfo else "Unknown",
                        latency_ms=0.0,
                        status_code=None,
                        message=f"Exception: {str(exc)[:50]}",
                        capabilities=minfo.capabilities if minfo else None,
                    )
                results_map[model_id] = res
                progress.advance(task)

    overall_elapsed = time.perf_counter() - overall_start

    # Sort results to match original model list order
    sorted_results = [results_map[m.id] for m in models if m.id in results_map]

    # 3. Print table & summary
    console.print()
    display_results_table(sorted_results)
    console.print()
    display_summary(sorted_results, overall_elapsed)

    if args.json:
        json_output = [
            {
                "model_id": r.model_id,
                "status": r.status,
                "model_type": r.model_type,
                "context_length": r.capabilities.context_length if r.capabilities else None,
                "max_output_tokens": r.capabilities.max_output_tokens if r.capabilities else None,
                "modalities": r.capabilities.modalities if r.capabilities else [],
                "capabilities": r.capabilities.features if r.capabilities else [],
                "latency_ms": round(r.latency_ms, 2),
                "status_code": r.status_code,
                "message": r.message,
            }
            for r in sorted_results
        ]
        console.print("\n[bold]JSON Output:[/bold]")
        console.print_json(data=json_output)

    # Exit code: 0 if all OK or IGNORED, 1 if any model returned an error
    has_errors = any(r.status == "ERROR" for r in sorted_results)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
