"""
Unit tests for the AI Model Availability Tester.
Can be executed with either `pytest` or Python's built-in `unittest`:
    python3 -m unittest discover -s tests
"""

import unittest
from unittest.mock import MagicMock, patch
import requests

from model_tester import (
    ModelCapabilities,
    ModelInfo,
    TestResult,
    fetch_models,
    format_capabilities,
    format_context_length,
    format_latency,
    get_headers,
    is_model_ignored,
    parse_model_info,
    resolve_api_endpoints,
    test_single_model,
)
import config


class TestEndpointsAndHeaders(unittest.TestCase):
    def test_resolve_api_endpoints_without_v1(self):
        endpoints = resolve_api_endpoints("http://localhost:8000")
        self.assertEqual(endpoints["models"], "http://localhost:8000/v1/models")
        self.assertEqual(endpoints["chat"], "http://localhost:8000/v1/chat/completions")
        self.assertEqual(endpoints["embeddings"], "http://localhost:8000/v1/embeddings")
        self.assertEqual(endpoints["rerank"], "http://localhost:8000/v1/rerank")
        self.assertEqual(endpoints["alt_rerank"], "http://localhost:8000/rerank")

    def test_resolve_api_endpoints_with_v1_and_trailing_slash(self):
        endpoints = resolve_api_endpoints("https://api.openai.com/v1/")
        self.assertEqual(endpoints["models"], "https://api.openai.com/v1/models")
        self.assertEqual(endpoints["chat"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(endpoints["alt_rerank"], "https://api.openai.com/rerank")

    def test_get_headers_with_token(self):
        headers = get_headers("secret-token-123")
        self.assertEqual(headers["Authorization"], "Bearer secret-token-123")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertIn("User-Agent", headers)

    def test_get_headers_without_token(self):
        headers = get_headers("")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(headers["Content-Type"], "application/json")


class TestConfigHelpers(unittest.TestCase):
    def test_mask_token_empty(self):
        masked = config.mask_token("")
        self.assertIn("<NOT SET>", masked)

    def test_mask_token_short(self):
        masked = config.mask_token("short")
        self.assertEqual(masked, "****")

    def test_mask_token_regular(self):
        masked = config.mask_token("sk-1234567890abcdef")
        self.assertEqual(masked, "sk-1...cdef")

    def test_is_model_ignored(self):
        ignored = ["whisper-1", "dall-e-3"]
        self.assertTrue(is_model_ignored("whisper-1", ignored))
        self.assertTrue(is_model_ignored("WHISPER-1", ignored))
        self.assertTrue(is_model_ignored(" dall-e-3 ", ignored))
        self.assertFalse(is_model_ignored("gpt-4o", ignored))


class TestModelParsing(unittest.TestCase):
    def test_parse_model_info_from_string(self):
        info = parse_model_info("gpt-4o")
        self.assertEqual(info.id, "gpt-4o")
        self.assertEqual(info.capabilities.task_type, "Chat")
        self.assertIn("image", info.capabilities.modalities)

    def test_parse_model_info_heuristics(self):
        # Embedding detection
        emb = parse_model_info({"id": "text-embedding-3-small"})
        self.assertEqual(emb.capabilities.task_type, "Embedding")

        # Reranker detection
        rerank = parse_model_info({"id": "bge-reranker-v2-m3"})
        self.assertEqual(rerank.capabilities.task_type, "Reranker")

        # Reasoning detection
        reason = parse_model_info({"id": "deepseek-r1-distill"})
        self.assertIn("reasoning", reason.capabilities.features)

    def test_parse_model_info_with_openrouter_schema(self):
        payload = {
            "id": "meta-llama/llama-3.3-70b-instruct",
            "context_length": 131072,
            "architecture": {"modality": "text+image->text"},
            "supported_parameters": ["tools", "reasoning", "structured_outputs"],
        }
        info = parse_model_info(payload)
        self.assertEqual(info.capabilities.context_length, 131072)
        self.assertIn("image", info.capabilities.modalities)
        self.assertIn("tools", info.capabilities.features)
        self.assertIn("reasoning", info.capabilities.features)
        self.assertIn("structured_outputs", info.capabilities.features)


class TestFormattingHelpers(unittest.TestCase):
    def test_format_context_length(self):
        self.assertEqual(format_context_length(None).plain, "-")
        self.assertEqual(format_context_length(0).plain, "-")
        self.assertEqual(format_context_length(128000).plain, "128k")
        self.assertEqual(format_context_length(1000000).plain, "1M")
        self.assertEqual(format_context_length(1500000).plain, "1.5M")

    def test_format_latency(self):
        self.assertEqual(format_latency(0.0, "IGNORED").plain, "-")
        self.assertEqual(format_latency(150.0, "OK").plain.strip(), "150 ms")
        self.assertEqual(format_latency(1500.0, "OK").plain.strip(), "1500 ms")
        self.assertEqual(format_latency(3500.0, "OK").plain.strip(), "3.50 s")

    def test_format_capabilities(self):
        caps = ModelCapabilities(
            modalities=["image"],
            features=["tools", "structured_outputs"],
        )
        rendered = format_capabilities(caps).plain
        self.assertIn("Vision", rendered)
        self.assertIn("Tools", rendered)
        self.assertIn("JSON", rendered)


class TestModelTesting(unittest.TestCase):
    def test_test_single_model_ignored(self):
        info = ModelInfo(id="whisper-1")
        result = test_single_model(
            model_info=info,
            base_url="https://api.openai.com/v1",
            token="test-token",
            test_prompt="OK",
            timeout=5,
            ignored_models=["whisper-1"],
        )
        self.assertEqual(result.status, "IGNORED")
        self.assertEqual(result.latency_ms, 0.0)
        self.assertEqual(result.message, "Ignored in configuration")

    @patch("requests.post")
    def test_test_single_model_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }
        mock_post.return_value = mock_response

        info = ModelInfo(id="gpt-4o-mini")
        result = test_single_model(
            model_info=info,
            base_url="https://api.openai.com/v1",
            token="test-token",
            test_prompt="OK",
            timeout=5,
            ignored_models=[],
        )
        self.assertEqual(result.status, "OK")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.message, "OK")

    @patch("requests.post")
    def test_test_single_model_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_post.return_value = mock_response

        info = ModelInfo(id="gpt-4o")
        result = test_single_model(
            model_info=info,
            base_url="https://api.openai.com/v1",
            token="wrong-token",
            test_prompt="OK",
            timeout=5,
            ignored_models=[],
        )
        self.assertEqual(result.status, "ERROR")
        self.assertEqual(result.status_code, 401)
        self.assertIn("Invalid API key", result.message)

    @patch("requests.post")
    def test_test_single_model_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        info = ModelInfo(id="gpt-4o")
        result = test_single_model(
            model_info=info,
            base_url="https://api.openai.com/v1",
            token="test-token",
            test_prompt="OK",
            timeout=5,
            ignored_models=[],
        )
        self.assertEqual(result.status, "ERROR")
        self.assertIn("Timeout after 5s", result.message)


class TestFetchModels(unittest.TestCase):
    @patch("requests.get")
    def test_fetch_models_openai_format(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "text-embedding-3-small"},
            ]
        }
        mock_get.return_value = mock_response

        models = fetch_models("https://api.openai.com/v1", "token", timeout=5)
        self.assertEqual(len(models), 2)
        model_ids = [m.id for m in models]
        self.assertIn("gpt-4o", model_ids)
        self.assertIn("text-embedding-3-small", model_ids)

    @patch("requests.get")
    def test_fetch_models_ollama_format(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "nomic-embed-text:latest"},
            ]
        }
        mock_get.return_value = mock_response

        models = fetch_models("http://localhost:11434/v1", "", timeout=5)
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].id, "llama3.2:latest")


if __name__ == "__main__":
    unittest.main()
