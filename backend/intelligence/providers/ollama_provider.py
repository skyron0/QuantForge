import time
import re
from typing import Dict, Any, Optional
import requests
import json

from backend.intelligence.providers.base import BaseAIProvider
from backend.intelligence.models import AIRequest, AIResponse, ProviderHealth
from backend.intelligence.exceptions import (
    ProviderUnavailableError,
    ProviderTimeoutError,
    ProviderResponseError,
    ProviderConfigurationError,
)


class OllamaProvider(BaseAIProvider):
    """
    Concrete implementation of BaseAIProvider targeting a local or remote Ollama instance.
    """

    def __init__(self, config: Dict[str, Any]):
        self._provider_name = "ollama"
        model_val = config.get("model_name")
        if not model_val:
            raise ProviderConfigurationError(
                "model_name is not configured. AI_MODEL must be explicitly set."
            )
        self._model_name: str = str(model_val)
        self._base_url = config.get("base_url", "http://localhost:11434").rstrip("/")
        self._conn_timeout = float(config.get("connection_timeout", 5.0))
        self._inf_timeout = float(config.get("inference_timeout", 30.0))

    def health_check(self) -> ProviderHealth:
        """
        Check connectivity with Ollama server and optionally verify if the model is locally installed.
        """
        start = time.perf_counter()
        try:
            # 1. Check if Ollama service is reachable
            resp = requests.get(
                f"{self._base_url}/",
                timeout=self._conn_timeout
            )
            if resp.status_code != 200:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return ProviderHealth(
                    available=False,
                    provider=self._provider_name,
                    model=self._model_name,
                    latency_ms=elapsed_ms,
                    error=f"Unhealthy root status: {resp.status_code}"
                )

            # 2. Check if the model is installed locally
            model_exists = False
            tags_resp = requests.get(
                f"{self._base_url}/api/tags",
                timeout=self._conn_timeout
            )
            if tags_resp.status_code == 200:
                data = tags_resp.json()
                models = data.get("models", [])
                # Compare full match or base match (ignoring tag e.g. "llama3:latest" vs "llama3")
                model_names = []
                for m in models:
                    m_name = m.get("name", "")
                    model_names.append(m_name.lower())
                    if ":" in m_name:
                        model_names.append(m_name.split(":")[0].lower())

                target_lower = self._model_name.lower()
                if target_lower in model_names:
                    model_exists = True

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not model_exists:
                return ProviderHealth(
                    available=False,
                    provider=self._provider_name,
                    model=self._model_name,
                    latency_ms=elapsed_ms,
                    error=f"Model '{self._model_name}' is not downloaded/running in Ollama."
                )

            return ProviderHealth(
                available=True,
                provider=self._provider_name,
                model=self._model_name,
                latency_ms=elapsed_ms
            )

        except requests.exceptions.Timeout as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return ProviderHealth(
                available=False,
                provider=self._provider_name,
                model=self._model_name,
                latency_ms=elapsed_ms,
                error=f"Health check connection timeout: {str(e)}"
            )
        except requests.exceptions.RequestException as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return ProviderHealth(
                available=False,
                provider=self._provider_name,
                model=self._model_name,
                latency_ms=elapsed_ms,
                error=f"Health check connection error: {str(e)}"
            )

    def generate(self, request: AIRequest) -> AIResponse:
        """Send a standard text generation request to the Ollama /api/generate endpoint."""
        return self._send_request(request, format_type=None)

    def generate_structured(self, request: AIRequest) -> AIResponse:
        """Send a structured JSON generation request by adding "format": "json" to the Ollama payload."""
        return self._send_request(request, format_type="json")

    def _send_request(self, request: AIRequest, format_type: Optional[str]) -> AIResponse:
        url = f"{self._base_url}/api/generate"
        options = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        payload = {
            "model": self._model_name,
            "prompt": request.user_prompt,
            "system": request.system_prompt,
            "stream": False,
            "options": options
        }

        if format_type:
            payload["format"] = format_type

        start = time.perf_counter()
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=(self._conn_timeout, self._inf_timeout)
            )

            latency_ms = (time.perf_counter() - start) * 1000.0

            if resp.status_code != 200:
                raise ProviderResponseError(
                    f"Ollama returned bad status code: {resp.status_code}. Response: {resp.text}"
                )

            data = resp.json()
            response_text = data.get("response", "")
            thinking_text = data.get("thinking", "")

            # Thinking models (e.g. qwen3.5) may place output in `thinking`
            # field while `response` is empty. Use both fields for content.
            content = response_text if response_text else thinking_text

            # If JSON output is requested, attempt to parse the content string
            structured_output = None
            if format_type == "json":
                # Try all available content sources for JSON extraction
                candidates = [response_text, thinking_text]
                for candidate in candidates:
                    if not candidate:
                        continue
                    cleaned = self._extract_json_content(candidate)
                    if cleaned:
                        try:
                            structured_output = json.loads(cleaned)
                            content = cleaned  # Use the successfully parsed content
                            break
                        except json.JSONDecodeError:
                            continue

            return AIResponse(
                request_id=request.request_id,
                provider=self._provider_name,
                model=self._model_name,
                content=content,
                structured_output=structured_output,
                latency_ms=latency_ms,
                success=True
            )

        except requests.exceptions.Timeout as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            raise ProviderTimeoutError(f"Ollama request timed out: {str(e)}") from e
        except requests.exceptions.RequestException as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            raise ProviderUnavailableError(f"Ollama connection error: {str(e)}") from e
        except Exception as e:
            if isinstance(e, (ProviderTimeoutError, ProviderUnavailableError, ProviderResponseError)):
                raise e
            raise ProviderResponseError(f"Unexpected error: {str(e)}") from e

    @staticmethod
    def _extract_json_content(raw: str) -> str:
        """
        Strip common model wrapper artifacts from raw response content before JSON parsing.
        Handles: <think>...</think> blocks, markdown code fences, leading/trailing whitespace.
        """
        cleaned = raw

        # 1. Remove <think>...</think> blocks (qwen3.5 thinking mode)
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

        # 2. Remove markdown code fences (```json ... ``` or ``` ... ```)
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, flags=re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1)

        # 3. Strip leading/trailing whitespace
        cleaned = cleaned.strip()

        # 4. If the cleaned content does not start with '{', try to extract the first JSON object
        if cleaned and not cleaned.startswith("{"):
            brace_start = cleaned.find("{")
            if brace_start != -1:
                cleaned = cleaned[brace_start:]

        return cleaned

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name
