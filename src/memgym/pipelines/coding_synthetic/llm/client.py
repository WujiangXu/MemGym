"""
LiteLLM-based client for Coding-Synthetic pipeline.

Supports all LiteLLM backends:
- OpenAI: gpt-5-mini, gpt-4o, etc.
- Anthropic: claude-3-5-sonnet, etc.
- Local models via SGLang/vLLM: openai/model-name with api_base
- Ollama: ollama/model-name
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from litellm import completion, acompletion
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class LLMClient:
    """
    LiteLLM-based client for memory extraction and conversation generation.

    Provides structured JSON output via response_format parameter.
    Includes retry logic with exponential backoff.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        """
        Initialize LLM client.

        Args:
            model: LLM model name in LiteLLM format
            api_base: API base URL for local models
            api_key: API key (optional, uses env var if not set)
            temperature: Generation temperature
            max_retries: Maximum retry attempts for API calls
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        """Generate empty value based on JSON schema type."""
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number" or schema_type == "integer":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        """Generate empty response matching JSON schema."""
        if "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(
                    prop_schema.get("type", "string"),
                    prop_schema.get("items")
                )

        return result

    def _build_completion_args(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Build arguments for LiteLLM completion call."""
        args = {
            "model": self.model,
            "messages": messages,
        }

        # Only add temperature for models that support it
        # gpt-5 models don't support custom temperature
        if "gpt-5" not in self.model.lower():
            args["temperature"] = temperature or self.temperature

        if max_tokens:
            args["max_completion_tokens"] = max_tokens

        if response_format:
            args["response_format"] = response_format
        if self.api_base:
            args["api_base"] = self.api_base
        if self.api_key:
            args["api_key"] = self.api_key

        return args

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _call_completion(self, args: dict) -> str:
        """Make completion call with retry logic."""
        response = completion(**args)
        return response.choices[0].message.content

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Get completion from LLM.

        Args:
            prompt: User prompt
            response_format: JSON schema for structured output
            temperature: Override default temperature
            system_prompt: Optional system prompt

        Returns:
            LLM response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif response_format:
            messages.append({"role": "system", "content": "You must respond with a valid JSON object."})

        messages.append({"role": "user", "content": prompt})

        try:
            args = self._build_completion_args(
                messages, response_format, temperature, max_tokens=max_tokens,
            )
            return self._call_completion(args)
        except Exception as e:
            print(f"LiteLLM completion error: {e}")
            if response_format:
                empty_response = self._generate_empty_response(response_format)
                return json.dumps(empty_response)
            return "{}"

    def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Get chat completion from LLM with custom messages.

        Args:
            messages: List of message dicts with role and content
            response_format: JSON schema for structured output
            temperature: Override default temperature

        Returns:
            LLM response text
        """
        try:
            args = self._build_completion_args(messages, response_format, temperature)
            return self._call_completion(args)
        except Exception as e:
            print(f"LiteLLM chat completion error: {e}")
            if response_format:
                empty_response = self._generate_empty_response(response_format)
                return json.dumps(empty_response)
            return "{}"

    def get_json_completion(
        self,
        prompt: str,
        response_format: Dict,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Get completion and parse as JSON.

        Args:
            prompt: User prompt
            response_format: JSON schema for structured output
            temperature: Override default temperature
            system_prompt: Optional system prompt

        Returns:
            Parsed JSON response as dict
        """
        response = self.get_completion(
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

        return self._extract_json(response, response_format)

    @staticmethod
    def _extract_json(text: str, response_format: Optional[Dict] = None) -> dict:
        """Extract JSON from raw text, handling markdown code blocks."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        # Fallback: empty response matching schema
        if response_format:
            return LLMClient._generate_empty_response_static(response_format)
        return {}

    @staticmethod
    def _generate_empty_response_static(response_format: dict) -> dict:
        """Generate empty response matching JSON schema (static version)."""
        if "json_schema" not in response_format:
            return {}
        schema = response_format["json_schema"]["schema"]
        result = {}
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                ptype = prop_schema.get("type", "string")
                if ptype == "array":
                    result[prop_name] = []
                elif ptype == "string":
                    result[prop_name] = ""
                elif ptype in ("number", "integer"):
                    result[prop_name] = 0
                elif ptype == "boolean":
                    result[prop_name] = False
                elif ptype == "object":
                    result[prop_name] = {}
        return result

    def get_completion_messages(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Complete a multi-turn conversation (list of message dicts)."""
        args = self._build_completion_args(
            messages, temperature=temperature, max_tokens=max_tokens,
        )
        try:
            return self._call_completion(args)
        except Exception as e:
            print(f"LiteLLM chat completion error: {e}")
            return "{}"

    # -------------------------------------------------------------------------
    # Async API
    # -------------------------------------------------------------------------

    async def _call_completion_async(self, args: dict) -> str:
        """Async completion with retry."""
        for attempt in range(self.max_retries):
            try:
                response = await acompletion(**args)
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"LLM async completion error after {self.max_retries} retries: {e}")
                    return "{}"
                wait = min(30, 2 ** (attempt + 1))
                await asyncio.sleep(wait)
        return "{}"

    async def get_completion_async(
        self,
        prompt: str,
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Async version of get_completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif response_format:
            messages.append({"role": "system", "content": "You must respond with a valid JSON object."})
        messages.append({"role": "user", "content": prompt})
        args = self._build_completion_args(
            messages, response_format, temperature, max_tokens=max_tokens,
        )
        return await self._call_completion_async(args)

    async def get_json_completion_async(
        self,
        prompt: str,
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Async version of get_json_completion."""
        response = await self.get_completion_async(
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        return self._extract_json(response, response_format)


class VerifierClient(LLMClient):
    """Wrapper that uses the expensive/strong Verifier model.

    Defaults to claude-opus-4-20250514 but can be overridden.
    Inherits all LLMClient methods.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-20250514",
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
