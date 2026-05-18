"""LiteLLM-based client for MemGym-IR v2 pipeline.

Worker (cheap) and Verifier (strong) LLM clients.
Supports both sync and async operations for parallel processing.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import litellm
from litellm import completion, acompletion

litellm.modify_params = True
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)


class LLMClient:
    """LiteLLM-based client with structured JSON output and retry logic."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 8,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries

    def _build_args(
        self,
        prompt: str,
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Build completion args (shared by sync and async)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif response_format:
            messages.append(
                {"role": "system", "content": "You must respond with valid JSON."}
            )
        messages.append({"role": "user", "content": prompt})

        args = {"model": self.model, "messages": messages}
        if temperature is not None:
            args["temperature"] = temperature
        else:
            args["temperature"] = self.temperature
        if max_tokens:
            args["max_tokens"] = max_tokens
        if response_format:
            args["response_format"] = response_format
        if self.api_base:
            args["api_base"] = self.api_base
        if self.api_key:
            args["api_key"] = self.api_key
        return args

    # -------------------------------------------------------------------------
    # Sync API
    # -------------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential_jitter(initial=4, max=120, jitter=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _call_completion(self, args: dict) -> str:
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
        args = self._build_args(prompt, response_format, temperature, system_prompt, max_tokens)
        try:
            return self._call_completion(args)
        except Exception as e:
            print(f"LLM completion error: {e}")
            return "{}"

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from raw text, handling markdown code blocks."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        return {}

    def get_completion_messages(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Complete a multi-turn conversation (list of message dicts)."""
        args = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if max_tokens:
            args["max_tokens"] = max_tokens
        if self.api_base:
            args["api_base"] = self.api_base
        if self.api_key:
            args["api_key"] = self.api_key
        try:
            return self._call_completion(args)
        except Exception as e:
            print(f"LLM completion error: {e}")
            return "{}"

    def get_json_completion(
        self,
        prompt: str,
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        response = self.get_completion(
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        return self._extract_json(response)

    # -------------------------------------------------------------------------
    # Async API
    # -------------------------------------------------------------------------

    async def _call_completion_async(self, args: dict) -> str:
        """Async completion with retry and jittered exponential backoff."""
        import random
        for attempt in range(self.max_retries):
            try:
                response = await acompletion(**args)
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"LLM async completion error after {self.max_retries} retries: {e}")
                    return "{}"
                wait = min(120, 4 * (2 ** attempt)) + random.uniform(0, 10)
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
        args = self._build_args(prompt, response_format, temperature, system_prompt, max_tokens)
        return await self._call_completion_async(args)

    async def get_json_completion_async(
        self,
        prompt: str,
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        response = await self.get_completion_async(
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        return self._extract_json(response)


class VerifierClient(LLMClient):
    """Wrapper using the expensive/strong Verifier model."""

    def __init__(self, model: str = "gpt-4o", **kwargs):
        super().__init__(model=model, **kwargs)
