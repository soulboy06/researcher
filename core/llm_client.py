# -*- coding: utf-8 -*-
"""
High-Availability Asynchronous LLM Client.
Features:
- Native AsyncOpenAI async calling
- Exponential Backoff with Full Jitter for 429/5xx transient error resilience
- Adaptive Reasoning Content (DeepSeek-R1 / o1 / <think> tags) extraction
- Native OpenAI Tool Call mapping
"""
import asyncio
import random
import re
import time
from typing import Any, Dict, List, Optional, Union

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    NotFoundError,
    RateLimitError,
)

from researcher.config import settings, logger
from researcher.schema.message import (
    Message,
    LLMResponse,
    LLMUsage,
    ToolCall,
    ToolCallFunction,
)


class LLMClient:
    """
    Unified asynchronous LLM client with built-in retry, error categorization,
    and CoT reasoning extraction.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        base_retry_delay: Optional[float] = None,
        max_retry_delay: Optional[float] = None,
    ):
        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or settings.llm_base_url
        self.model_name = model_name or settings.llm_model_name
        self.timeout = timeout or settings.llm_timeout
        self.max_retries = max_retries if max_retries is not None else settings.llm_max_retries
        self.base_retry_delay = base_retry_delay or settings.llm_retry_base_delay
        self.max_retry_delay = max_retry_delay or settings.llm_retry_max_delay

        self._client = AsyncOpenAI(
            api_key=self.api_key or "EMPTY",
            base_url=self.base_url,
            timeout=self.timeout,
        )

    async def close(self) -> None:
        """Explicitly close AsyncOpenAI client and underlying connection pool."""
        if hasattr(self, "_client") and self._client:
            try:
                await self._client.close()
            except Exception:
                pass

    def _extract_reasoning_and_clean_content(
        self, raw_content: str, raw_reasoning: Optional[str]
    ) -> tuple[str, Optional[str]]:
        """
        Extract reasoning content and sanitize main content.
        Supports both provider-native reasoning_content (DeepSeek-R1) and
        in-content <think>...</think> tags (vLLM, Ollama, QwQ).
        """
        # Case 1: Provider returned native reasoning_content field
        if raw_reasoning and raw_reasoning.strip():
            return raw_content.strip(), raw_reasoning.strip()

        # Case 2: Reasoning is embedded inside <think> tags in content
        if raw_content and "<think>" in raw_content:
            pattern = r"<think>(.*?)</think>"
            matches = re.findall(pattern, raw_content, flags=re.DOTALL)
            if matches:
                extracted_reasoning = "\n\n".join(m.strip() for m in matches if m.strip())
                cleaned_content = re.sub(pattern, "", raw_content, flags=re.DOTALL).strip()
                return cleaned_content, extracted_reasoning

        return raw_content.strip() if raw_content else "", None

    def _parse_tool_calls(self, raw_tool_calls: Optional[List[Any]]) -> Optional[List[ToolCall]]:
        """Map raw OpenAI tool calls to unified ToolCall models."""
        if not raw_tool_calls:
            return None

        parsed = []
        for tc in raw_tool_calls:
            tc_id = getattr(tc, "id", "")
            tc_type = getattr(tc, "type", "function")
            func_obj = getattr(tc, "function", None)
            if func_obj:
                func_name = getattr(func_obj, "name", "")
                func_args = getattr(func_obj, "arguments", "{}")
                parsed.append(
                    ToolCall(
                        id=tc_id,
                        type=tc_type,
                        function=ToolCallFunction(name=func_name, arguments=func_args),
                    )
                )
        return parsed if parsed else None

    async def chat(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        model: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        tools: Optional[List[Dict[str, Any]]] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Execute chat completion with automatic retry on transient errors.

        Args:
            messages: List of Message objects or OpenAI dicts
            model: Override default model name
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            tools: Native OpenAI tool definitions (JSON Schema)
            stop: Stop sequences
            extra_body: Additional vendor parameters (e.g. {"thinking": {"type": "enabled"}})
            max_tokens: Max completion tokens

        Returns:
            Unified LLMResponse object
        """
        target_model = model or self.model_name
        formatted_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, Message):
                formatted_messages.append(msg.to_openai_dict())
            elif isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        request_payload: Dict[str, Any] = {
            "model": target_model,
            "messages": formatted_messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        if tools:
            request_payload["tools"] = tools
        if stop:
            request_payload["stop"] = stop
        if max_tokens:
            request_payload["max_tokens"] = max_tokens
        if extra_body:
            request_payload["extra_body"] = extra_body

        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Calling LLM [{target_model}] (attempt {attempt + 1}/{self.max_retries}) | base_url: {self.base_url}"
                )

                response = await self._client.chat.completions.create(**request_payload)
                elapsed = time.time() - start_time

                choice = response.choices[0]
                message = choice.message
                raw_content = message.content or ""
                raw_reasoning = getattr(message, "reasoning_content", None)

                # Process reasoning & clean content
                content, reasoning_content = self._extract_reasoning_and_clean_content(
                    raw_content, raw_reasoning
                )

                # Process tool calls
                tool_calls = self._parse_tool_calls(getattr(message, "tool_calls", None))

                # Process token usage
                usage = None
                if getattr(response, "usage", None):
                    usage = LLMUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

                logger.debug(
                    f"LLM call succeeded in {elapsed:.2f}s | Output len: {len(content)} | Has reasoning: {bool(reasoning_content)}"
                )

                return LLMResponse(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                    raw_message=message,
                    usage=usage,
                )

            # ❌ Fatal errors: Do not retry
            except (AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError) as fatal_err:
                logger.error(f"Fatal non-retriable LLM error: {fatal_err}")
                raise fatal_err

            # ✅ Transient errors: Exponential backoff with Full Jitter
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIError) as trans_err:
                logger.warning(
                    f"Transient error on attempt {attempt + 1}/{self.max_retries}: {type(trans_err).__name__} - {trans_err}"
                )

                if attempt == self.max_retries - 1:
                    logger.error(f"All {self.max_retries} retry attempts exhausted.")
                    raise trans_err

                # Full Jitter backoff formula
                delay_cap = min(self.max_retry_delay, self.base_retry_delay * (2 ** attempt))
                sleep_time = random.uniform(0.5, max(1.0, delay_cap))
                logger.info(f"Retrying in {sleep_time:.2f}s (Full Jitter Backoff)...")
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Unexpected error during LLM invocation: {e}")
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(1.0)

        raise RuntimeError("LLM invocation failed unexpectedly without explicit error.")
