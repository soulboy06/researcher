# -*- coding: utf-8 -*-
"""
Context Guard and Token Management Module.
Provides:
1. BPE Token Counting with Tiktoken (supports model-aware encoding and cl100k_base fallback)
2. Observation Truncation Guard with Head 75% + Tail 25% weighted preservation
3. Context Token Budget Monitoring and Warning Alerts
"""
from typing import Any, Dict, List, Optional, Union
import re
from researcher.config import settings, logger
from researcher.schema.message import Message, ContentItem


class TokenCounter:
    """
    Precision Token Counter using BPE encoding.
    """

    def __init__(self, default_model: str = "gpt-4o"):
        self.default_model = default_model
        self._encoders: Dict[str, Any] = {}

    def _get_encoder(self, model: Optional[str] = None):
        """Retrieve or cache tiktoken encoding for model."""
        target_model = model or self.default_model
        if target_model in self._encoders:
            return self._encoders[target_model]

        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(target_model)
            except (KeyError, ValueError):
                enc = tiktoken.get_encoding("cl100k_base")
            self._encoders[target_model] = enc
            return enc
        except ImportError:
            return None

    def count_string(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in a text string.
        Falls back to whitespace/character estimation if tiktoken is not installed.
        """
        if not text:
            return 0

        enc = self._get_encoder(model)
        if enc is not None:
            try:
                return len(enc.encode(str(text)))
            except Exception:
                pass

        # Fallback estimation for Chinese and English mixed text
        # CJK characters are ~1.5 tokens each, English words ~1.3 tokens
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        non_cjk = re.sub(r"[\u4e00-\u9fff]", " ", text)
        words = len(non_cjk.split())
        return max(1, int(cjk_chars * 1.5 + words * 1.3))

    def count_messages(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        model: Optional[str] = None,
    ) -> int:
        """
        Count total tokens in a multi-turn message history.
        Includes role prefixes and formatting overhead.
        """
        total = 0
        for msg in messages:
            # Add message envelope overhead (~4 tokens per message)
            total += 4
            if isinstance(msg, Message):
                total += self.count_string(msg.role, model)
                total += self.count_string(msg.extract_text(), model)
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        total += self.count_string(tc.function.name, model)
                        total += self.count_string(tc.function.arguments, model)
            elif isinstance(msg, dict):
                role = str(msg.get("role", ""))
                content = msg.get("content", "")
                total += self.count_string(role, model)
                if isinstance(content, str):
                    total += self.count_string(content, model)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            total += self.count_string(item["text"], model)
                if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                    for tc in msg["tool_calls"]:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            total += self.count_string(fn.get("name", ""), model)
                            total += self.count_string(fn.get("arguments", ""), model)
        total += 3  # Assistant reply priming overhead
        return total


# Global token counter instance
token_counter = TokenCounter()


class ContextGuard:
    """
    Context Guard to prevent context window overflows and preserve key information.
    """

    def __init__(
        self,
        max_chars: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        head_ratio: float = 0.75,
    ):
        self.max_chars = max_chars or settings.obs_max_chars
        self.max_input_tokens = max_input_tokens or settings.max_input_tokens
        self.head_ratio = head_ratio
        self.tail_ratio = 1.0 - head_ratio

    def truncate_observation(
        self,
        observation: str,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Truncate tool observation using Head 75% + Tail 25% weighted preservation.
        
        Args:
            observation: Raw tool observation string
            max_chars: Character limit override
            
        Returns:
            Sanitized and truncated observation string
        """
        if not observation:
            return ""

        limit = max_chars or self.max_chars
        if len(observation) <= limit:
            return observation

        head_len = int(limit * self.head_ratio)
        tail_len = int(limit * self.tail_ratio)

        head = observation[:head_len]
        tail = observation[-tail_len:] if tail_len > 0 else ""

        truncated_count = len(observation) - (head_len + tail_len)
        tag = f"\n\n...[ContextGuard: Truncated {truncated_count} characters of intermediate content]...\n\n"

        result = head + tag + tail
        logger.warning(
            f"[ContextGuard] Observation truncated: {len(observation)} -> {len(result)} chars (dropped {truncated_count} chars)"
        )
        return result

    def check_context_budget(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        model: Optional[str] = None,
        warning_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Inspect current token budget and alert if approaching threshold.
        
        Returns:
            Dict containing:
            - current_tokens: int
            - max_tokens: int
            - usage_ratio: float
            - is_warning: bool
        """
        current_tokens = token_counter.count_messages(messages, model)
        usage_ratio = current_tokens / max(1, self.max_input_tokens)
        is_warning = usage_ratio >= warning_threshold

        if is_warning:
            logger.warning(
                f"[ContextGuard] High token usage warning: {current_tokens}/{self.max_input_tokens} ({usage_ratio*100:.1f}%)"
            )

        return {
            "current_tokens": current_tokens,
            "max_tokens": self.max_input_tokens,
            "usage_ratio": usage_ratio,
            "is_warning": is_warning,
        }


# Global context guard instance
context_guard = ContextGuard()
