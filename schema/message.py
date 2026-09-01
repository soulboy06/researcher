# -*- coding: utf-8 -*-
"""
Unified Message and LLM Response Schema.
Fully aligned with OpenAI API standards while providing native support for
multimodal content, reasoning tokens (DeepSeek-R1 / o1), and tool calls.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Supported message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    # Backward compatibility alias
    FUNCTION = "tool"


class ContentType(str, Enum):
    """Multimodal content type."""
    TEXT = "text"
    IMAGE_URL = "image_url"
    INPUT_AUDIO = "input_audio"


class ContentItem(BaseModel):
    """Multimodal message content item."""
    type: str = ContentType.TEXT.value
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None

    def to_openai_dict(self) -> Dict[str, Any]:
        """Convert to standard OpenAI multimodal dict format."""
        item: Dict[str, Any] = {"type": self.type}
        if self.text is not None:
            item["text"] = self.text
        if self.image_url is not None:
            item["image_url"] = self.image_url
        return item


class ToolCallFunction(BaseModel):
    """Tool function invocation specification."""
    name: str
    arguments: str  # JSON string according to OpenAI specification


class ToolCall(BaseModel):
    """OpenAI Tool Call structure."""
    id: str
    type: str = "function"
    function: ToolCallFunction


class Message(BaseModel):
    """
    Standard message model across all agent layers.
    """
    role: MessageRole
    content: Union[str, List[ContentItem]] = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_content: Optional[str] = None  # DeepSeek-R1 / o1 CoT thinking process

    def to_openai_dict(self) -> Dict[str, Any]:
        """Convert message to standard OpenAI dictionary format."""
        msg: Dict[str, Any] = {"role": self.role.value}

        # Content serialization
        if isinstance(self.content, list):
            msg["content"] = [item.to_openai_dict() for item in self.content]
        else:
            msg["content"] = self.content

        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]

        return msg

    @classmethod
    def from_openai_dict(cls, data: Dict[str, Any]) -> "Message":
        """Construct Message from OpenAI response or prompt dictionary."""
        role = MessageRole(data.get("role", MessageRole.USER.value))
        raw_content = data.get("content", "")

        content: Union[str, List[ContentItem]] = ""
        if isinstance(raw_content, list):
            content = [
                ContentItem(
                    type=item.get("type", "text"),
                    text=item.get("text"),
                    image_url=item.get("image_url")
                )
                for item in raw_content
            ]
        else:
            content = str(raw_content or "")

        # Handle tool calls
        tool_calls: Optional[List[ToolCall]] = None
        if "tool_calls" in data and data["tool_calls"]:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=ToolCallFunction(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", "{}")
                    )
                )
                for tc in data["tool_calls"]
            ]

        return cls(
            role=role,
            content=content,
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=tool_calls,
            reasoning_content=data.get("reasoning_content")
        )

    def extract_text(self) -> str:
        """Extract pure text representation of the message content."""
        if isinstance(self.content, str):
            return self.content
        return " ".join([item.text for item in self.content if item.text])


class LLMUsage(BaseModel):
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """
    Unified standard LLM call response across all models.
    """
    content: str = ""
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    raw_message: Optional[Any] = None
    usage: Optional[LLMUsage] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls and len(self.tool_calls) > 0)
