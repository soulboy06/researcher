# -*- coding: utf-8 -*-
"""
Secondary Fact & Evidence Extractor Tool.
Distills dense key evidence and structured summaries from lengthy webpage contents.
"""
from typing import Any, Dict, List, Optional, Union
import json

from researcher.config import settings, logger
from researcher.schema.message import Message, MessageRole
from researcher.core.llm_client import LLMClient
from researcher.core.context_guard import context_guard
from researcher.tools.base import BaseTool


EXTRACTOR_SYSTEM_PROMPT = """你是一个顶尖的研究证据提炼专家。
你的任务是从用户提供的网页长文或资料中，紧紧围绕用户的【研究目标 (Goal)】提炼出最核心的事实依据。

请按照以下 JSON 格式输出，不要输出任何多余的解释说明：
```json
{
  "summary": "简明扼要的核心结论（100-200字）",
  "evidence_chunks": [
    "直接支持研究目标的事实片段1（保留原文关键数据、时间、人物、结论）",
    "直接支持研究目标的事实片段2"
  ]
}
```

提炼准则：
1. 剔除所有与 Goal 无关的冗余叙述、广告、导航信息；
2. 严禁捏造事实，保留原文中的具体数据、年份、专有名词与核心逻辑；
3. 如果原文完全不包含与 Goal 相关的信息，请在 summary 中明确说明。
"""


class EvidenceExtractorTool(BaseTool):
    """
    Secondary Fact & Evidence Extractor Tool.
    Uses a lightweight/fast LLM to extract high-density evidence from long documents.
    """

    name = "extract_evidence"
    description = (
        "Extracts high-density facts and key evidence from a long document or webpage content "
        "tailored specifically to a given research goal."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The raw document or webpage text content to distill.",
            },
            "goal": {
                "type": "string",
                "description": "The specific research goal or question to guide extraction.",
            },
            "max_chunks": {
                "type": "integer",
                "description": "Maximum number of key evidence chunks to extract (default 5).",
            },
        },
        "required": ["content", "goal"],
    }

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model_name: Optional[str] = None,
    ):
        # Auto-detect target model: fallback to main LLM if summary model is default gpt-4o-mini on non-OpenAI endpoints
        target_model = model_name
        if not target_model:
            if "openai.com" not in settings.llm_base_url and settings.summary_model_name == "gpt-4o-mini":
                target_model = settings.llm_model_name
            else:
                target_model = settings.summary_model_name or settings.llm_model_name

        self.model_name = target_model
        self.llm_client = llm_client

    async def _execute_extraction(
        self, client: LLMClient, params: Union[Dict[str, Any], str], **kwargs
    ) -> str:
        """Internal worker executing LLM extraction prompt."""
        if isinstance(params, str):
            content = params
            goal = kwargs.get("goal", "Summarize key findings")
            max_chunks = kwargs.get("max_chunks", 5)
        else:
            content = params.get("content", "")
            goal = params.get("goal", "Summarize key findings")
            max_chunks = params.get("max_chunks", 5)

        if not content.strip():
            return "[Extractor Error]: Empty content provided for extraction."

        # Guard content length to prevent prompt blowout
        truncated_content = context_guard.truncate_observation(content, max_chars=40000)

        user_prompt = (
            f"【研究目标 (Goal)】\n{goal}\n\n"
            f"【待提炼正文资料 (Content)】\n{truncated_content}\n\n"
            f"请根据上述目标提炼，最多返回 {max_chunks} 条核心证据片段。"
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content=EXTRACTOR_SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=user_prompt),
        ]

        try:
            response = await client.chat(
                messages=messages,
                model=self.model_name,
                temperature=0.2,
            )

            raw_text = response.content.strip()
            # Clean possible markdown json codeblock
            if "```json" in raw_text:
                raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```", 1)[1].split("```", 1)[0].strip()

            try:
                data = json.loads(raw_text)
                summary = data.get("summary", "")
                chunks = data.get("evidence_chunks", [])

                formatted = [f"### Extracted Summary for Goal: {goal}\n{summary}\n"]
                if chunks:
                    formatted.append("#### Key Evidence Chunks:")
                    for idx, chunk in enumerate(chunks[:max_chunks], 1):
                        formatted.append(f"{idx}. {chunk}")
                return "\n".join(formatted)
            except json.JSONDecodeError:
                # Direct text fallback if model outputted plain text
                return f"### Extracted Summary:\n{raw_text}"

        except Exception as e:
            logger.error(f"[EvidenceExtractorTool] Extraction failed: {e}")
            return f"[Extractor Fallback]: Unable to run LLM extraction ({e}). Raw snippet:\n{truncated_content[:1500]}"

    def call(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        """Synchronous wrapper for evidence extraction with clean connection lifecycle."""
        import asyncio

        async def _run_and_cleanup():
            client = self.llm_client or LLMClient(model_name=self.model_name)
            try:
                return await self._execute_extraction(client, params, **kwargs)
            finally:
                if not self.llm_client:
                    await client.close()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _run_and_cleanup()).result()
        else:
            return asyncio.run(_run_and_cleanup())

    async def acall(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        """Asynchronously extract evidence from content based on goal."""
        client = self.llm_client or LLMClient(model_name=self.model_name)
        try:
            return await self._execute_extraction(client, params, **kwargs)
        finally:
            if not self.llm_client:
                await client.close()
