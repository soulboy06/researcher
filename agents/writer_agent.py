# -*- coding: utf-8 -*-
"""
WebWeaver Citation Writer Agent.
Consumes the structured outline produced by the Planner, dynamically retrieves
raw evidence chunks from the MemoryBank using RetrieveTool, and writes detailed
publication-grade sections embedded with strict inline citations ([cite:id_N]).
"""
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set

from researcher.config import logger
from researcher.schema.message import Message, MessageRole
from researcher.core.llm_client import LLMClient
from researcher.core.context_guard import context_guard
from researcher.core.prompt import get_webweaver_writer_prompt
from researcher.core.protocol_parser import ProtocolParser, ParsedOutput
from researcher.tools.base import ToolRegistry
from researcher.tools.memory_bank import MemoryBank, RetrieveTool
from researcher.agents.base import BaseAgent, AgentResult


class WebWeaverWriter(BaseAgent):
    """
    Citation Writer Agent that composes deep, evidence-grounded research reports
    section by section using targeted MemoryBank retrieval.
    """

    def __init__(
        self,
        memory_bank: MemoryBank,
        llm_client: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_rounds: int = 6,
        timeout: float = 300.0,
        use_xml_protocol: bool = True,
        instruction: str = "",
        model_name: Optional[str] = None,
    ):
        self.memory_bank = memory_bank

        # Writer only gets the RetrieveTool
        if tool_registry is not None:
            tools = tool_registry
        else:
            tools = ToolRegistry()
            tools.register(RetrieveTool(self.memory_bank))

        super().__init__(
            tool_registry=tools,
            max_rounds=max_rounds,
            timeout=timeout,
            use_xml_protocol=use_xml_protocol,
        )
        self.llm_client = llm_client or LLMClient(model_name=model_name)
        self.instruction = instruction
        self.model_name = model_name or self.llm_client.model_name

    def _extract_write_block(self, text: str) -> str:
        """
        Extract written content from LLM response with 4-tier fallback parsing.
        """
        if not text:
            return ""

        # Tier 1: Explicit <write>...</write>
        match = re.search(r"<\s*write\s*>(.*?)<\s*/\s*write\s*>", text, flags=re.DOTALL)
        if match and len(match.group(1).strip()) > 30:
            return match.group(1).strip()

        # Tier 2: <answer>...</answer> or <report>...</report>
        ans_match = re.search(r"<\s*(?:answer|report)\s*>(.*?)<\s*/\s*(?:answer|report)\s*>", text, flags=re.DOTALL)
        if ans_match and len(ans_match.group(1).strip()) > 50:
            return ans_match.group(1).strip()

        # Tier 3: Strip <plan>, <tool_call>, <terminate> and keep clean markdown
        cleaned = re.sub(r"<\s*plan\s*>.*?<\s*/\s*plan\s*>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<\s*tool_call\s*>.*?<\s*/\s*tool_call\s*>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<\s*terminate\s*>.*?<\s*/\s*terminate\s*>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<\s*thought\s*>.*?<\s*/\s*thought\s*>", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()

        if len(cleaned) > 80:
            return cleaned

        return ""

    async def run_writer(
        self,
        question: str,
        final_outline: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> str:
        """
        Execute Writer composing loop.
        
        Returns:
            draft_report (str): Complete written report with raw [cite:id_N] tags.
        """
        start_time = time.time()
        system_prompt = get_webweaver_writer_prompt(
            question=question,
            instruction=self.instruction,
        )

        report_written_so_far = ""
        last_observation = "尚未开始检索证据。请先查看大纲第一章节对应的证据 ID，调用 retrieve 进行拉取。"

        seen_retrievals: Set[str] = set()
        round_num = 0

        while round_num < self.max_rounds:
            round_num += 1

            if time.time() - start_time > self.timeout:
                logger.warning(f"[WebWeaverWriter] Timeout reached ({self.timeout}s). Concluding report.")
                break

            logger.info(f"[WebWeaverWriter] Round {round_num}/{self.max_rounds} starting...")

            is_last_round = (round_num == self.max_rounds)
            extra_prompt = None
            if is_last_round:
                extra_prompt = (
                    "⚠️ **系统强制收尾提示**：已达到最大撰写轮次。严禁再调用 retrieve！"
                    "请将整篇研报剩余各章节完整展开写完，并在 `<write>...</write>` 后输出 `<terminate>撰写完成</terminate>`。"
                )

            # Build Writer context
            user_content = (
                f"**【研究问题 (Question)】**\n{question}\n\n"
                f"**【最终精细研究大纲 (Final Outline)】**\n{final_outline}\n\n"
                f"**【已撰写报告正文 (Report Written So Far)】**\n"
                f"{report_written_so_far if report_written_so_far else '(尚未撰写任何章节)'}\n\n"
                f"**【最新检索到的证据内容 (Last Observation)】**\n{last_observation}"
            )
            if extra_prompt:
                user_content += f"\n\n{extra_prompt}"

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_content),
            ]

            try:
                response = await self.llm_client.chat(
                    messages=messages,
                    model=self.model_name,
                    temperature=0.2,
                )
            except Exception as e:
                logger.error(f"[WebWeaverWriter] LLM invocation failed in round {round_num}: {e}")
                break

            if response.reasoning_content:
                await self._emit_event(
                    progress_callback,
                    "thinking",
                    {"round": round_num, "agent": "writer", "content": response.reasoning_content},
                )

            # Parse writer output
            written_section = self._extract_write_block(response.content)
            parsed: ParsedOutput = ProtocolParser.parse_iterresearch(response.content)

            # Accumulate newly written section
            if written_section:
                if report_written_so_far:
                    report_written_so_far += f"\n\n{written_section}"
                else:
                    report_written_so_far = written_section

                await self._emit_event(
                    progress_callback,
                    "write",
                    {
                        "round": round_num,
                        "agent": "writer",
                        "section": written_section,
                        "total_written": report_written_so_far,
                        "total_length": len(report_written_so_far),
                    },
                )

            await self._emit_event(
                progress_callback,
                "plan",
                {
                    "round": round_num,
                    "agent": "writer",
                    "plan": parsed.plan,
                    "tool_name": parsed.tool_name,
                    "terminate": parsed.terminate,
                },
            )

            # Check termination
            if parsed.terminate or is_last_round:
                logger.info(f"[WebWeaverWriter] Report writing loop ended in round {round_num}. Length: {len(report_written_so_far)}")
                break

            # Execute retrieve tool
            if parsed.has_tool_call:
                tool_name = parsed.tool_name
                tool_args = parsed.tool_args

                # Loop breaker for repeated retrievals
                retrieval_key = str(sorted(tool_args.get("citation_ids", [])))
                if retrieval_key in seen_retrievals and len(seen_retrievals) > 0:
                    logger.warning(f"[WebWeaverWriter] Detected repeated retrieval for {retrieval_key}. Forcing writing.")
                    last_observation = (
                        "提示：你已获取过上述证据，请立刻使用 `<write>## 章节标题\n章节正文详细阐述... [cite:id_N]</write>` 撰写正文，严禁重复检索！"
                    )
                else:
                    seen_retrievals.add(retrieval_key)
                    logger.info(f"[WebWeaverWriter] Executing writer tool '{tool_name}' with args: {tool_args}")
                    raw_obs = await self.tools.execute(tool_name, tool_args)
                    last_observation = context_guard.truncate_observation(raw_obs, max_chars=30000)

                await self._emit_event(
                    progress_callback,
                    "tool",
                    {
                        "round": round_num,
                        "agent": "writer",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "observation": last_observation[:400],
                    },
                )
            else:
                last_observation = "提示：已记录已写内容。请继续检索下一章节的证据 ID 或直接使用 `<write>` 撰写后续章节。"

        # ===================================================================
        # Guaranteed Fallback: If written content is missing or too short, synthesize all chapters
        # ===================================================================
        if len(report_written_so_far.strip()) < 300:
            logger.warning(f"[WebWeaverWriter] Report too short ({len(report_written_so_far)} chars). Executing guaranteed full synthesis fallback...")
            
            all_evidence = self.memory_bank.get_all()
            evidence_blocks = []
            for ev in all_evidence[:20]:
                content_sample = ev.content[:1500] if ev.content else ev.summary
                evidence_blocks.append(f"【证据 [{ev.id}] {ev.title}】\n{content_sample}")
            
            evidence_text = "\n\n".join(evidence_blocks)

            fallback_prompt = (
                f"**【研究问题】**\n{question}\n\n"
                f"**【精细大纲】**\n{final_outline}\n\n"
                f"**【全部事实证据库 (MemoryBank)】**\n{evidence_text}\n\n"
                "请严格依据上述大纲与全部事实证据，一次性完整撰写出整篇结构严谨、内容详实的学术长篇研究报告。"
                "要求：\n"
                "1. 每个章节都必须充分展开论述，严禁只有标题或简要概述！\n"
                "2. 每一个核心事实与数据必须标注对应的引用标签，例如 `[cite:id_1]`、`[cite:id_2]`；\n"
                "3. 将完整长文正文包裹在 `<write>...</write>` 标签中。"
            )

            try:
                resp = await self.llm_client.chat(
                    messages=[
                        Message(role=MessageRole.SYSTEM, content=system_prompt),
                        Message(role=MessageRole.USER, content=fallback_prompt),
                    ],
                    model=self.model_name,
                    temperature=0.2,
                )
                fallback_content = self._extract_write_block(resp.content) or resp.content.strip()
                if len(fallback_content) > len(report_written_so_far):
                    report_written_so_far = fallback_content
                    await self._emit_event(
                        progress_callback,
                        "write",
                        {
                            "round": round_num + 1,
                            "agent": "writer",
                            "section": report_written_so_far,
                            "total_written": report_written_so_far,
                            "total_length": len(report_written_so_far),
                        },
                    )
            except Exception as e:
                logger.error(f"[WebWeaverWriter] Guaranteed synthesis fallback failed: {e}")

        return report_written_so_far

    async def run(
        self,
        question: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> AgentResult:
        """Standard BaseAgent execution entrypoint."""
        draft = await self.run_writer(question, question, progress_callback=progress_callback)
        return AgentResult(
            question=question,
            prediction=draft,
            report=draft,
            termination="writer_completed",
        )
