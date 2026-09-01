# -*- coding: utf-8 -*-
"""
WebWeaver Outline Planner Agent.
Conducts broad web exploration, populates the shared MemoryBank with facts,
and dynamically constructs a structured research outline annotated with [cite:id_N] citation tags.
"""
import time
from typing import Any, Callable, Dict, List, Optional

from researcher.config import logger
from researcher.schema.message import Message, MessageRole
from researcher.core.llm_client import LLMClient
from researcher.core.context_guard import context_guard
from researcher.core.prompt import get_webweaver_planner_prompt
from researcher.core.protocol_parser import ProtocolParser, ParsedOutput
from researcher.tools.base import ToolRegistry
from researcher.tools.memory_bank import MemoryBank
from researcher.tools.planner_tools import PlannerSearchTool, PlannerVisitTool, PlannerPythonTool
from researcher.agents.base import BaseAgent, AgentResult


class WebWeaverPlanner(BaseAgent):
    """
    Outline Planner Agent that drives exploratory search and dynamically evolves
    the multi-level research outline with linked MemoryBank citation IDs.
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

        # Build Planner's memory-backed tool registry
        if tool_registry is not None:
            tools = tool_registry
        else:
            tools = ToolRegistry()
            tools.register(PlannerSearchTool(self.memory_bank))
            tools.register(PlannerVisitTool(self.memory_bank))
            tools.register(PlannerPythonTool(self.memory_bank))

        super().__init__(
            tool_registry=tools,
            max_rounds=max_rounds,
            timeout=timeout,
            use_xml_protocol=use_xml_protocol,
        )
        self.llm_client = llm_client or LLMClient(model_name=model_name)
        self.instruction = instruction
        self.model_name = model_name or self.llm_client.model_name

    async def run_planner(
        self,
        question: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> str:
        """
        Execute Planner exploration loop.
        
        Returns:
            final_outline (str): The structured research outline with attached [cite:id_N] tags.
        """
        start_time = time.time()
        tool_definitions = self.tools.get_openai_tools()

        system_prompt = get_webweaver_planner_prompt(
            tools=tool_definitions,
            question=question,
            instruction=self.instruction,
        )

        current_outline = "大纲尚在规划中。请先通过检索获取事实依据。"
        last_observation = "尚未执行任何搜索。请发起第一轮探索。"

        round_num = 0
        while round_num < self.max_rounds:
            round_num += 1

            if time.time() - start_time > self.timeout:
                logger.warning(f"[WebWeaverPlanner] Timeout reached ({self.timeout}s). Finalizing outline.")
                break

            logger.info(f"[WebWeaverPlanner] Round {round_num}/{self.max_rounds} starting...")

            is_last_round = (round_num == self.max_rounds)
            extra_prompt = None
            if is_last_round:
                extra_prompt = (
                    "⚠️ **系统提示**：已达到最大大纲规划轮次。严禁再调用工具！"
                    "请输出最终完整的 `<write_outline>...</write_outline>` 并紧随 `<terminate>大纲构建完成</terminate>`。"
                )

            # Build Planner context
            user_content = (
                f"**【研究问题 (Question)】**\n{question}\n\n"
                f"**【当前演化研究大纲 (Current Outline)】**\n{current_outline}\n\n"
                f"**【最新收集事实卡片 (Last Observation)】**\n{last_observation}"
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
                logger.error(f"[WebWeaverPlanner] LLM invocation failed in round {round_num}: {e}")
                break

            if response.reasoning_content:
                await self._emit_event(
                    progress_callback,
                    "thinking",
                    {"round": round_num, "agent": "planner", "content": response.reasoning_content},
                )

            parsed: ParsedOutput = ProtocolParser.parse_iterresearch(response.content)

            # Update outline if model generated <write_outline>
            if parsed.outline:
                current_outline = parsed.outline
                await self._emit_event(
                    progress_callback,
                    "outline",
                    {"round": round_num, "outline": current_outline},
                )

            await self._emit_event(
                progress_callback,
                "plan",
                {
                    "round": round_num,
                    "agent": "planner",
                    "plan": parsed.plan,
                    "outline": current_outline[:300],
                    "tool_name": parsed.tool_name,
                    "terminate": parsed.terminate,
                },
            )

            # Tool execution
            if parsed.has_tool_call:
                tool_name = parsed.tool_name
                tool_args = parsed.tool_args

                logger.info(f"[WebWeaverPlanner] Executing planner tool '{tool_name}' with args: {tool_args}")
                raw_obs = await self.tools.execute(tool_name, tool_args)
                last_observation = context_guard.truncate_observation(raw_obs, max_chars=30000)

                # Emit memory bank state to UI
                bank_items = [
                    {
                        "id": item.id,
                        "title": item.title,
                        "url": item.url,
                        "summary": item.summary[:150] if item.summary else "",
                    }
                    for item in self.memory_bank.get_all()
                ]
                await self._emit_event(
                    progress_callback,
                    "memory_bank_update",
                    {
                        "total_items": len(bank_items),
                        "items": bank_items,
                    },
                )

                await self._emit_event(
                    progress_callback,
                    "tool",
                    {
                        "round": round_num,
                        "agent": "planner",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "observation": last_observation[:500],
                    },
                )

            # Termination check
            if parsed.terminate or is_last_round:
                if self.memory_bank.size() == 0 and round_num < self.max_rounds + 2:
                    logger.warning("[WebWeaverPlanner] Attempted to terminate with 0 evidence items. Forcing search...")
                    last_observation = (
                        "⚠️ 警告：当前证据库 (MemoryBank) 中尚无任何有效证据（0 条）！"
                        "你严禁在没有证据时直接结束大纲。请立即调用 `<tool_call>` 发起 `search` 检索该课题的核心背景与事实！"
                    )
                    continue

                logger.info(f"[WebWeaverPlanner] Outline completed in round {round_num}. Total evidence items: {self.memory_bank.size()}")
                break

            if not parsed.has_tool_call:
                # If model forgot action tag, prompt it to supply tool call or finalize
                last_observation = "提示：已保存大纲。若需补充事实请输出 <tool_call>，若大纲已完备请输出 <terminate>。"

        return current_outline

    async def run(
        self,
        question: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> AgentResult:
        """Standard BaseAgent execution wrapper."""
        outline = await self.run_planner(question, progress_callback)
        return AgentResult(
            question=question,
            prediction=outline,
            report=outline,
            trajectory=[],
            termination="outline_completed",
        )
