# -*- coding: utf-8 -*-
"""
IterResearch Fixed-Context State Machine Agent.
State Space: s_t = (Q, R_{i-1}, O_{i-1})
Executes single-turn 3-part generation per round: <plan> -> <report> -> <action>
Eliminates context bloat, prevents attention dilution, and features true breakpoint resumption.
"""
import time
from typing import Any, Callable, Dict, List, Optional

from researcher.config import settings, logger
from researcher.schema.message import Message, MessageRole
from researcher.core.llm_client import LLMClient
from researcher.core.context_guard import context_guard
from researcher.core.prompt import get_iterresearch_prompt
from researcher.core.protocol_parser import ProtocolParser, ParsedOutput
from researcher.tools.base import ToolRegistry
from researcher.agents.base import BaseAgent, AgentResult


class ResearchWorkspace:
    """
    Fixed-context workspace state manager representing s_t = (Q, R_{i-1}, O_{i-1}).
    """

    def __init__(self, question: str):
        self.question = question
        self.current_report = "当前为第一轮研究，报告草稿为空。"
        self.last_observation = "当前为第一轮，尚未执行任何工具调用。"

    def set_observation(self, observation: str) -> None:
        """Update latest observation O_i with length guard."""
        self.last_observation = context_guard.truncate_observation(observation)

    def get_messages(self, system_prompt: str, extra_user_prompt: Optional[str] = None) -> List[Message]:
        """Construct compact 2-message context for LLM invocation."""
        user_content = (
            f"**【研究问题 (Question)】**\n{self.question}\n\n"
            f"**【上轮沉淀全局报告 (Current Report R_{{i-1}})】**\n{self.current_report}\n\n"
            f"**【最新工具观察结果 (Last Observation O_{{i-1}})】**\n{self.last_observation}"
        )
        if extra_user_prompt:
            user_content += f"\n\n{extra_user_prompt}"

        return [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_content),
        ]


class IterResearchAgent(BaseAgent):
    """
    IterResearch Agent implementing cyclic 3-part knowledge accumulation
    with robust breakpoint continuation.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_rounds: int = 10,
        timeout: float = 600.0,
        use_xml_protocol: bool = True,
        instruction: str = "",
        model_name: Optional[str] = None,
    ):
        super().__init__(
            tool_registry=tool_registry,
            max_rounds=max_rounds,
            timeout=timeout,
            use_xml_protocol=use_xml_protocol,
        )
        self.llm_client = llm_client or LLMClient(model_name=model_name)
        self.instruction = instruction
        self.model_name = model_name or self.llm_client.model_name

    async def run(
        self,
        question: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> AgentResult:
        start_time = time.time()
        tool_definitions = self.tools.get_openai_tools()

        # Build base system prompt
        system_prompt = get_iterresearch_prompt(
            tools=tool_definitions,
            question=question,
            instruction=self.instruction,
        )

        workspace = ResearchWorkspace(question=question)
        trajectory: List[Dict[str, Any]] = []

        total_tokens = 0
        round_num = 0
        final_prediction = ""
        termination_status = "max_rounds_reached"

        while round_num < self.max_rounds:
            round_num += 1

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                logger.warning(f"[IterResearchAgent] Timeout reached ({elapsed:.1f}s > {self.timeout}s)")
                termination_status = "timeout"
                final_prediction = workspace.current_report or "Research stopped due to timeout."
                break

            logger.info(f"[IterResearchAgent] Round {round_num}/{self.max_rounds} starting...")

            # 1. Check if force finalize round
            is_last_round = (round_num == self.max_rounds)
            extra_prompt = None
            if is_last_round:
                extra_prompt = (
                    "⚠️ **系统强制收敛提示**：已达到最大探索轮次。严禁再调用任何工具！"
                    "请根据当前研究报告及已累积的所有事实，直接在 `<answer>...</answer>` 标签中给出最终结论。"
                )

            # 2. Build compact messages s_t = (Q, R_{i-1}, O_{i-1})
            request_msgs = workspace.get_messages(system_prompt, extra_user_prompt=extra_prompt)

            # 3. Call LLM (single turn outputs Plan + Report + Action)
            try:
                response = await self.llm_client.chat(
                    messages=request_msgs,
                    model=self.model_name,
                    temperature=0.3,
                )
            except Exception as e:
                logger.error(f"[IterResearchAgent] LLM call failed in round {round_num}: {e}")
                termination_status = f"llm_error: {e}"
                final_prediction = f"Error occurred: {e}"
                break

            if response.usage:
                total_tokens += response.usage.total_tokens

            # Emit reasoning if captured
            if response.reasoning_content:
                await self._emit_event(
                    progress_callback,
                    "thinking",
                    {"round": round_num, "content": response.reasoning_content},
                )

            # 4. Parse Three-Part Output: <plan>, <report>, (<tool_call> | <answer> | <terminate>)
            parsed: ParsedOutput = ProtocolParser.parse_iterresearch(response.content)

            # Update accumulated research report R_i
            if parsed.report:
                workspace.current_report = parsed.report

            # Emit round progress event
            await self._emit_event(
                progress_callback,
                "round",
                {
                    "round": round_num,
                    "plan": parsed.plan,
                    "report": parsed.report,
                    "tool_call": parsed.tool_call_raw,
                    "answer": parsed.answer,
                    "terminate": parsed.terminate,
                },
            )

            trajectory.append({
                "round": round_num,
                "plan": parsed.plan,
                "report": parsed.report,
                "tool_name": parsed.tool_name,
                "tool_args": parsed.tool_args,
                "answer": parsed.answer,
            })

            # 5. Check Termination Conditions
            if parsed.has_final_answer:
                final_prediction = parsed.answer
                termination_status = "answer_found"
                # If report was a trivial placeholder or shorter than answer, sync to answer
                if len(final_prediction) >= len(workspace.current_report) or len(workspace.current_report) < 100:
                    workspace.current_report = final_prediction

                await self._emit_event(
                    progress_callback,
                    "final",
                    {"round": round_num, "answer": final_prediction, "report": workspace.current_report},
                )
                break

            if parsed.terminate:
                final_prediction = parsed.terminate_reason or workspace.current_report
                termination_status = "terminated_by_agent"
                if len(final_prediction) >= len(workspace.current_report):
                    workspace.current_report = final_prediction

                await self._emit_event(
                    progress_callback,
                    "final",
                    {"round": round_num, "answer": final_prediction, "report": workspace.current_report},
                )
                break

            if is_last_round:
                final_prediction = workspace.current_report
                termination_status = "finalized_on_max_rounds"
                break

            # 6. Execute Action or Breakpoint Resumption
            if parsed.has_tool_call:
                tool_name = parsed.tool_name
                tool_args = parsed.tool_args

                logger.info(f"[IterResearchAgent] Executing tool '{tool_name}' with args: {tool_args}")
                raw_obs = await self.tools.execute(tool_name, tool_args)

                # Store tool observation O_i for next round s_{t+1}
                workspace.set_observation(raw_obs)

                await self._emit_event(
                    progress_callback,
                    "tool",
                    {
                        "round": round_num,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "observation": workspace.last_observation[:500],
                    },
                )
            else:
                # 🛠️ True Breakpoint Resumption Engine:
                # When model forgets the action tag, prompt it to supply tool_call or answer,
                # then RESUME normal execution instead of abruptly stopping!
                logger.warning(
                    f"[IterResearchAgent] Round {round_num} omitted action tag. Triggering breakpoint continuation..."
                )
                resume_prompt = (
                    "⚠️ **系统断点重续与纠偏提示**：你上一轮已成功更新了研究报告，但遗漏了行动标签（未输出 <tool_call> 或 <answer>）。\n"
                    "请根据你的规划做出决定：\n"
                    "1. 若需要继续进行深度搜索或访问网页，请立即输出：<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>\n"
                    "2. 若掌握的事实已足够回答问题，请立即输出：<answer>你的最终答案</answer>\n"
                    "请直接输出上述标签，不要输出多余解释。"
                )
                resume_msgs = workspace.get_messages(system_prompt, extra_user_prompt=resume_prompt)
                try:
                    resume_resp = await self.llm_client.chat(resume_msgs, model=self.model_name, temperature=0.2)
                    if resume_resp.usage:
                        total_tokens += resume_resp.usage.total_tokens

                    resume_parsed = ProtocolParser.parse_iterresearch(resume_resp.content)

                    # Case A: Model supplied tool_call -> EXECUTE TOOL AND CONTINUE TO NEXT ROUND!
                    if resume_parsed.has_tool_call:
                        tool_name = resume_parsed.tool_name
                        tool_args = resume_parsed.tool_args
                        logger.info(
                            f"[IterResearchAgent] Breakpoint resumed successfully! Executing tool '{tool_name}' and continuing..."
                        )

                        raw_obs = await self.tools.execute(tool_name, tool_args)
                        workspace.set_observation(raw_obs)

                        await self._emit_event(
                            progress_callback,
                            "tool",
                            {
                                "round": round_num,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "observation": workspace.last_observation[:500],
                            },
                        )
                        # Intentionally DO NOT BREAK - Loop smoothly advances to round_num + 1!
                        continue

                    # Case B: Model supplied final answer
                    elif resume_parsed.has_final_answer:
                        final_prediction = resume_parsed.answer
                        termination_status = "answer_found_after_resume"
                        await self._emit_event(
                            progress_callback,
                            "final",
                            {"round": round_num, "answer": final_prediction, "report": workspace.current_report},
                        )
                        break

                    # Case C: Model signaled termination
                    elif resume_parsed.terminate:
                        final_prediction = resume_parsed.terminate_reason or workspace.current_report
                        termination_status = "terminated_by_agent"
                        break

                    # Case D: Still unformatted
                    else:
                        if is_last_round:
                            final_prediction = workspace.current_report
                            termination_status = "report_fallback"
                            break
                        else:
                            # Keep moving forward
                            workspace.last_observation = "System: Please proceed by selecting a tool or finalizing answer."
                            continue

                except Exception as resume_err:
                    logger.error(f"[IterResearchAgent] Breakpoint continuation failed: {resume_err}")
                    if is_last_round:
                        final_prediction = workspace.current_report
                        termination_status = "report_fallback"
                        break

        if not final_prediction:
            final_prediction = workspace.current_report

        total_elapsed = time.time() - start_time

        return AgentResult(
            question=question,
            prediction=final_prediction,
            report=workspace.current_report,
            trajectory=trajectory,
            termination=termination_status,
            total_tokens=total_tokens,
            total_steps=round_num,
            elapsed_time=total_elapsed,
        )
