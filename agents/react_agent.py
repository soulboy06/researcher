# -*- coding: utf-8 -*-
"""
Standard Multi-Turn ReAct Agent (Baseline Paradigm).
Maintains full conversational message history across all turns.
"""
import time
from typing import Any, Callable, Dict, List, Optional

from researcher.config import settings, logger
from researcher.schema.message import (
    Message,
    MessageRole,
    ToolCall,
    ToolCallFunction,
)
from researcher.core.llm_client import LLMClient
from researcher.core.context_guard import context_guard, token_counter
from researcher.core.prompt import get_react_prompt
from researcher.core.protocol_parser import ProtocolParser, ParsedOutput
from researcher.tools.base import ToolRegistry
from researcher.agents.base import BaseAgent, AgentResult


class ReactAgent(BaseAgent):
    """
    Standard multi-turn ReAct Agent.
    Serves as the evaluation baseline for tokens, latency, and reasoning fidelity.
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

        # Build initial system prompt and messages
        system_prompt = get_react_prompt(
            tools=tool_definitions,
            question=question,
            instruction=self.instruction,
        )

        messages: List[Message] = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=question),
        ]

        trajectory: List[Dict[str, Any]] = [
            {"step": 0, "role": "system", "content": system_prompt},
            {"step": 0, "role": "user", "content": question},
        ]

        total_tokens = 0
        round_num = 0
        final_prediction = ""
        termination_status = "max_rounds_reached"
        last_thought = ""

        while round_num < self.max_rounds:
            round_num += 1

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                logger.warning(f"[ReactAgent] Timeout reached ({elapsed:.1f}s > {self.timeout}s)")
                termination_status = "timeout"
                final_prediction = last_thought or "Research stopped due to timeout."
                break

            logger.info(f"[ReactAgent] Round {round_num}/{self.max_rounds} starting...")

            # 1. Budget check and warning
            context_guard.check_context_budget(messages, model=self.model_name)

            # 2. Check if final force round
            is_last_round = (round_num == self.max_rounds)
            if is_last_round:
                force_msg = Message(
                    role=MessageRole.USER,
                    content=(
                        "You have reached the maximum round limit. Do not call any tools anymore. "
                        "Synthesize all evidence gathered so far and provide the final answer enclosed in <answer>...</answer> tags."
                    )
                )
                messages.append(force_msg)

            # 3. Call LLM
            try:
                response = await self.llm_client.chat(
                    messages=messages,
                    model=self.model_name,
                    temperature=0.3,
                    tools=tool_definitions if not self.use_xml_protocol else None,
                )
            except Exception as e:
                logger.error(f"[ReactAgent] LLM call failed in round {round_num}: {e}")
                termination_status = f"llm_error: {e}"
                final_prediction = f"Error occurred: {e}"
                break

            if response.usage:
                total_tokens += response.usage.total_tokens

            # Emit reasoning if present
            if response.reasoning_content:
                await self._emit_event(
                    progress_callback,
                    "thinking",
                    {"round": round_num, "content": response.reasoning_content},
                )

            # 4. Handle XML Protocol Mode (Default)
            if self.use_xml_protocol:
                parsed: ParsedOutput = ProtocolParser.parse_react(response.content)
                last_thought = parsed.plan

                # Emit step progress
                await self._emit_event(
                    progress_callback,
                    "step",
                    {
                        "round": round_num,
                        "thought": parsed.plan,
                        "tool_call": parsed.tool_call_raw,
                        "answer": parsed.answer,
                    },
                )

                trajectory.append({
                    "round": round_num,
                    "role": "assistant",
                    "content": response.content,
                    "parsed": {
                        "thought": parsed.plan,
                        "tool_name": parsed.tool_name,
                        "tool_args": parsed.tool_args,
                        "answer": parsed.answer,
                    }
                })

                # Check termination
                if parsed.has_final_answer:
                    final_prediction = parsed.answer
                    termination_status = "answer_found"
                    await self._emit_event(
                        progress_callback,
                        "final",
                        {"round": round_num, "answer": final_prediction},
                    )
                    break

                if parsed.terminate:
                    final_prediction = parsed.terminate_reason or parsed.plan
                    termination_status = "terminated_by_agent"
                    break

                # Execute Tool if requested
                if parsed.has_tool_call:
                    tool_name = parsed.tool_name
                    tool_args = parsed.tool_args

                    logger.info(f"[ReactAgent] Executing tool '{tool_name}' with args: {tool_args}")
                    raw_obs = await self.tools.execute(tool_name, tool_args)
                    guarded_obs = context_guard.truncate_observation(raw_obs)

                    await self._emit_event(
                        progress_callback,
                        "tool",
                        {
                            "round": round_num,
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "observation": guarded_obs[:500],
                        },
                    )

                    # Append Assistant Message & Tool Observation to message chain
                    messages.append(Message(role=MessageRole.ASSISTANT, content=response.content))
                    messages.append(Message(
                        role=MessageRole.USER,
                        content=f"<tool_response>\n{guarded_obs}\n</tool_response>"
                    ))

                    trajectory.append({
                        "round": round_num,
                        "role": "tool_observation",
                        "tool_name": tool_name,
                        "observation": guarded_obs,
                    })
                else:
                    # No tool call and no answer: model output plain text
                    if is_last_round:
                        final_prediction = response.content
                        termination_status = "finalized_plain_text"
                        break
                    else:
                        messages.append(Message(role=MessageRole.ASSISTANT, content=response.content))
                        messages.append(Message(
                            role=MessageRole.USER,
                            content="请明确下一步行动：若需继续调用工具检索更多信息，请输出 <tool_call>...；若已收集完备，请将最终答案包裹在 <answer>...</answer> 中输出。"
                        ))

            # 5. Handle Native Function Calling Mode
            else:
                # Handle native tool calls
                if response.tool_calls:
                    messages.append(Message(role=MessageRole.ASSISTANT, content=response.content, tool_calls=response.tool_calls))
                    for tc in response.tool_calls:
                        name, args = ProtocolParser.parse_tool_payload(tc.function.arguments)
                        raw_obs = await self.tools.execute(tc.function.name, args)
                        guarded_obs = context_guard.truncate_observation(raw_obs)
                        messages.append(Message(role=MessageRole.TOOL, content=guarded_obs, name=tc.function.name))
                else:
                    final_prediction = response.content
                    termination_status = "answer_found"
                    break

        if not final_prediction:
            final_prediction = last_thought or "Unable to reach a definitive answer within maximum rounds."

        total_elapsed = time.time() - start_time

        return AgentResult(
            question=question,
            prediction=final_prediction,
            report=last_thought,
            trajectory=trajectory,
            termination=termination_status,
            total_tokens=total_tokens,
            total_steps=round_num,
            elapsed_time=total_elapsed,
        )
