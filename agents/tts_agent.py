# -*- coding: utf-8 -*-
"""
Test-Time Scaling (TTS) Agent with Parallel Exploration & Chief Synthesis.
Implements Inference Scaling / Test-Time Search:
1. Dispatches N parallel research branches with diverse sampling temperatures & perspective biases.
2. Cross-verifies multi-branch findings, eliminates single-agent hallucinations.
3. Synthesizes a high-confidence, comprehensive master research report.
"""
import asyncio
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional

from researcher.config import logger
from researcher.schema.message import Message, MessageRole
from researcher.core.llm_client import LLMClient
from researcher.core.prompt import get_synthesizer_prompt
from researcher.agents.base import BaseAgent, AgentResult
from researcher.agents.iter_agent import IterResearchAgent


@dataclass
class TTSBranchResult:
    """Outcome and trajectory metadata from a single parallel research branch."""

    branch_id: int
    temperature: float
    bias_description: str
    prediction: str
    report: str
    total_tokens: int
    total_steps: int
    elapsed_time: float
    termination: str


DEFAULT_BRANCH_BIASES = [
    "【侧重点：官方权威通告与核心事实】请重点聚焦于权威发布、官方新闻稿、核心确切数据与基础事实定义。",
    "【侧重点：深度技术机理与数学/物理推导】请重点聚焦于底层算法逻辑、物理/数学公式机理及技术架构实现细节。",
    "【侧重点：学术历史沿革与行业深远影响】请重点聚焦于学术研究的历史演进脉络、行业现实应用及未来的深远启示。",
]


class TestTimeScalingAgent(BaseAgent):
    """
    Test-Time Scaling (TTS) Multi-Branch Agent.
    Scales test-time compute by running N diverse IterResearchAgent instances in parallel,
    followed by a Master Synthesizer LLM phase.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        num_parallel_agents: int = 3,
        temperatures: Optional[List[float]] = None,
        branch_biases: Optional[List[str]] = None,
        branch_max_rounds: int = 5,
        timeout: float = 600.0,
        instruction: str = "",
        model_name: Optional[str] = None,
    ):
        super().__init__(max_rounds=branch_max_rounds, timeout=timeout)
        self.llm_client = llm_client or LLMClient(model_name=model_name)
        self.num_parallel_agents = num_parallel_agents
        self.temperatures = temperatures or [0.3, 0.6, 0.9]
        self.branch_biases = branch_biases or DEFAULT_BRANCH_BIASES
        self.branch_max_rounds = branch_max_rounds
        self.instruction = instruction
        self.model_name = model_name or self.llm_client.model_name
        self._active_branch_agents: List[IterResearchAgent] = []

    def estimate_cost(self, num_parallel_agents: Optional[int] = None) -> str:
        """
        Calculate expected compute and token cost multiplier.
        """
        n = num_parallel_agents or self.num_parallel_agents
        total_multiplier = n + 0.5
        return (
            f"📊 【TTS 测试时计算扩展成本评估】:\n"
            f"   • 并行探索阶段: {n} 个独立智能体 × 基准算力\n"
            f"   • 首席裁决阶段: ~0.5× 基准算力\n"
            f"   • 总算力消耗: 约 {total_multiplier:.1f}× 单智能体开销\n"
            f"   • 适用场景: 高价值决策、前沿学术调研、严谨事实交叉核对。"
        )

    async def close(self) -> None:
        """Gracefully close all active branch agents and main client."""
        for agent in self._active_branch_agents:
            try:
                await agent.close()
            except Exception:
                pass
        self._active_branch_agents.clear()
        if hasattr(self, "llm_client") and self.llm_client:
            await self.llm_client.close()

    async def run_parallel_research(
        self,
        question: str,
        num_parallel_agents: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> List[TTSBranchResult]:
        """
        Phase 1: Dispatch N diverse IterResearchAgent instances in parallel.
        """
        n = num_parallel_agents or self.num_parallel_agents
        logger.info(f"[TestTimeScalingAgent] Launching Phase 1: {n} parallel research branches...")

        tasks = []
        self._active_branch_agents.clear()

        branches_meta = [
            {
                "branch_id": i + 1,
                "temperature": self.temperatures[i % len(self.temperatures)],
                "bias": self.branch_biases[i % len(self.branch_biases)],
            }
            for i in range(n)
        ]
        await self._emit_event(progress_callback, "tts_branches_init", {"branches": branches_meta})

        for i in range(n):
            temp = self.temperatures[i % len(self.temperatures)]
            bias = self.branch_biases[i % len(self.branch_biases)]
            branch_instruction = f"{self.instruction}\n{bias}".strip()

            branch_agent = IterResearchAgent(
                max_rounds=self.branch_max_rounds,
                timeout=self.timeout * 0.8,
                instruction=branch_instruction,
                model_name=self.model_name,
            )
            self._active_branch_agents.append(branch_agent)

            async def _run_branch(idx: int, agent: IterResearchAgent, t: float, b: str):
                logger.info(f"[TTS Branch {idx+1}] Starting research with Temperature={t}...")
                
                async def _branch_cb(event: Dict[str, Any]):
                    # Construct structured branch payload that won't get its type overwritten
                    branch_payload = {
                        "branch_id": idx + 1,
                        "inner_type": event.get("type", "round"),
                        "round": event.get("round", 1),
                        "plan": event.get("plan", ""),
                        "tool_name": event.get("tool_name", ""),
                        "tool_args": event.get("tool_args", {}),
                        "observation": event.get("observation", ""),
                        "report": event.get("report", ""),
                        "answer": event.get("answer", ""),
                    }
                    await self._emit_event(progress_callback, "branch_event", branch_payload)

                try:
                    res = await agent.run(question, progress_callback=_branch_cb)
                    return TTSBranchResult(
                        branch_id=idx + 1,
                        temperature=t,
                        bias_description=b,
                        prediction=res.prediction,
                        report=res.report,
                        total_tokens=res.total_tokens,
                        total_steps=res.total_steps,
                        elapsed_time=res.elapsed_time,
                        termination=res.termination,
                    )
                except Exception as e:
                    logger.error(f"[TTS Branch {idx+1}] Failed with exception: {e}")
                    return TTSBranchResult(
                        branch_id=idx + 1,
                        temperature=t,
                        bias_description=b,
                        prediction=f"Branch failed: {e}",
                        report="Branch encountered error.",
                        total_tokens=0,
                        total_steps=0,
                        elapsed_time=0.0,
                        termination="error",
                    )

            tasks.append(_run_branch(i, branch_agent, temp, bias))

        branch_results = await asyncio.gather(*tasks)
        logger.info(f"[TestTimeScalingAgent] All {len(branch_results)} branches completed.")
        return list(branch_results)

    async def synthesize_findings(
        self,
        question: str,
        branch_results: List[TTSBranchResult],
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> str:
        """
        Phase 2: Chief Synthesizer cross-verifies and merges findings.
        """
        logger.info(f"[TestTimeScalingAgent] Launching Phase 2: Master Synthesis & Adjudication...")

        await self._emit_event(
            progress_callback,
            "phase",
            {
                "phase": "synthesizer_start",
                "message": f"首席裁决合成器启动，正在交叉核对 {len(branch_results)} 路独立研报并消除矛盾...",
            },
        )

        system_prompt = get_synthesizer_prompt(question=question, instruction=self.instruction)

        # Assemble multi-branch report dossier
        dossier_sections = [f"**【用户研究课题 (Question)】**\n{question}\n"]
        for res in branch_results:
            header = f"### === 【第 {res.branch_id} 路独立研究报告 (采样温度 T={res.temperature})】 ==="
            bias_info = f"探索偏置: {res.bias_description}"
            content = res.report or res.prediction
            dossier_sections.append(f"{header}\n{bias_info}\n\n{content}\n")

        user_content = (
            "请作为首席裁决专家，对以下多路独立研报进行深度交叉比对：\n"
            "1. 提取所有报告公认的一致事实；\n"
            "2. 裁决各报告之间的冲突与歧义，剔除幻觉；\n"
            "3. 融合成一篇逻辑严密、论证详实的终极研究报告。\n\n"
            + "\n----------------------------------------\n".join(dossier_sections)
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_content),
        ]

        try:
            resp = await self.llm_client.chat(
                messages=messages,
                model=self.model_name,
                temperature=0.2,
            )
            final_report = resp.content.strip()
        except Exception as e:
            logger.error(f"[TestTimeScalingAgent] Synthesis failed: {e}. Falling back to longest branch report.")
            valid_branches = [b for b in branch_results if b.termination != "error"]
            if valid_branches:
                final_report = max(valid_branches, key=lambda b: len(b.report)).report
            else:
                final_report = "Synthesis failed due to LLM error."

        return final_report

    async def run(
        self,
        question: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> AgentResult:
        start_time = time.time()
        logger.info(f"[TestTimeScalingAgent] Running full TTS pipeline for: '{question}'")

        # 1. Phase 1: Parallel Exploration
        await self._emit_event(
            progress_callback,
            "phase",
            {
                "phase": "parallel_start",
                "message": f"并发启动 {self.num_parallel_agents} 路多视角探索智能体...",
                "num_agents": self.num_parallel_agents,
                "cost_estimation": self.estimate_cost(),
            },
        )

        branch_results = await self.run_parallel_research(
            question=question,
            num_parallel_agents=self.num_parallel_agents,
            progress_callback=progress_callback,
        )

        # 2. Phase 2: Synthesis
        final_synthesized_report = await self.synthesize_findings(
            question=question,
            branch_results=branch_results,
            progress_callback=progress_callback,
        )

        total_elapsed = time.time() - start_time
        total_tokens = sum(b.total_tokens for b in branch_results)
        total_steps = sum(b.total_steps for b in branch_results)

        await self._emit_event(
            progress_callback,
            "final",
            {
                "question": question,
                "report": final_synthesized_report,
                "branches_count": len(branch_results),
                "total_tokens": total_tokens,
                "elapsed_time": total_elapsed,
            },
        )

        # Clean up
        await self.close()

        return AgentResult(
            question=question,
            prediction=final_synthesized_report,
            report=final_synthesized_report,
            trajectory=[
                {
                    "branch_id": b.branch_id,
                    "temperature": b.temperature,
                    "tokens": b.total_tokens,
                    "report_len": len(b.report),
                }
                for b in branch_results
            ],
            termination="tts_synthesized",
            total_tokens=total_tokens,
            total_steps=total_steps,
            elapsed_time=total_elapsed,
        )
