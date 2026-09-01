# -*- coding: utf-8 -*-
"""
WebWeaver Dual-Agent Research Orchestrator and Citation Post-Processor.
Coordinates:
1. WebWeaverPlanner (Breadth Search & Dynamic Outline Generation)
2. WebWeaverWriter (Deep Section Composing via Targeted MemoryBank Retrieval)
3. CitationProcessor (Converts [cite:id_N] to academic indices [1], [2] and appends bibliography)
"""
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from researcher.config import logger
from researcher.schema.message import Message
from researcher.core.llm_client import LLMClient
from researcher.tools.memory_bank import MemoryBank, EvidenceItem
from researcher.agents.base import BaseAgent, AgentResult
from researcher.agents.planner_agent import WebWeaverPlanner
from researcher.agents.writer_agent import WebWeaverWriter


class CitationProcessor:
    """
    Academic Citation Formatter and Bibliography Generator.
    Scans generated text for [cite:id_N] or [id_N] tags, maps them to chronological
    numbered footnotes [1], [2], and appends a verified References section.
    """

    @classmethod
    def format_report_with_citations(
        cls,
        draft_report: str,
        memory_bank: MemoryBank,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Process draft report text and generate referenced publication-grade output.
        
        Returns:
            (formatted_report_text, reference_list)
        """
        if not draft_report:
            return "", []

        # Find all raw citation tags: [cite:id_1], [cite: id_1], [id_1]
        pattern = r"\[cite:\s*(id_\d+)\]|\[(id_\d+)\]"
        matches = re.findall(pattern, draft_report)

        citation_id_order: List[str] = []
        seen_ids: Set[str] = set()

        for m in matches:
            cid = m[0] if m[0] else m[1]
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                citation_id_order.append(cid)

        # Mapping: 'id_3' -> '[1]', 'id_1' -> '[2]'
        id_to_index: Dict[str, int] = {}
        for idx, cid in enumerate(citation_id_order, 1):
            id_to_index[cid] = idx

        # Replace tags in text
        def _replace_tag(match):
            cid = match.group(1) or match.group(2)
            if cid in id_to_index:
                return f"[{id_to_index[cid]}]"
            return match.group(0)

        cleaned_report = re.sub(pattern, _replace_tag, draft_report)

        # Build References Section
        references: List[Dict[str, str]] = []
        ref_lines = ["\n\n---\n\n## 参考文献 (References)\n"]

        if id_to_index:
            for cid, idx in id_to_index.items():
                item = memory_bank.get_evidence_item(cid)
                if item:
                    title = item.title or f"Source {cid}"
                    url = item.url
                    summary_clip = item.summary[:150].replace("\n", " ") if item.summary else ""
                    
                    if url:
                        line = f"[{idx}] **[{title}]({url})**"
                    else:
                        line = f"[{idx}] **{title}**"
                    if summary_clip:
                        line += f" - {summary_clip}"
                    
                    ref_lines.append(line)
                    references.append({"index": str(idx), "id": cid, "title": title, "url": url})
                else:
                    ref_lines.append(f"[{idx}] Source [{cid}] (Archive record)")
                    references.append({"index": str(idx), "id": cid, "title": f"Source {cid}", "url": ""})
        else:
            # If model didn't use explicit cite tags, check if MemoryBank has evidence to list as general bibliography
            all_items = memory_bank.get_all_items()
            if all_items:
                for idx, item in enumerate(all_items[:10], 1):
                    title = item.title or f"Source {idx}"
                    url = item.url
                    if url:
                        ref_lines.append(f"[{idx}] **[{title}]({url})**")
                    else:
                        ref_lines.append(f"[{idx}] **{title}**")
                    references.append({"index": str(idx), "id": item.citation_id, "title": title, "url": url})

        final_report = cleaned_report.strip() + "\n" + "\n".join(ref_lines)
        return final_report, references


class WebWeaverAgent(BaseAgent):
    """
    Main WebWeaver Orchestrator.
    Sequentially runs Planner (Outline & Evidence Discovery) -> Writer (Targeted Deep Composing) -> CitationProcessor.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        planner_rounds: int = 5,
        writer_rounds: int = 5,
        timeout: float = 600.0,
        instruction: str = "",
        model_name: Optional[str] = None,
    ):
        super().__init__(max_rounds=planner_rounds + writer_rounds, timeout=timeout)
        self.memory_bank = MemoryBank()
        self.instruction = instruction
        self.model_name = model_name

        self.planner = WebWeaverPlanner(
            memory_bank=self.memory_bank,
            llm_client=llm_client,
            max_rounds=planner_rounds,
            timeout=timeout / 2.0,
            instruction=instruction,
            model_name=model_name,
        )

        self.writer = WebWeaverWriter(
            memory_bank=self.memory_bank,
            llm_client=llm_client,
            max_rounds=writer_rounds,
            timeout=timeout / 2.0,
            instruction=instruction,
            model_name=model_name,
        )

    async def close(self) -> None:
        """Close sub-agent connections gracefully."""
        if hasattr(self, "planner") and self.planner:
            await self.planner.close()
        if hasattr(self, "writer") and self.writer:
            await self.writer.close()

    async def run(
        self,
        question: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> AgentResult:
        start_time = time.time()
        logger.info(f"[WebWeaverAgent] Starting dual-agent research on: '{question}'")

        # 1. Phase 1: Planner Agent builds dynamic outline and fills MemoryBank
        await self._emit_event(
            progress_callback,
            "phase",
            {"phase": "planner_start", "message": "大纲规划智能体 (Planner) 启动，开始广度搜索并构建大纲..."},
        )

        final_outline = await self.planner.run_planner(question, progress_callback=progress_callback)
        logger.info(f"[WebWeaverAgent] Planner completed outline ({len(final_outline)} chars, {self.memory_bank.size()} evidence items)")

        # Broadcast complete MemoryBank and Outline snapshot to UI
        bank_items = [
            {
                "id": item.id,
                "title": item.title,
                "url": item.url,
                "summary": item.summary[:180] if item.summary else "",
            }
            for item in self.memory_bank.get_all()
        ]
        await self._emit_event(
            progress_callback,
            "memory_bank_update",
            {"total_items": len(bank_items), "items": bank_items},
        )
        await self._emit_event(
            progress_callback,
            "outline",
            {"round": self.planner.max_rounds, "outline": final_outline},
        )

        # 2. Phase 2: Writer Agent writes report using RetrieveTool
        await self._emit_event(
            progress_callback,
            "phase",
            {
                "phase": "writer_start",
                "message": "深度撰写智能体 (Writer) 启动，根据大纲按章节拉取证据并撰写学术长文...",
                "outline": final_outline,
            },
        )

        draft_report = await self.writer.run_writer(question, final_outline, progress_callback=progress_callback)
        logger.info(f"[WebWeaverAgent] Writer completed draft report ({len(draft_report)} chars)")

        # 3. Phase 3: Citation Post-Processing
        final_report, references = CitationProcessor.format_report_with_citations(
            draft_report=draft_report,
            memory_bank=self.memory_bank,
        )

        total_elapsed = time.time() - start_time
        logger.info(f"[WebWeaverAgent] WebWeaver completed research in {total_elapsed:.1f}s with {len(references)} citations.")

        await self._emit_event(
            progress_callback,
            "final",
            {
                "question": question,
                "outline": final_outline,
                "report": final_report,
                "citations_count": len(references),
                "elapsed_time": total_elapsed,
            },
        )

        return AgentResult(
            question=question,
            prediction=final_report,
            report=final_report,
            trajectory=[],
            termination="webweaver_completed",
            total_steps=self.planner.max_rounds + self.writer.max_rounds,
            elapsed_time=total_elapsed,
        )
