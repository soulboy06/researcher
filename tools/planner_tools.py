# -*- coding: utf-8 -*-
"""
Memory-Augmented Planner Tool Suite.
Wraps core sensors (Search, Visit, Python Sandbox) to automatically capture,
deduplicate, and assign unique citation identifiers ([cite:id_N]) in the MemoryBank.
"""
from typing import Any, Dict, List, Optional, Set, Union
import concurrent.futures
import json
import re

from researcher.config import logger
from researcher.tools.base import BaseTool
from researcher.tools.search import SearchTool
from researcher.tools.visit import VisitTool
from researcher.tools.python_sandbox import PythonSandboxTool
from researcher.tools.memory_bank import MemoryBank


class PlannerSearchTool(BaseTool):
    """
    Search tool for Planner Agent.
    Executes batched searches and deposits each result into the MemoryBank with an id_N tag.
    """

    name = "search"
    description = (
        "Performs batched web searches: supply an array of search queries (max 5). "
        "Each result is automatically indexed into the MemoryBank with a citation ID (<id>id_N</id>)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of search query strings (1 to 5 queries).",
            }
        },
        "required": ["query"],
    }

    def __init__(self, memory_bank: MemoryBank, base_search: Optional[SearchTool] = None):
        self.memory_bank = memory_bank
        self.search_tool = base_search or SearchTool()

    def call(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        query_input: Union[str, List[str]] = ""
        if isinstance(params, str):
            try:
                parsed = json.loads(params)
                query_input = parsed.get("query", params)
            except Exception:
                query_input = params
        elif isinstance(params, dict):
            query_input = params.get("query", "")

        queries: List[str] = []
        if isinstance(query_input, list):
            queries = [str(q).strip() for q in query_input if str(q).strip()][:5]
        elif isinstance(query_input, str) and query_input.strip():
            queries = [query_input.strip()]

        if not queries:
            return "[Search Error]: No valid search queries provided."

        query_to_results: Dict[str, List[Dict[str, str]]] = {}
        seen_urls: Set[str] = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
            future_to_query = {executor.submit(self.search_tool._search_single_query, q): q for q in queries}
            for future in concurrent.futures.as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    raw_results = future.result()
                    deduped = []
                    for item in raw_results:
                        if item["url"] not in seen_urls:
                            seen_urls.add(item["url"])
                            deduped.append(item)
                    query_to_results[q] = deduped
                except Exception as exc:
                    logger.error(f"[PlannerSearch] Query '{q}' failed: {exc}")
                    query_to_results[q] = []

        # Deposit into MemoryBank and format output for Planner
        output_blocks = []
        for idx, q in enumerate(queries, 1):
            items = query_to_results.get(q, [])
            header = f"### Search Results for Query [{idx}/{len(queries)}]: '{q}'"
            if not items:
                output_blocks.append(f"{header}\n(No results found)")
                continue

            chunk_cards = [f"{header} ({len(items)} items):"]
            for item in items[:self.search_tool.max_results_per_query]:
                title = item.get("title", "Search Result")
                url = item.get("url", "")
                snippet = item.get("snippet", "")
                if not snippet:
                    continue

                card = self.memory_bank.add_evidence(
                    content=snippet,
                    summary=snippet[:400],
                    title=title,
                    url=url,
                    source_type="search",
                )
                chunk_cards.append(card)

            output_blocks.append("\n".join(chunk_cards))

        return "\n\n---\n\n".join(output_blocks)


class PlannerVisitTool(BaseTool):
    """
    Webpage visiting tool for Planner Agent.
    Fetches clean Markdown, registers evidence into MemoryBank, and returns id_N card.
    """

    name = "visit"
    description = (
        "Visits one or multiple webpages by URL. Extracted contents are indexed into the MemoryBank "
        "with unique citation IDs for the Writer to retrieve later."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of URL strings to visit (1 to 3 URLs).",
            },
            "goal": {
                "type": "string",
                "description": "The specific research goal or question to guide extraction.",
            },
        },
        "required": ["url"],
    }

    def __init__(self, memory_bank: MemoryBank, base_visit: Optional[VisitTool] = None):
        self.memory_bank = memory_bank
        self.visit_tool = base_visit or VisitTool()

    def call(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        url_input: Union[str, List[str]] = ""
        goal: Optional[str] = None

        if isinstance(params, str):
            try:
                parsed = json.loads(params)
                url_input = parsed.get("url", params)
                goal = parsed.get("goal")
            except Exception:
                url_input = params
        elif isinstance(params, dict):
            url_input = params.get("url", "")
            goal = params.get("goal")

        urls: List[str] = []
        if isinstance(url_input, list):
            urls = [str(u).strip() for u in url_input if str(u).strip()][:3]
        elif isinstance(url_input, str) and url_input.strip():
            urls = [url_input.strip()]

        if not urls:
            return "[Visit Error]: No valid URLs provided."

        cards = []
        for url in urls:
            raw_content = self.visit_tool._visit_single_url(url, goal)
            summary_preview = raw_content[:500].replace("\n", " ")

            card = self.memory_bank.add_evidence(
                content=raw_content,
                summary=summary_preview,
                title=f"Webpage: {url}",
                url=url,
                source_type="visit",
            )
            cards.append(card)

        return "\n\n".join(cards)


class PlannerPythonTool(BaseTool):
    """
    Python code execution tool for Planner Agent.
    Runs calculations and logs numerical conclusions into MemoryBank.
    """

    name = "python"
    description = (
        "Executes isolated Python code for math, data analysis, or date computation. "
        "The output is registered as evidence in the MemoryBank."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code snippet to execute.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, memory_bank: MemoryBank, base_sandbox: Optional[PythonSandboxTool] = None):
        self.memory_bank = memory_bank
        self.sandbox = base_sandbox or PythonSandboxTool()

    def call(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        code_body = ""
        if isinstance(params, dict):
            code_body = params.get("code", "")
        elif isinstance(params, str):
            try:
                parsed = json.loads(params)
                code_body = parsed.get("code", params)
            except Exception:
                code_body = params

        exec_res = self.sandbox.call({"code": code_body})
        summary_preview = exec_res[:300].strip()

        card = self.memory_bank.add_evidence(
            content=f"```python\n{code_body}\n```\n\nExecution Output:\n{exec_res}",
            summary=summary_preview,
            title="Python Sandbox Execution Result",
            url="",
            source_type="python",
        )
        return card
