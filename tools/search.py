# -*- coding: utf-8 -*-
"""
Multi-Source Web Search Engine.
Provides a resilient fallback chain: Tavily -> Google Serper -> Baidu.
Supports concurrent multi-query execution, deduplication, and per-item budgeting.
"""
import concurrent.futures
import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set, Union
import requests
from bs4 import BeautifulSoup

from researcher.config import settings, logger
from researcher.tools.base import BaseTool


class SearchTool(BaseTool):
    """
    Batched Web Search Tool with Per-Item Budgeting.
    Retrieves web results across multiple complementary queries in parallel,
    deduplicating items and enforcing per-snippet budgeting to prevent query starvation.
    """

    name = "search"
    description = (
        "Performs batched web searches: supply an array of search query strings (max 5 queries). "
        "The tool retrieves top relevant results for each query in parallel and deduplicates them."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings (1 to 5 queries). Provide complementary keywords.",
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
        max_results_per_query: int = 3,
        max_snippet_chars: int = 500,
    ):
        self.tavily_api_key = tavily_api_key or settings.tavily_api_key
        self.serper_api_key = serper_api_key or settings.serper_api_key
        self.max_results_per_query = max_results_per_query
        self.max_snippet_chars = max_snippet_chars

    def _clean_text(self, text: str) -> str:
        """Sanitize whitespace and line breaks."""
        if not text:
            return ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        return text.strip()

    def _search_tavily(self, query: str, num_results: int = 3) -> Optional[List[Dict[str, str]]]:
        """Tavily search provider."""
        if not self.tavily_api_key:
            return None

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": num_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for item in data.get("results", []):
                    results.append(
                        {
                            "title": self._clean_text(item.get("title", "")),
                            "url": item.get("url", ""),
                            "snippet": self._clean_text(item.get("content", "")),
                            "source": "tavily",
                        }
                    )
                if results:
                    return results
        except Exception as e:
            logger.warning(f"[Search:Tavily] Query '{query}' failed: {e}")

        return None

    def _search_serper(self, query: str, num_results: int = 3) -> Optional[List[Dict[str, str]]]:
        """Google Serper provider."""
        if not self.serper_api_key:
            return None

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                # Check organic search results
                for item in data.get("organic", []):
                    results.append(
                        {
                            "title": self._clean_text(item.get("title", "")),
                            "url": item.get("link", ""),
                            "snippet": self._clean_text(item.get("snippet", "")),
                            "source": "serper",
                        }
                    )
                if results:
                    return results
        except Exception as e:
            logger.warning(f"[Search:Serper] Query '{query}' failed: {e}")

        return None

    def _search_baidu(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """Baidu scraper fallback (no API key required)."""
        results = []
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.baidu.com/s?wd={encoded_query}&rn={num_results}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8", errors="ignore")

            soup = BeautifulSoup(html, "html.parser")
            containers = soup.find_all("div", class_=re.compile(r"result.*c-container"))

            for container in containers[:num_results]:
                # Extract title & url
                h3 = container.find("h3")
                if not h3:
                    continue
                a_tag = h3.find("a")
                if not a_tag:
                    continue

                title = self._clean_text(a_tag.get_text())
                href = a_tag.get("href", "")

                # Extract snippet
                snippet = ""
                snippet_div = (
                    container.find("div", class_=re.compile(r"c-abstract|content-right_8Zs40|cos-row"))
                    or container.find("span", class_=re.compile(r"content-right_8Zs40"))
                )
                if snippet_div:
                    snippet = self._clean_text(snippet_div.get_text())

                if title:
                    results.append(
                        {
                            "title": title,
                            "url": href,
                            "snippet": snippet,
                            "source": "baidu",
                        }
                    )
        except Exception as e:
            logger.warning(f"[Search:Baidu] Fallback query '{query}' failed: {e}")

        return results

    def _search_single_query(self, query: str) -> List[Dict[str, str]]:
        """Execute resilient 3-tier fallback chain for single query."""
        # Tier 1: Tavily (Cleanest structured extracts)
        results = self._search_tavily(query, self.max_results_per_query)
        if results:
            return results

        # Tier 2: Google Serper Fallback
        results = self._search_serper(query, self.max_results_per_query)
        if results:
            return results

        # Tier 3: Baidu Fallback
        return self._search_baidu(query, self.max_results_per_query)

    def _format_results(
        self, query: str, results: List[Dict[str, str]], query_idx: int = 1, total_queries: int = 1
    ) -> str:
        """
        Format raw search results with strict per-snippet character limits.
        Ensures each query is guaranteed to display completely without intermediate dropouts.
        """
        header = f"### Query [{query_idx}/{total_queries}]: '{query}'"
        if not results:
            return f"{header}\n(No direct results found for this query)"

        lines = [f"{header} (Found {len(results)} items):"]
        for idx, item in enumerate(results[:self.max_results_per_query], 1):
            snippet = item["snippet"]
            if len(snippet) > self.max_snippet_chars:
                snippet = snippet[:self.max_snippet_chars] + "..."
            snippet_part = f"\n   {snippet}" if snippet else ""
            lines.append(f"{idx}. [{item['title']}]({item['url']}){snippet_part}")

        return "\n".join(lines)

    def call(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        """
        Execute search.
        Handles both {"query": ["q1", "q2"]} and {"query": "q1"} or raw string.
        """
        query_input: Union[str, List[str]] = ""

        if isinstance(params, str):
            try:
                parsed = json.loads(params)
                query_input = parsed.get("query", params)
            except Exception:
                query_input = params
        elif isinstance(params, dict):
            query_input = params.get("query", "")

        # Normalize to list of queries (max 5)
        queries: List[str] = []
        if isinstance(query_input, list):
            queries = [str(q).strip() for q in query_input if str(q).strip()][:5]
        elif isinstance(query_input, str) and query_input.strip():
            queries = [query_input.strip()]

        if not queries:
            return "[Search Error]: No valid search queries provided."

        logger.debug(f"[Search] Executing queries ({len(queries)}): {queries}")

        # Concurrent search for multiple queries
        query_to_results: Dict[str, List[Dict[str, str]]] = {}
        seen_urls: Set[str] = set()

        if len(queries) == 1:
            raw_results = self._search_single_query(queries[0])
            query_to_results[queries[0]] = raw_results
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
                future_to_query = {executor.submit(self._search_single_query, q): q for q in queries}
                for future in concurrent.futures.as_completed(future_to_query):
                    q = future_to_query[future]
                    try:
                        raw_results = future.result()
                        # Deduplicate across queries
                        deduped = []
                        for item in raw_results:
                            if item["url"] not in seen_urls:
                                seen_urls.add(item["url"])
                                deduped.append(item)
                        query_to_results[q] = deduped
                    except Exception as exc:
                        logger.error(f"[Search] Query '{q}' generated an exception: {exc}")
                        query_to_results[q] = []

        # Assemble final output with guaranteed per-query budgeting
        output_sections = []
        total_q = len(queries)
        for idx, q in enumerate(queries, 1):
            formatted = self._format_results(q, query_to_results.get(q, []), query_idx=idx, total_queries=total_q)
            output_sections.append(formatted)

        return "\n\n" + "\n\n---\n\n".join(output_sections) + "\n\n"
