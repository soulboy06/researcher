# -*- coding: utf-8 -*-
"""
Webpage Scraping, DOM Sanitization, and Automatic Distillation Tool (Visit).
Features:
- Jina Reader API integration for clean direct Markdown
- Local BeautifulSoup DOM cleaning + markdownify conversion fallback
- Automatic secondary LLM evidence distillation for lengthy documents (>8,000 chars)
- Concurrent multi-URL fetching support
"""
import concurrent.futures
import json
import re
from typing import Any, Dict, List, Optional, Union
import requests
from bs4 import BeautifulSoup

try:
    from markdownify import MarkdownConverter
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False
    MarkdownConverter = None

from researcher.config import settings, logger
from researcher.tools.base import BaseTool


def parse_html_to_clean_markdown(html_content: str, url: str = "") -> str:
    """
    Parse raw HTML into clean, high-density Markdown text.
    Decomposes noise DOM nodes (scripts, styles, navigations, footers, sidebars).
    """
    if not html_content or not html_content.strip():
        return "[Empty Webpage Content]"

    soup = BeautifulSoup(html_content, "html.parser")
    page_title = soup.title.get_text().strip() if soup.title else "No Title"

    # Remove all scripts, stylesheets, and irrelevant components
    for noise_tag in soup(["script", "style", "noscript", "svg", "iframe", "button"]):
        noise_tag.decompose()

    for layout_noise in soup.find_all(["nav", "footer", "aside", "form", "figure", "header"]):
        layout_noise.decompose()

    # Specialized container extraction
    main_body = None
    if "wikipedia.org" in url:
        main_body = soup.find("div", {"id": "mw-content-text"})
    elif soup.find("article"):
        main_body = soup.find("article")
    elif soup.find("main"):
        main_body = soup.find("main")

    target_element = main_body or soup

    # Convert DOM element to Markdown
    if HAS_MARKDOWNIFY and MarkdownConverter is not None:
        try:
            converter = MarkdownConverter(heading_style="ATX", strip=["a", "img"])
            markdown_text = converter.convert_soup(target_element)
        except Exception as e:
            logger.warning(f"[Visit:Markdownify] Conversion failed: {e}. Falling back to plain text.")
            markdown_text = target_element.get_text(separator="\n\n")
    else:
        # High quality fallback using DOM text extraction
        text_lines = []
        for child in target_element.children:
            if hasattr(child, "name") and child.name in ["h1", "h2", "h3", "h4"]:
                text_lines.append(f"## {child.get_text().strip()}\n")
            elif hasattr(child, "get_text"):
                t = child.get_text().strip()
                if t:
                    text_lines.append(t)
        markdown_text = "\n\n".join(text_lines) if text_lines else target_element.get_text(separator="\n\n")

    # Clean redundant whitespace and multi-newlines
    markdown_text = re.sub(r"\r\n", "\n", markdown_text)
    markdown_text = re.sub(r"[ \t]+", " ", markdown_text)
    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text).strip()

    # Ensure title header
    if not markdown_text.startswith("#"):
        markdown_text = f"# {page_title}\n\n{markdown_text}"

    return markdown_text


class VisitTool(BaseTool):
    """
    Webpage Visiting and Content Extraction Tool.
    """

    name = "visit"
    description = (
        "Visits one or multiple webpages by URL and extracts clean, readable Markdown content. "
        "Supply a single URL string or an array of URLs, along with an optional research goal."
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

    def __init__(
        self,
        jina_api_key: Optional[str] = None,
        timeout: float = 20.0,
        auto_distill: bool = True,
        distill_threshold_chars: int = 8000,
    ):
        self.jina_api_key = jina_api_key or settings.jina_api_key
        self.timeout = timeout
        self.auto_distill = auto_distill
        self.distill_threshold_chars = distill_threshold_chars

    def _fetch_via_jina(self, url: str) -> Optional[str]:
        """Fetch clean markdown via Jina Reader API."""
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "text/markdown",
            "User-Agent": "WebResearcher/1.0",
        }
        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"

        try:
            resp = requests.get(jina_url, headers=headers, timeout=self.timeout)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text.strip()
        except Exception as e:
            logger.warning(f"[Visit:Jina] URL {url} fetch failed: {e}. Trying local fallback.")

        return None

    def _fetch_locally(self, url: str) -> str:
        """Fetch and clean webpage using local requests + BeautifulSoup."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        try:
            resp = requests.get(url, headers=headers, timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()

            # Decode content with encoding fallback
            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type and "xml" not in content_type:
                return f"[Visit Error]: Unsupported content type '{content_type}' at {url}"

            encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else resp.apparent_encoding
            html_text = resp.content.decode(encoding or "utf-8", errors="replace")

            markdown_text = parse_html_to_clean_markdown(html_text, url=url)
            return markdown_text

        except requests.exceptions.Timeout:
            return f"[Visit Error]: Request timed out while accessing {url}"
        except requests.exceptions.HTTPError as e:
            return f"[Visit Error]: HTTP status error {e.response.status_code} while accessing {url}"
        except Exception as e:
            return f"[Visit Error]: Failed to fetch webpage {url}. Error: {str(e)}"

    def _visit_single_url(self, url: str, goal: Optional[str] = None) -> str:
        """Fetch single URL and auto-distill if length exceeds threshold."""
        url = url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        # Tier 1: Jina Reader
        content = self._fetch_via_jina(url)
        if not content:
            # Tier 2: Local HTML Scraper + Markdownify
            content = self._fetch_locally(url)

        # Automatic Secondary Distillation for lengthy contents
        if self.auto_distill and goal and len(content) > self.distill_threshold_chars:
            try:
                from researcher.tools.extractor import EvidenceExtractorTool
                extractor = EvidenceExtractorTool()
                logger.info(
                    f"[Visit] Page {url} is lengthy ({len(content)} chars). Auto-distilling evidence for Goal: '{goal}'..."
                )
                distilled = extractor.call({"content": content, "goal": goal, "max_chunks": 5})
                return f"### Webpage Content for: {url} (Auto-Distilled for Goal: {goal})\n\n{distilled}"
            except Exception as dist_err:
                logger.warning(f"[Visit] Auto-distillation failed: {dist_err}. Returning raw markdown.")

        goal_header = f" (Goal: {goal})" if goal else ""
        return f"### Webpage Content for: {url}{goal_header}\n\n{content}"

    def call(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        """
        Execute webpage visit.
        """
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

        # Normalize URLs
        urls: List[str] = []
        if isinstance(url_input, list):
            urls = [str(u).strip() for u in url_input if str(u).strip()][:3]
        elif isinstance(url_input, str) and url_input.strip():
            urls = [url_input.strip()]

        if not urls:
            return "[Visit Error]: No valid URLs provided."

        logger.debug(f"[Visit] Visiting {len(urls)} URLs: {urls}")

        if len(urls) == 1:
            return self._visit_single_url(urls[0], goal)

        # Multi-URL concurrent visit
        results: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as executor:
            future_to_url = {executor.submit(self._visit_single_url, u, goal): u for u in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                u = future_to_url[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(f"[Visit Error]: Failed to fetch {u}: {exc}")

        return "\n\n==============================\n\n".join(results)
