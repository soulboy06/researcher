# -*- coding: utf-8 -*-
"""
Shared Evidence Memory Bank and Writer Retrieve Tool.
Stores evidence collected by the Planner with unique citation identifiers ([cite:id_N]),
and provides section-targeted retrieval for the Writer.
"""
from dataclasses import dataclass, field
import datetime
import threading
from typing import Any, Dict, List, Optional, Union

from researcher.config import logger
from researcher.tools.base import BaseTool


@dataclass
class EvidenceItem:
    """Represents a single verified piece of factual evidence in the bank."""

    citation_id: str             # e.g., 'id_1', 'id_2'
    title: str = ""
    url: str = ""
    summary: str = ""            # Query-relevant concise summary
    content: str = ""            # Full detailed evidence text
    source_type: str = "web"     # 'search', 'visit', 'python', 'file'
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    @property
    def id(self) -> str:
        return self.citation_id


class MemoryBank:
    """
    Thread-safe Memory Bank for storing and retrieving factual evidence.
    Planner writes evidence via add_evidence, and Writer queries it via retrieve.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.evidence: Dict[str, EvidenceItem] = {}
        self.id_counter: int = 0

    def add_evidence(
        self,
        content: str,
        summary: str,
        title: str = "",
        url: str = "",
        source_type: str = "web",
    ) -> str:
        """
        Add a new piece of evidence to the memory bank and return an XML observation card.
        """
        with self._lock:
            self.id_counter += 1
            citation_id = f"id_{self.id_counter}"

            item = EvidenceItem(
                citation_id=citation_id,
                title=title or "Untitled Evidence",
                url=url,
                summary=summary.strip(),
                content=content.strip(),
                source_type=source_type,
            )
            self.evidence[citation_id] = item

            logger.debug(f"[MemoryBank] Stored evidence {citation_id} from {url or title} (Total: {len(self.evidence)})")

            # Return standard Planner observation format
            url_part = f"\n<url>{url}</url>" if url else ""
            title_part = f"\n<title>{title}</title>" if title else ""
            return (
                f"<evidence_chunk>\n"
                f"<id>{citation_id}</id>"
                f"{title_part}"
                f"{url_part}\n"
                f"<summary>{summary.strip()}</summary>\n"
                f"</evidence_chunk>"
            )

    def retrieve(self, citation_ids: List[str]) -> str:
        """
        Retrieve raw content for a given list of citation IDs.
        Dedicated for the Writer Agent.
        """
        if not citation_ids:
            return "No citation IDs provided for retrieval."

        with self._lock:
            retrieved_sections = []
            for cid in citation_ids:
                cid_clean = cid.strip()
                if cid_clean in self.evidence:
                    item = self.evidence[cid_clean]
                    header = f"--- Evidence [{cid_clean}]"
                    if item.title:
                        header += f": {item.title}"
                    if item.url:
                        header += f" ({item.url})"
                    header += " ---"

                    retrieved_sections.append(f"{header}\n{item.content}\n")
                else:
                    logger.warning(f"[MemoryBank] Citation ID '{cid_clean}' not found in bank.")
                    retrieved_sections.append(f"<error>Citation ID '{cid_clean}' not found in Memory Bank.</error>")

            return "\n\n".join(retrieved_sections)

    def get_evidence_item(self, citation_id: str) -> Optional[EvidenceItem]:
        """Retrieve individual EvidenceItem by citation ID."""
        with self._lock:
            return self.evidence.get(citation_id.strip())

    def get_all_items(self) -> List[EvidenceItem]:
        """Return all evidence items currently in the bank."""
        with self._lock:
            return list(self.evidence.values())

    def get_all(self) -> List[EvidenceItem]:
        """Alias for get_all_items."""
        return self.get_all_items()

    def get_all_ids(self) -> List[str]:
        """Return list of all registered citation IDs."""
        with self._lock:
            return list(self.evidence.keys())

    def size(self) -> int:
        """Return total count of evidence items."""
        with self._lock:
            return len(self.evidence)

    def clear(self) -> None:
        """Reset memory bank."""
        with self._lock:
            self.evidence.clear()
            self.id_counter = 0


class RetrieveTool(BaseTool):
    """
    Writer Agent's exclusive tool for retrieving detailed evidence chunks from the Memory Bank.
    """

    name = "retrieve"
    description = (
        "Retrieves specific evidence chunks from the Memory Bank using their citation IDs (e.g. ['id_1', 'id_2']). "
        "Use this tool to acquire detailed facts and numbers needed to write a specific report section."
    )
    parameters = {
        "type": "object",
        "properties": {
            "citation_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of citation IDs to retrieve from the Memory Bank (e.g., ['id_1', 'id_3']).",
            }
        },
        "required": ["citation_ids"],
    }

    def __init__(self, memory_bank: MemoryBank):
        self.memory_bank = memory_bank

    def call(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        citation_ids: List[str] = []
        if isinstance(params, dict):
            raw_ids = params.get("citation_ids", [])
            if isinstance(raw_ids, list):
                citation_ids = [str(c).strip() for c in raw_ids if str(c).strip()]
            elif isinstance(raw_ids, str):
                citation_ids = [raw_ids.strip()]
        elif isinstance(params, list):
            citation_ids = [str(c).strip() for c in params if str(c).strip()]
        elif isinstance(params, str):
            citation_ids = [params.strip()]

        logger.debug(f"[RetrieveTool] Fetching IDs: {citation_ids}")
        return self.memory_bank.retrieve(citation_ids)
