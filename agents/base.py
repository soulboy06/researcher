# -*- coding: utf-8 -*-
"""
Base Agent Interface, Event Dispatcher, and Result Data Model.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import datetime
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Union

from researcher.config import logger
from researcher.tools.base import BaseTool, ToolRegistry
from researcher.tools.search import SearchTool
from researcher.tools.visit import VisitTool
from researcher.tools.python_sandbox import PythonSandboxTool
from researcher.tools.extractor import EvidenceExtractorTool


@dataclass
class AgentResult:
    """
    Standard outcome object returned upon research completion.
    """
    question: str
    prediction: str                      # Final definitive answer
    report: str                          # Complete accumulated research report
    trajectory: List[Dict[str, Any]]     # Full chronological execution steps
    termination: str                     # Termination status (e.g. 'answer_found', 'max_rounds', 'timeout')
    total_tokens: int = 0                # Total prompt + completion tokens used
    total_steps: int = 0                 # Number of interaction steps/rounds taken
    elapsed_time: float = 0.0            # Execution duration in seconds


class BaseAgent(ABC):
    """
    Abstract base class for all research agent paradigms.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        max_rounds: int = 10,
        timeout: float = 600.0,
        use_xml_protocol: bool = True,
    ):
        self.max_rounds = max_rounds
        self.timeout = timeout
        self.use_xml_protocol = use_xml_protocol

        # Initialize or populate tool registry
        if tool_registry is not None:
            self.tools = tool_registry
        else:
            self.tools = ToolRegistry()
            self.tools.register(SearchTool())
            self.tools.register(VisitTool())
            self.tools.register(PythonSandboxTool())
            self.tools.register(EvidenceExtractorTool())

    async def _emit_event(
        self,
        callback: Optional[Callable[[Dict[str, Any]], Any]],
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Helper to dispatch progress events to async/sync callbacks."""
        if not callback:
            return

        payload = {
            "type": event_type,
            "timestamp": datetime.datetime.now().isoformat(),
            **data,
        }

        try:
            res = callback(payload)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            logger.warning(f"Error in progress_callback: {e}")

    async def close(self) -> None:
        """Close resources and underlying clients."""
        if hasattr(self, "llm_client") and self.llm_client:
            await self.llm_client.close()

    @abstractmethod
    async def run(
        self,
        question: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> AgentResult:
        """
        Execute research lifecycle asynchronously.
        """
        raise NotImplementedError
