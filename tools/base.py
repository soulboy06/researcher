# -*- coding: utf-8 -*-
"""
Base Tool Interface and Tool Registry.
Defines strict schemas, uniform execution contracts, and function calling definitions.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import inspect
import asyncio

from researcher.config import logger


class BaseTool(ABC):
    """
    Abstract Base Class for all tools.
    Every tool must declare:
    - name: Unique tool identifier
    - description: Semantic functionality description for LLM planning
    - parameters: JSON Schema defining expected inputs
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    @abstractmethod
    def call(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        """
        Synchronous tool execution contract.
        Must return formatted string representation of the observation.
        """
        raise NotImplementedError(f"Tool '{self.name}' must implement call()")

    async def acall(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        """
        Asynchronous tool execution interface.
        Defaults to running the sync call() in a separate executor thread.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.call(params, **kwargs))

    def get_openai_tool_definition(self) -> Dict[str, Any]:
        """
        Export tool specification in standard OpenAI Tool Definition format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }


class ToolRegistry:
    """
    Central Tool Registry to manage, inspect, and invoke registered tools.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if not tool.name:
            raise ValueError(f"Tool {tool} has an empty name.")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: '{tool.name}'")

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool by name."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.debug(f"Unregistered tool: '{tool_name}'")

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Retrieve tool instance by name."""
        return self._tools.get(tool_name)

    def list_tool_names(self) -> List[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())

    def get_openai_tools(self, filter_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Export tool definitions for OpenAI Function Calling."""
        tools = []
        target_names = filter_names or list(self._tools.keys())
        for name in target_names:
            if name in self._tools:
                tools.append(self._tools[name].get_openai_tool_definition())
        return tools

    async def execute(self, tool_name: str, params: Union[Dict[str, Any], str], **kwargs) -> str:
        """
        Execute tool asynchronously by name with defensive error handling.
        """
        tool = self.get(tool_name)
        if not tool:
            return f"[Tool Error]: Tool '{tool_name}' is not registered in ToolRegistry."

        try:
            # Check if custom acall is directly an async function
            if inspect.iscoroutinefunction(tool.call):
                return await tool.call(params, **kwargs)
            return await tool.acall(params, **kwargs)
        except Exception as e:
            logger.error(f"Execution error in tool '{tool_name}': {e}")
            return f"[{tool_name} Error]: Execution failed with error: {str(e)}"
