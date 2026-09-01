# -*- coding: utf-8 -*-
"""
Deterministic Python Execution Sandbox.
Provides isolated execution for complex arithmetic, date calculations, and data processing.
"""
import concurrent.futures
import contextlib
import io
import json
import re
import traceback
from typing import Any, Dict, Optional, Union

from researcher.config import logger
from researcher.tools.base import BaseTool


class PythonSandboxTool(BaseTool):
    """
    Deterministic Python Code Interpreter.
    Executes Python scripts for accurate mathematical computation, string manipulation, and dates.
    """

    name = "python"
    description = (
        "Executes a Python code snippet and returns its standard output and errors. "
        "Use this for deterministic calculations, counting, date/time differences, and data filtering."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Valid Python 3 source code to execute. Use print() to display results.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def _extract_code_from_input(self, params: Union[Dict[str, Any], str]) -> str:
        """Extract clean Python code string from dict, json, or markdown block."""
        code_str = ""
        if isinstance(params, dict):
            code_str = params.get("code", "")
        elif isinstance(params, str):
            # Check if JSON string
            try:
                parsed = json.loads(params)
                if isinstance(parsed, dict) and "code" in parsed:
                    code_str = parsed["code"]
            except Exception:
                pass

            if not code_str:
                # Check for <code>...</code> or ```python ... ```
                if "<code>" in params and "</code>" in params:
                    code_str = params.split("<code>", 1)[1].rsplit("</code>", 1)[0]
                elif "```python" in params:
                    code_str = params.split("```python", 1)[1].rsplit("```", 1)[0]
                elif "```" in params:
                    code_str = params.split("```", 1)[1].rsplit("```", 1)[0]
                else:
                    code_str = params

        return code_str.strip()

    def _run_code_internal(self, code: str) -> str:
        """Execute Python code in an isolated scope and capture stdout/stderr."""
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Build isolated sandbox environment with preloaded utilities
        global_scope: Dict[str, Any] = {
            "__name__": "__sandbox__",
            "__doc__": None,
        }
        # Preload common safe calculation modules
        safe_modules = [
            "math", "datetime", "re", "json", "random",
            "collections", "itertools", "functools", "statistics"
        ]
        for mod_name in safe_modules:
            try:
                global_scope[mod_name] = __import__(mod_name)
            except ImportError:
                pass

        local_scope: Dict[str, Any] = {}

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                # Try evaluating as an expression first
                try:
                    expr_result = eval(code, global_scope, local_scope)
                    if expr_result is not None:
                        stdout_buffer.write(str(expr_result) + "\n")
                except SyntaxError:
                    # Fallback to statement execution
                    exec(code, global_scope, local_scope)

            stdout_val = stdout_buffer.getvalue().strip()
            stderr_val = stderr_buffer.getvalue().strip()

            if stderr_val and not stdout_val:
                return f"[Python Stderr]:\n{stderr_val}"
            elif stderr_val and stdout_val:
                return f"[Python Output]:\n{stdout_val}\n\n[Python Warnings/Stderr]:\n{stderr_val}"
            elif stdout_val:
                return f"[Python Output]:\n{stdout_val}"
            else:
                return "[Python Output]: Code executed successfully with no stdout output. Tip: Use print(...) to display results."

        except Exception:
            err_msg = traceback.format_exc()
            return f"[Python Runtime Error]:\n{err_msg}"

    def call(self, params: Union[Dict[str, Any], str], **kwargs) -> str:
        """
        Execute code within timeout constraints.
        """
        code = self._extract_code_from_input(params)
        if not code:
            return "[Python Error]: No code provided to execute. Supply valid Python code in 'code' parameter."

        logger.debug(f"[PythonSandbox] Executing code snippet:\n{code[:200]}...")

        # Run with timeout protection
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_code_internal, code)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                return f"[Python Error]: Code execution timed out after {self.timeout} seconds."
            except Exception as e:
                return f"[Python Error]: Execution failed unexpectedly: {str(e)}"
