# -*- coding: utf-8 -*-
"""
Robust Dual-Protocol Parser and Schema Auto-Repair Engine.
Handles:
1. XML Structured Protocol parsing with multi-level regex reverse fallback
2. DeepSeek DSML tool call tokens (<｜｜DSML｜｜tool_call>) compatibility
3. Truncated & Malformed JSON automatic structural repair
4. Fault-tolerant JSON5 parsing (unquoted keys, trailing commas, single quotes)
5. Parameter Auto-Repair (correcting schema omission when models leak top-level arguments)
6. Native Function Calling <-> XML Protocol bidirectional mapping
"""
from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False

from researcher.config import logger


# Known common tool parameter keys across all tools
KNOWN_ARG_KEYS = {
    "query",
    "url",
    "goal",
    "code",
    "content",
    "citation_ids",
    "max_chunks",
    "max_results",
    "files",
    "path",
    "question",
}


@dataclass
class ParsedOutput:
    """Standard parsed output model for agent reasoning steps."""

    plan: str = ""
    report: str = ""
    outline: str = ""
    tool_call_raw: str = ""
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None
    terminate: bool = False
    terminate_reason: str = ""
    raw_content: str = ""

    @property
    def has_tool_call(self) -> bool:
        """Check if output contains an actionable tool invocation."""
        return bool(self.tool_name and (self.tool_args is not None))

    @property
    def has_final_answer(self) -> bool:
        """Check if output completed research with a final answer."""
        return bool(self.answer and self.answer.strip())


class ProtocolParser:
    """
    Unified parser for XML protocol tags and JSON tool execution payloads.
    """

    # Multi-dialect tool call regex: standard XML + DeepSeek DSML tokens + markdown JSON fences
    TOOL_CALL_PATTERNS = [
        r"<\s*tool_call\s*>(.*?)<\s*/\s*tool_call\s*>",
        r"(?:<[｜|]{1,2}DSML[｜|]{1,2}tool_call>|<[｜|]{1,2}tool_call[｜|]{1,2}>)(.*?)(?:<[｜|]{1,2}DSML[｜|]{1,2}call_end>|<[｜|]{1,2}call_end[｜|]{1,2}>|<\s*/\s*tool_call\s*>|$)",
    ]

    @staticmethod
    def extract_last_block(pattern: str, text: str) -> str:
        """
        Extract the last non-empty match of a regex pattern.
        Prevents earlier self-correction ramblings from confusing the state machine.
        """
        if not text:
            return ""
        matches = re.findall(pattern, text, flags=re.DOTALL | re.MULTILINE)
        for m in reversed(matches):
            if isinstance(m, str) and m.strip():
                return m.strip()
            elif isinstance(m, tuple):
                for sub in m:
                    if sub and sub.strip():
                        return sub.strip()
        return ""

    @classmethod
    def extract_tool_call_block(cls, text: str) -> str:
        """Extract tool call using multi-dialect patterns with fallback."""
        for pattern in cls.TOOL_CALL_PATTERNS:
            res = cls.extract_last_block(pattern, text)
            if res:
                return res

        # Fallback: look for naked JSON structure containing known tool names
        naked_pattern = r"(\{\s*\"(?:name|tool)\"\s*:\s*\"(?:search|visit|python|extract_evidence|retrieve)\"[\s\S]*?\})"
        naked_match = cls.extract_last_block(naked_pattern, text)
        if naked_match:
            return naked_match

        return ""

    @classmethod
    def _attempt_repair_truncated_json(cls, s: str) -> Optional[Dict[str, Any]]:
        """
        Heuristically repair cut-off or truncated JSON strings.
        e.g., {"name": "visit", "arguments": {"url": ["https://..."], "goal": ...
        """
        s = s.strip()
        if not s.startswith("{"):
            return None

        # 1. Regex regex extraction fallback for common parameters
        name_match = re.search(r'"(?:name|tool)"\s*:\s*"([^"]+)"', s)
        if not name_match:
            return None
        tool_name = name_match.group(1)

        args_dict: Dict[str, Any] = {}

        # Extract "query": [...] or "query": "..."
        query_match = re.search(r'"query"\s*:\s*(\[[^\]]*\]|"[^"]*")', s)
        if query_match:
            try:
                args_dict["query"] = json.loads(query_match.group(1))
            except Exception:
                pass

        # Extract "url": [...] or "url": "..."
        url_match = re.search(r'"url"\s*:\s*(\[[^\]]*\]|"[^"]*")', s)
        if url_match:
            try:
                args_dict["url"] = json.loads(url_match.group(1))
            except Exception:
                pass

        # Extract "goal": "..."
        goal_match = re.search(r'"goal"\s*:\s*"([^"]*)"', s)
        if goal_match:
            args_dict["goal"] = goal_match.group(1)

        # Extract "code": "..."
        code_match = re.search(r'"code"\s*:\s*"([^"]*)"', s)
        if code_match:
            args_dict["code"] = code_match.group(1)

        if args_dict:
            logger.info(f"[AutoRepair] Reconstructed truncated tool call for '{tool_name}' with fields: {list(args_dict.keys())}")
            return {"name": tool_name, "arguments": args_dict}

        # 2. Structural closing repair: append missing quotes and braces
        repaired = s
        if repaired.count('"') % 2 == 1:
            repaired += '"'
        open_braces = repaired.count("{") - repaired.count("}")
        if open_braces > 0:
            repaired += "}" * open_braces

        try:
            if HAS_JSON5:
                return json5.loads(repaired)
            return json.loads(repaired)
        except Exception:
            return None

    @classmethod
    def parse_tool_payload(cls, payload_str: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Parse and repair tool call string (handles JSON, JSON5, DSML tokens, truncated JSON, and <code> tags).
        
        Returns:
            (tool_name, tool_arguments_dict)
        """
        if not payload_str or not payload_str.strip():
            return None, {}

        payload_str = payload_str.strip()

        # Clean any remaining DSML or XML artifacts
        payload_str = re.sub(r"^<[｜|]{1,2}[^>]+>\s*", "", payload_str)
        payload_str = re.sub(r"\s*<[｜|]{1,2}[^>]+>$", "", payload_str).strip()

        # 1. Check for <code>...</code> tag (Direct Python code execution shorthand)
        if "<code>" in payload_str and "</code>" in payload_str:
            code_body = payload_str.split("<code>", 1)[1].rsplit("</code>", 1)[0].strip()
            return "python", {"code": code_body}

        # 2. Parse JSON/JSON5 data
        parsed_dict = None
        if HAS_JSON5:
            try:
                parsed_dict = json5.loads(payload_str)
            except Exception:
                pass

        if parsed_dict is None:
            try:
                parsed_dict = json.loads(payload_str)
            except Exception:
                # Try cleaning markdown code block fences
                cleaned = payload_str
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in cleaned:
                    cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
                try:
                    parsed_dict = json.loads(cleaned)
                except Exception:
                    # Try heuristic repair for cut-off / truncated JSON
                    parsed_dict = cls._attempt_repair_truncated_json(payload_str)

        if not isinstance(parsed_dict, dict):
            return None, {}

        tool_name = parsed_dict.get("name") or parsed_dict.get("tool") or ""
        tool_args = parsed_dict.get("arguments") or parsed_dict.get("parameters")

        # 3. 🛠️ Parameter Auto-Repair Engine:
        # Handles small models outputting {"name": "search", "query": ["..."]} without "arguments"
        if tool_args is None or not isinstance(tool_args, dict):
            remaining_keys = {
                k: v for k, v in parsed_dict.items() if k not in ["name", "tool", "type"]
            }
            if remaining_keys:
                tool_args = remaining_keys
                logger.info(
                    f"[AutoRepair] Automatically wrapped top-level keys {list(remaining_keys.keys())} into 'arguments' for tool '{tool_name}'"
                )
            else:
                tool_args = {}

        return tool_name, tool_args

    @classmethod
    def parse_iterresearch(cls, text: str) -> ParsedOutput:
        """
        Parse IterResearch three-part output format:
        <plan>...</plan>
        <report>...</report>
        <tool_call>...</tool_call> | <answer>...</answer> | <terminate>...</terminate>
        """
        res = ParsedOutput(raw_content=text)

        # 1. Extract <plan>
        res.plan = cls.extract_last_block(r"<\s*plan\s*>(.*?)<\s*/\s*plan\s*>", text)

        # 2. Extract <report>
        res.report = cls.extract_last_block(r"<\s*report\s*>(.*?)<\s*/\s*report\s*>", text)

        # 3. Extract <write_outline> (for WebWeaver)
        res.outline = cls.extract_last_block(r"<\s*write_outline\s*>(.*?)<\s*/\s*write_outline\s*>", text)

        # 4. Extract Action Tags
        res.tool_call_raw = cls.extract_tool_call_block(text)
        res.answer = cls.extract_last_block(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", text)
        term_text = cls.extract_last_block(r"<\s*terminate\s*>(.*?)<\s*/\s*terminate\s*>", text)

        if term_text:
            res.terminate = True
            res.terminate_reason = term_text

        # Parse tool details if tool_call present
        if res.tool_call_raw:
            name, args = cls.parse_tool_payload(res.tool_call_raw)
            res.tool_name = name
            res.tool_args = args

        return res

    @classmethod
    def parse_react(cls, text: str) -> ParsedOutput:
        """
        Parse standard ReAct output format:
        Thought: ...
        <tool_call>...</tool_call> | <answer>...</answer>
        """
        res = ParsedOutput(raw_content=text)

        # Extract thought (either via <thought> tag or leading text before tool_call)
        thought = cls.extract_last_block(r"<\s*thought\s*>(.*?)<\s*/\s*thought\s*>", text)
        if not thought:
            cleaned = re.sub(r"<\s*(tool_call|answer|terminate)\s*>.*", "", text, flags=re.DOTALL)
            cleaned = re.sub(r"<[｜|]{1,2}.*", "", cleaned, flags=re.DOTALL).strip()
            thought = cleaned
        res.plan = thought

        # Extract Action Tags (supporting both standard XML and DSML)
        res.tool_call_raw = cls.extract_tool_call_block(text)
        res.answer = cls.extract_last_block(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", text)
        term_text = cls.extract_last_block(r"<\s*terminate\s*>(.*?)<\s*/\s*terminate\s*>", text)

        # 🛠️ Fallback 1: Alternative tags (<report>, <final_answer>, <response>, <result>)
        if not res.answer and not res.tool_call_raw:
            alt_answer = cls.extract_last_block(
                r"<\s*(?:report|final_answer|response|result)\s*>(.*?)<\s*/\s*(?:report|final_answer|response|result)\s*>",
                text,
            )
            if alt_answer and len(alt_answer.strip()) > 5:
                res.answer = alt_answer.strip()

        # 🛠️ Fallback 2: Implicit Markdown Answer detection
        # If no tool call was requested and the model produced a structured response (>50 chars or markdown headers),
        # safely treat it as the final answer rather than spinning empty loops.
        if not res.answer and not res.tool_call_raw and not term_text:
            cleaned_text = re.sub(r"<\s*thought\s*>.*?<\s*/\s*thought\s*>", "", text, flags=re.DOTALL)
            cleaned_text = re.sub(r"^Thought:.*?(?=\n\n|\n[#\d]|\n[A-Z]|$)", "", cleaned_text, flags=re.DOTALL | re.IGNORECASE).strip()
            if len(cleaned_text) > 50 or ("#" in cleaned_text and len(cleaned_text) > 20):
                res.answer = cleaned_text

        if term_text:
            res.terminate = True
            res.terminate_reason = term_text

        if res.tool_call_raw:
            name, args = cls.parse_tool_payload(res.tool_call_raw)
            res.tool_name = name
            res.tool_args = args

        return res
