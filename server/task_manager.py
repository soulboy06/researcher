# -*- coding: utf-8 -*-
"""
Asynchronous Research Task Manager and SSE Event Broadcasting Engine.
Manages task lifecycles, background execution, and real-time streaming queues.
"""
import asyncio
from dataclasses import dataclass, field
import datetime
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import uuid

from researcher.config import logger
from researcher.agents.base import AgentResult
from researcher.agents.iter_agent import IterResearchAgent
from researcher.agents.react_agent import ReactAgent
from researcher.agents.web_weaver import WebWeaverAgent
from researcher.agents.tts_agent import TestTimeScalingAgent


@dataclass
class ResearchTask:
    """Represents a background research session."""

    task_id: str
    question: str
    mode: str                     # 'iter', 'react', 'webweaver', 'tts'
    status: str = "pending"       # 'pending', 'running', 'completed', 'failed', 'cancelled'
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[AgentResult] = None
    error_message: Optional[str] = None
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    async_task: Optional[asyncio.Task] = None


class TaskManager:
    """
    Central Task Manager orchestrating multi-mode research runs and SSE dispatching.
    """

    def __init__(self):
        self.tasks: Dict[str, ResearchTask] = {}

    def create_task(
        self,
        question: str,
        mode: str = "iter",
        max_rounds: int = 5,
        temperature: Optional[float] = None,
        instruction: str = "",
        model_name: Optional[str] = None,
        **kwargs,
    ) -> ResearchTask:
        """Create and register a new research task."""
        task_id = str(uuid.uuid4())[:8]
        params = {
            "max_rounds": max_rounds,
            "temperature": temperature,
            "instruction": instruction,
            "model_name": model_name,
            **kwargs,
        }

        task = ResearchTask(
            task_id=task_id,
            question=question,
            mode=mode.lower().strip(),
            params=params,
        )
        self.tasks[task_id] = task
        logger.info(f"[TaskManager] Created task {task_id} (Mode: {task.mode}) for: '{question[:60]}'")
        return task

    def get_task(self, task_id: str) -> Optional[ResearchTask]:
        """Retrieve task by ID."""
        return self.tasks.get(task_id)

    async def _execute_task_pipeline(self, task: ResearchTask):
        """Worker executing selected agent paradigm and streaming events."""
        task.status = "running"
        await task.event_queue.put({
            "type": "status",
            "status": "running",
            "message": f"任务已启动，正在使用【{task.mode.upper()}】模式进行深度研究...",
        })

        async def _progress_cb(event: Dict[str, Any]):
            await task.event_queue.put(event)

        agent = None
        try:
            mode = task.mode
            max_rounds = task.params.get("max_rounds", 5)
            instruction = task.params.get("instruction", "")
            model_name = task.params.get("model_name")

            if mode == "react":
                agent = ReactAgent(
                    max_rounds=max_rounds,
                    instruction=instruction,
                    model_name=model_name,
                )
            elif mode == "webweaver":
                planner_rounds = max(2, max_rounds // 2)
                writer_rounds = max(2, max_rounds - planner_rounds)
                agent = WebWeaverAgent(
                    planner_rounds=planner_rounds,
                    writer_rounds=writer_rounds,
                    instruction=instruction,
                    model_name=model_name,
                )
            elif mode == "tts":
                agent = TestTimeScalingAgent(
                    num_parallel_agents=task.params.get("num_parallel_agents", 2),
                    branch_max_rounds=max(2, max_rounds // 2),
                    instruction=instruction,
                    model_name=model_name,
                )
            else:
                # Default: IterResearchAgent
                agent = IterResearchAgent(
                    max_rounds=max_rounds,
                    instruction=instruction,
                    model_name=model_name,
                )

            # Run agent
            result = await agent.run(task.question, progress_callback=_progress_cb)
            task.result = result
            task.status = "completed"

            # Final SSE completion notification
            await task.event_queue.put({
                "type": "complete",
                "status": "completed",
                "prediction": result.prediction,
                "report": result.report,
                "total_tokens": result.total_tokens,
                "total_steps": result.total_steps,
                "elapsed_time": result.elapsed_time,
            })

        except asyncio.CancelledError:
            task.status = "cancelled"
            task.error_message = "任务被用户主动取消"
            await task.event_queue.put({"type": "status", "status": "cancelled", "message": "任务已取消"})
            logger.warning(f"[TaskManager] Task {task.task_id} was cancelled.")
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            logger.error(f"[TaskManager] Task {task.task_id} failed: {e}")
            await task.event_queue.put({"type": "error", "status": "failed", "error": str(e)})
        finally:
            if agent and hasattr(agent, "close"):
                try:
                    await agent.close()
                except Exception:
                    pass
            # Sentinel value to mark stream completion
            await task.event_queue.put(None)

    def start_task(self, task_id: str) -> bool:
        """Spawn background asyncio task."""
        task = self.get_task(task_id)
        if not task or task.status != "pending":
            return False

        task.async_task = asyncio.create_task(self._execute_task_pipeline(task))
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel running task."""
        task = self.get_task(task_id)
        if not task or not task.async_task:
            return False

        if not task.async_task.done():
            task.async_task.cancel()
            return True
        return False

    async def stream_events(self, task_id: str) -> AsyncGenerator[str, None]:
        """
        SSE Generator yielding event chunks: `data: {...}\n\n`.
        """
        task = self.get_task(task_id)
        if not task:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Task not found'})}\n\n"
            return

        while True:
            event = await task.event_queue.get()
            if event is None:
                # End of stream
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


task_manager = TaskManager()
