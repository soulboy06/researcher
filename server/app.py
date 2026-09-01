# -*- coding: utf-8 -*-
"""
FastAPI Server Entrypoint.
Provides RESTful APIs, SSE streaming endpoints, and static UI file serving.
"""
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from researcher.config import settings, logger
from researcher.server.task_manager import task_manager

app = FastAPI(
    title="WebResearcher Deep Research Agent API",
    description="Production-grade Autonomous Research Agent API with SSE Live Streaming",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


class CreateTaskRequest(BaseModel):
    question: str = Field(..., description="The complex research question to investigate")
    mode: str = Field(default="iter", description="Research paradigm: 'iter', 'react', 'webweaver', 'tts'")
    max_rounds: int = Field(default=5, ge=1, le=20, description="Max exploration rounds")
    instruction: Optional[str] = Field(default="", description="Optional custom focus instruction")
    num_parallel_agents: Optional[int] = Field(default=2, ge=2, le=5, description="Number of parallel agents for TTS mode")


@app.get("/api/health")
async def health_check():
    """Service health check."""
    return {"status": "ok", "version": "1.0.0", "default_model": settings.llm_model_name}


@app.get("/api/research/modes")
async def get_supported_modes():
    """Get metadata for supported research paradigms."""
    return {
        "modes": [
            {
                "id": "iter",
                "name": "IterResearch (推荐/核心范式)",
                "badge": "高效精简",
                "description": "固定精简上下文迭代状态机，每轮生成规划+报告+动作，节约 50% Token 并彻底杜绝长上下文击穿。",
                "best_for": "大多数通用深度搜索与复杂问题分析",
            },
            {
                "id": "webweaver",
                "name": "WebWeaver (双智能体大纲撰写)",
                "badge": "学术长文",
                "description": "大纲规划者 (Planner) 动态拓扑构建大纲 + 撰写者 (Writer) 深度拉取证据，支持 100% 行内学术引用溯源。",
                "best_for": "行业深度研报、长篇综述、多章节结构化长文",
            },
            {
                "id": "tts",
                "name": "Test-Time Scaling (多路计算扩展)",
                "badge": "超高置信度",
                "description": "多路智能体配置温度梯度并发探索，结合首席裁决大模型进行事实交叉比对与矛盾消解，消除单点幻觉。",
                "best_for": "高价值关键决策、前沿科学争端、多跳推理难题",
            },
            {
                "id": "react",
                "name": "ReAct (经典基准模式)",
                "badge": "基线对比",
                "description": "传统的多轮思考-行动-观察累加范式，用于学术对比与基准消融评测。",
                "best_for": "基准对比与简单工具调用",
            },
        ]
    }


@app.post("/api/research/start")
async def start_research(req: CreateTaskRequest):
    """
    Create a new research task and start execution in background.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    task = task_manager.create_task(
        question=req.question,
        mode=req.mode,
        max_rounds=req.max_rounds,
        instruction=req.instruction or "",
        num_parallel_agents=req.num_parallel_agents or 2,
    )
    started = task_manager.start_task(task.task_id)
    if not started:
        raise HTTPException(status_code=500, detail="Failed to launch task executor")

    return {
        "task_id": task.task_id,
        "mode": task.mode,
        "status": task.status,
        "question": task.question,
        "stream_url": f"/api/research/stream/{task.task_id}",
    }


@app.get("/api/research/stream/{task_id}")
async def stream_research_events(task_id: str):
    """
    Server-Sent Events (SSE) streaming endpoint for live research updates.
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return StreamingResponse(
        task_manager.stream_events(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/research/status/{task_id}")
async def get_task_status(task_id: str):
    """Query task status and final results."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    res = {
        "task_id": task.task_id,
        "status": task.status,
        "question": task.question,
        "mode": task.mode,
        "created_at": task.created_at,
        "error": task.error_message,
    }
    if task.result:
        res["result"] = {
            "prediction": task.result.prediction,
            "report": task.result.report,
            "total_tokens": task.result.total_tokens,
            "total_steps": task.result.total_steps,
            "elapsed_time": task.result.elapsed_time,
            "termination": task.result.termination,
        }
    return res


@app.get("/api/research/tasks")
async def list_recent_tasks():
    """List all tasks in memory with summary."""
    items = []
    for tid, t in task_manager.tasks.items():
        items.append({
            "task_id": tid,
            "mode": t.mode,
            "status": t.status,
            "question": t.question,
            "created_at": t.created_at,
            "steps": t.result.total_steps if t.result else 0,
            "prediction_length": len(t.result.prediction) if t.result else 0,
            "error": t.error,
        })
    return {"tasks": items}


@app.post("/api/research/cancel/{task_id}")
async def cancel_research(task_id: str):
    """Cancel a running task."""
    success = task_manager.cancel_task(task_id)
    return {"task_id": task_id, "cancelled": success}


# Mount static web directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve modern minimalist single-page dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>WebResearcher API Server is Running</h1><p>Static UI dashboard under construction.</p>"
    )
