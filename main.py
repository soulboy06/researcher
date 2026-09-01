# -*- coding: utf-8 -*-
"""
WebResearcher CLI & Web Server Master Entrypoint.
Usage:
    python researcher/main.py serve [--port 8000] [--host 0.0.0.0]
    python researcher/main.py run "研究课题" [--mode iter|webweaver|tts|react] [--rounds 5]
"""
import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from researcher.config import logger
from researcher.agents.iter_agent import IterResearchAgent
from researcher.agents.web_weaver import WebWeaverAgent
from researcher.agents.tts_agent import TestTimeScalingAgent
from researcher.agents.react_agent import ReactAgent


async def run_cli_research(question: str, mode: str = "iter", max_rounds: int = 5):
    """Run CLI research directly from terminal."""
    print("\n" + "=" * 60)
    print(f"🔬 WebResearcher 启动研究 (模式: {mode.upper()})")
    print(f"课题: {question}")
    print("=" * 60 + "\n")

    agent = None
    if mode == "webweaver":
        p_rounds = max(2, max_rounds // 2)
        w_rounds = max(2, max_rounds - p_rounds)
        agent = WebWeaverAgent(planner_rounds=p_rounds, writer_rounds=w_rounds)
    elif mode == "tts":
        agent = TestTimeScalingAgent(num_parallel_agents=2, branch_max_rounds=max(2, max_rounds // 2))
    elif mode == "react":
        agent = ReactAgent(max_rounds=max_rounds)
    else:
        agent = IterResearchAgent(max_rounds=max_rounds)

    async def _progress_cb(event):
        t = event.get("type", "")
        p = event.get("phase", "")
        r = event.get("round", "")
        if p:
            print(f"\n⚡ 【阶段】: {event.get('message', p)}")
        elif t == "round":
            print(f"  [Round {r}] 📋 规划: {event.get('plan', '')[:60]}...")
        elif t == "tool":
            print(f"  [Round {r}] 🛠️ 工具: {event.get('tool_name')}")
        elif t == "outline":
            print(f"  [Round {r}] 📋 大纲已更新...")
        elif t == "write":
            print(f"  [Round {r}] ✍️ 章节撰写中...")

    try:
        res = await agent.run(question, progress_callback=_progress_cb)
        print("\n" + "=" * 60)
        print("📄 最终研究成果报告:")
        print("=" * 60)
        print(res.prediction)
        print("=" * 60)
        print(f"\n✨ 耗时: {res.elapsed_time:.1f}s | 消耗 Token: {res.total_tokens} | 状态: {res.termination}")
    finally:
        if hasattr(agent, "close"):
            await agent.close()


def main():
    parser = argparse.ArgumentParser(description="WebResearcher CLI & Web Server")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI Web UI server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run research directly in CLI")
    run_parser.add_argument("question", type=str, help="Research question")
    run_parser.add_argument("--mode", type=str, default="iter", choices=["iter", "webweaver", "tts", "react"], help="Research paradigm")
    run_parser.add_argument("--rounds", type=int, default=5, help="Max research rounds")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        print(f"\n🚀 启动 WebResearcher 现代简约 Web 服务: http://{args.host}:{args.port}")
        uvicorn.run("researcher.server.app:app", host=args.host, port=args.port, reload=False)
    elif args.command == "run":
        asyncio.run(run_cli_research(args.question, mode=args.mode, max_rounds=args.rounds))
    else:
        # Default to starting server
        import uvicorn
        print(f"\n🚀 启动 WebResearcher 现代简约 Web 服务: http://127.0.0.1:8000")
        uvicorn.run("researcher.server.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
