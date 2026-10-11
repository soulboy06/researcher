# WebResearcher (IterResearch)

A modular, async deep-research agent system. It calls search, page-reading, evidence-extraction, and Python
tools so an LLM can research a complex question over multiple rounds — with live progress in a web console.

**Live demo:** https://researcher.soulboy.xyz

## Research modes

- **IterResearch** — fixed-context iterative research; the default for most questions
- **WebWeaver** — planner drafts an outline, then a writer agent composes the long-form report
- **Test-Time Scaling** — runs multiple research branches in parallel, then a synthesizer adjudicates
- **ReAct** — classic think/act/observe loop, useful as a baseline

## Capabilities

- Search fallback chain: Tavily → Serper → Baidu
- Page fetching, content cleaning, evidence extraction, citation assembly
- FastAPI service with SSE real-time progress and a minimal web console
- OpenAI-compatible API — works with OpenAI, DeepSeek, vLLM, and similar endpoints

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set LLM_API_KEY and any search-service keys
```

Serve the web UI:

```bash
python main.py serve --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

Or run from the CLI:

```bash
python main.py run "Analyze the impact of AI on the EV industry" --mode iter --rounds 5
```

## API

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET | `/api/health` | health check |
| GET | `/api/research/modes` | list research modes |
| POST | `/api/research/start` | create a research task |
| GET | `/api/research/stream/{task_id}` | SSE live progress |
| GET | `/api/research/status/{task_id}` | task status + final report |
| POST | `/api/research/cancel/{task_id}` | cancel a task |

## Key design decisions

- **Fixed-context state machine.** Each round is a single LLM call that jointly emits Plan / Report / Action.
  The workspace retains only Question + rolling Report + last observation, so long research chains don't
  blow up the context window — new evidence is folded into the rolling report before the next round.
- **Dual-protocol structured output.** Accepts OpenAI function calling, XML / DSML `tool_call`, and JSON5
  payloads, with auto-repair for markdown fences, truncated JSON, and hoisted params — all normalized into
  one `ToolRegistry`.
- **Degradable everything.** Every external dependency has a fallback chain; per-error-type retry vs.
  fail-fast in the LLM client; research supports timeouts, max rounds, forced convergence, cancellation,
  and resume-from-breakpoint.

## Layout

```
agents/   agents for each research paradigm
core/     LLM client, prompts, context guards, protocol parsing
tools/    search, page reading, evidence extraction, memory, compute
schema/   message / tool-call / result data structures
server/   FastAPI endpoints, task manager, static pages
config/   env vars and runtime config
main.py   CLI + web entry
```

## Config

- `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME` — primary model
- `SUMMARY_MODEL_NAME` — synthesis model for long reports
- `TAVILY_API_KEY`, `SERPER_API_KEY`, `JINA_API_KEY` — optional external services
- `MAX_INPUT_TOKENS`, `OBS_MAX_CHARS` — context and observation guards

Commit only `.env.example`, never `.env` or logs. The task manager currently keeps state in process
memory — add persistent task storage, auth, queues, and multi-instance coordination for production use.
