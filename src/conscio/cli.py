from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree

from conscio.core.agent import ConsciousAgent
from conscio.core.cognition import InputEvent
from conscio.core.runtime import CognitiveRuntime
from conscio.eval import LIVE_SUITES, run_eval_suite
from conscio.memory.store import MemoryStore
from conscio.config import load_config, write_default_config

console = Console()


def _print_monologue(monologue_text: str) -> None:
    if not monologue_text.strip():
        return
    tree = Tree("🧠 Stream of Consciousness", guide_style="dim")
    lines = monologue_text.split("\n")
    for line in lines:
        if line.strip():
            prefix_map = {
                "👁": "observation",
                "💭": "reflection",
                "🎯": "intention",
                "✅": "evaluation",
                "📖": "learning",
                "🤔": "doubt",
                "⚡": "decision",
            }
            style = "dim"
            for emoji, label in prefix_map.items():
                if emoji in line:
                    style = {
                        "observation": "cyan",
                        "reflection": "yellow",
                        "intention": "green",
                        "evaluation": "blue",
                        "learning": "magenta",
                        "doubt": "red",
                        "decision": "bold green",
                    }.get(label, "dim")
                    line = line.replace(f"{emoji} ", f"[{style}]{emoji} [/]{label}: ", 1)
                    break
            tree.add(line)
    console.print(tree)


def _print_result(result) -> None:
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("Session", result.session_id[:12])
    info.add_row("Confidence", result.confidence)
    info.add_row("Rounds", str(result.rounds))
    info.add_row("Duration", f"{result.duration:.2f}s")
    if result.selected_action:
        info.add_row("Action", result.selected_action)
    if result.self_state:
        info.add_row("Uncertainty", f"{result.self_state.get('uncertainty', 0):.2f}")
        info.add_row("Conflict", f"{result.self_state.get('conflict_level', 0):.2f}")
    if result.attention_schema:
        info.add_row("Focus", result.attention_schema.get("focus", ""))
    console.print(Panel(info, title="⚡ Cycle Summary", border_style="dim"))

    if result.inner_monologue:
        console.print()
        _print_monologue(result.inner_monologue)

    console.print()
    console.print(Panel(Markdown(result.output), title="💬 Response", border_style="green"))

    if result.tool_results:
        console.print()
        tool_table = Table(title="🔧 Tool Calls", box=None)
        tool_table.add_column("Tool", style="cyan")
        tool_table.add_column("Result", style="dim", max_width=60)
        for tr in result.tool_results:
            tool_table.add_row(
                tr.get("tool", "?"),
                tr.get("output", "")[:120],
            )
        console.print(tool_table)

    if result.cognitive_trace:
        console.print()
        console.print(Panel(result.cognitive_trace, title="🧭 Cognitive Trace", border_style="dim"))


async def _run_interactive(
    name: str,
    persona: str,
    model: str | None,
    offline: bool = False,
) -> None:
    console.print(Panel.fit(
        f"[bold]conscio[/] — {name}\n"
        f"[dim]{persona or 'No persona set'}[/]",
        border_style="blue",
    ))
    agent = ConsciousAgent(name=name, persona=persona, model=model, use_llm=not offline)
    await agent.initialize()
    try:
        while True:
            try:
                user_input = console.input("\n[bold cyan]You:[/] ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Ending session...[/]")
                break
            if not user_input.strip():
                continue
            if user_input.strip().lower() in ("/exit", "/quit", "/q"):
                break
            if user_input.strip().lower() == "/clear":
                agent.workspace.clear()
                console.print("[dim]Workspace cleared.[/]")
                continue
            if user_input.strip().lower() == "/memory":
                episodes = await agent.memory.recent_episodes(10)
                for ep in episodes:
                    summary = ep.get("summary") or ep.get("input", "")
                    console.print(f"[dim]- {summary[:100]}[/]")
                continue
            console.print("[dim]Thinking...[/]")
            try:
                result = await agent.cycle(user_input)
                _print_result(result)
            except Exception as e:
                console.print(f"[red]Error during cycle: {e}[/]")
    finally:
        await agent.close()
        await asyncio.sleep(0.05)


async def _run_ask(
    input_text: str,
    name: str,
    persona: str,
    model: str | None,
    quiet: bool = False,
    offline: bool = False,
) -> None:
    agent = ConsciousAgent(name=name, persona=persona, model=model, use_llm=not offline)
    await agent.initialize()
    try:
        if not quiet:
            console.print("[dim]Thinking...[/]")
        result = await agent.cycle(input_text)
        if quiet:
            console.print(result.output)
        else:
            _print_result(result)
    finally:
        await agent.close()
        await asyncio.sleep(0.05)


async def _show_history() -> None:
    agent = ConsciousAgent()
    await agent.memory.initialize()
    try:
        sessions = await agent.memory.list_sessions()
        if not sessions:
            console.print("[yellow]No sessions found.[/]")
            return
        table = Table(title="Session History")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Created", style="dim")
        table.add_column("Summary", style="dim", max_width=60)
        for s in sessions:
            created = time.strftime(
                "%Y-%m-%d %H:%M",
                time.localtime(s.get("created_at", 0)),
            )
            table.add_row(
                s.get("id", "")[:12],
                s.get("name", "")[:20],
                created,
                (s.get("summary") or "")[:60],
            )
        console.print(table)
    finally:
        await agent.memory.close()


async def _search_memory(query: str) -> None:
    agent = ConsciousAgent()
    await agent.memory.initialize()
    try:
        from conscio.memory.search import search_memories
        result = await search_memories(agent.memory, query)
        console.print(Markdown(result))
    finally:
        await agent.memory.close()


async def _run_daemon_dry_run(events: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime = CognitiveRuntime(llm=None, memory=MemoryStore(db_path=f"{tmp}/daemon.db"))
        await runtime.initialize()
        try:
            input_events = [InputEvent(content=e, source="daemon", event_type="dry_run") for e in events]
            results = await runtime.run_daemon(input_events, dry_run=True)
            for result in results:
                console.print(Panel(result.output, title=f"Action: {result.selected_action}", border_style="green"))
                console.print(Panel(result.cognitive_trace, title="Cognitive Trace", border_style="dim"))
        finally:
            await runtime.close()


LIVE_EVAL_GATE_EXPLANATION = (
    "Live eval suites (ladder/ablations) make paid LLM calls and are double-gated:\n"
    "  1. pass --live on the command line, AND\n"
    "  2. set CONSCIO_EVAL_LIVE=1 in the environment.\n"
    "The stub suites (smoke, autonomy_long_horizon, goal_evolution, ssrf_rejection)\n"
    "never need either gate."
)


def _csv_list(value: str) -> list[str] | None:
    items = [item.strip() for item in (value or "").split(",") if item.strip()]
    return items or None


async def _run_live_eval(args: argparse.Namespace) -> None:
    if not args.live or os.environ.get("CONSCIO_EVAL_LIVE") != "1":
        raise SystemExit(LIVE_EVAL_GATE_EXPLANATION)

    from conscio.eval.judge import Judge
    from conscio.eval.runner import run_battery
    from conscio.llm.client import LLMClient

    agent_model = args.model or load_config().llm_model
    judge_model = args.judge_model
    run_id = args.run_id or f"{time.strftime('%Y-%m-%d')}_{args.suite}_{uuid.uuid4().hex[:4]}"
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_llm = LLMClient(model=agent_model)
    judge = Judge(
        LLMClient(model=judge_model),
        judge_model,
        out_dir / "judge_log.jsonl",
        agent_model=agent_model,
    )
    console.print(
        f"[bold]Live eval[/] run_id={run_id} suite={args.suite} "
        f"agent={agent_model} judge={judge_model}"
    )
    result = await run_battery(
        agent_llm=agent_llm,
        agent_model=agent_model,
        mode=args.suite,
        conditions=_csv_list(args.conditions),
        suites=_csv_list(args.tasks),
        seeds=args.seeds,
        judge=judge,
        out_dir=out_dir,
        run_id=run_id,
    )
    table = Table(title=f"Eval run {run_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    errored = sum(1 for r in result.records if r.error)
    passed = sum(1 for r in result.records if r.passed and not r.error)
    table.add_row("records", str(len(result.records)))
    table.add_row("passed", str(passed))
    table.add_row("errored", str(errored))
    table.add_row("agent calls", str(result.meta.total_agent_calls))
    table.add_row("judge calls", str(result.meta.total_judge_calls))
    table.add_row("est. cost", f"${result.meta.cost_estimate_usd:.2f}")
    table.add_row("wall time", f"{result.meta.wall_time_s:.0f}s")
    console.print(table)
    for name, path in result.paths.items():
        console.print(f"[dim]{name}: {path}[/]")


async def _run_eval(suite: str) -> None:
    rows = await run_eval_suite(suite)
    table = Table(title=f"Eval Suite: {suite}")
    table.add_column("Case", style="cyan")
    table.add_column("Mode")
    table.add_column("Pass")
    table.add_column("Action")
    table.add_column("Ticks")
    table.add_column("Attention")
    table.add_column("PredErr")
    for row in rows:
        table.add_row(
            row.name,
            row.mode,
            "yes" if row.passed else "no",
            row.selected_action,
            str(row.ticks),
            str(row.attention_selections),
            str(row.prediction_errors),
        )
    console.print(table)


def _service_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


async def _client_request(method: str, path: str, *, json_data: dict[str, Any] | None = None) -> Any:
    cfg = load_config()
    if not cfg.api_key:
        raise SystemExit("No API key configured. Run `conscio service init` first.")
    async with httpx.AsyncClient(base_url=cfg.base_url, timeout=60) as client:
        response = await client.request(method, path, headers=_service_headers(cfg.api_key), json=json_data)
        response.raise_for_status()
        return response.json()


async def _service_status() -> None:
    data = await _client_request("GET", "/status")
    table = Table(title="Conscio Service")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key in (
        "running",
        "paused",
        "session_id",
        "uptime",
        "autonomous",
        "unsafe_autonomy",
        "queue_depth",
        "current_event",
        "last_autonomous_action",
        "actions_last_hour",
        "episode_count",
        "last_error",
    ):
        table.add_row(key, str(data.get(key, "")))
    active = data.get("active_goal") or {}
    if active:
        table.add_row("active_goal", active.get("description", ""))
    console.print(table)


async def _client_chat(message: str) -> None:
    data = await _client_request("POST", "/message", json_data={"content": message})
    console.print(Panel(Markdown(data.get("output", "")), title="Conscio", border_style="green"))


async def _client_influence(kind: str, content: str) -> None:
    data = await _client_request("POST", f"/influence/{kind}", json_data={"content": content})
    console.print(Panel(str(data), title=f"Influence: {kind}", border_style="green"))


async def _client_control(action: str) -> None:
    data = await _client_request("POST", f"/control/{action}")
    console.print(data)


async def _client_goals() -> None:
    rows = await _client_request("GET", "/goals")
    table = Table(title="Goals")
    table.add_column("ID", style="cyan")
    table.add_column("Status")
    table.add_column("Source")
    table.add_column("Priority")
    table.add_column("Description")
    for row in rows:
        table.add_row(
            row.get("id", "")[:12],
            row.get("status", ""),
            row.get("source", ""),
            f"{row.get('priority', 0):.2f}",
            row.get("description", "")[:90],
        )
    console.print(table)


async def _client_influences() -> None:
    rows = await _client_request("GET", "/influences")
    table = Table(title="Influences")
    table.add_column("ID", style="cyan")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Content")
    table.add_column("Appraisal")
    for row in rows:
        table.add_row(
            row.get("id", "")[:12],
            row.get("kind", ""),
            row.get("status", ""),
            row.get("content", "")[:60],
            row.get("appraisal", "")[:80],
        )
    console.print(table)


async def _client_projects(project_id: str | None = None) -> None:
    if project_id:
        project = await _client_request("GET", f"/projects/{project_id}")
        console.print(Panel(str(project), title=f"Project {project_id[:12]}", border_style="green"))
        return
    rows = await _client_request("GET", "/projects")
    table = Table(title="Projects")
    table.add_column("ID", style="cyan")
    table.add_column("Goal", style="dim")
    table.add_column("Status")
    table.add_column("Title")
    for row in rows:
        table.add_row(
            row.get("id", "")[:12],
            row.get("goal_id", "")[:12],
            row.get("status", ""),
            row.get("title", "")[:90],
        )
    console.print(table)


async def _client_tick() -> None:
    data = await _client_request("POST", "/autonomy/tick")
    console.print(Panel(Markdown(data.get("output", "")), title=f"Autonomy: {data.get('selected_action', '')}", border_style="green"))


async def _client_trace() -> None:
    data = await _client_request("GET", "/trace")
    console.print(Panel(data.get("trace", ""), title="Cognitive Trace", border_style="dim"))


def _service_init() -> None:
    path = write_default_config()
    console.print(f"[green]Config ready:[/] {path}")
    console.print("[dim]Unsafe autonomy is disabled until config.toml sets unsafe_autonomy = true.[/]")


def _service_start() -> None:
    import uvicorn

    from conscio.api import create_app

    cfg = load_config()
    try:
        cfg.validate_public_bind()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    app = create_app(config=cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="conscio — a conscious autonomous agent runtime",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    run_p = sub.add_parser("run", help="Start an interactive conscious agent session")
    run_p.add_argument("--name", default="Conscio", help="Agent name")
    run_p.add_argument("--persona", default="", help="Agent persona/backstory")
    run_p.add_argument("--model", default=None, help="LLM model to use")
    run_p.add_argument("--offline", action="store_true", help="Disable LLM calls and use deterministic modules")

    ask_p = sub.add_parser("ask", help="Ask a single question")
    ask_p.add_argument("input", nargs="+", help="Question to ask")
    ask_p.add_argument("--name", default="Conscio", help="Agent name")
    ask_p.add_argument("--persona", default="", help="Agent persona/backstory")
    ask_p.add_argument("--model", default=None, help="LLM model to use")
    ask_p.add_argument("--quiet", action="store_true", help="Only print the answer")
    ask_p.add_argument("--offline", action="store_true", help="Disable LLM calls and use deterministic modules")

    sub.add_parser("history", help="Show past sessions")
    search_p = sub.add_parser("search", help="Search across memories")
    search_p.add_argument("query", nargs="+", help="Search query")
    daemon_p = sub.add_parser("daemon", help="Run daemon dry-run events")
    daemon_p.add_argument("--dry-run", action="store_true", default=True, help="Process events without unsafe autonomy")
    daemon_p.add_argument("events", nargs="*", default=["Daemon dry-run heartbeat"], help="Events to process")
    eval_p = sub.add_parser("eval", help="Run built-in evaluation suites")
    eval_p.add_argument(
        "--suite",
        default="smoke",
        help="Stub suite name (smoke, autonomy_long_horizon, goal_evolution, "
        "ssrf_rejection) or a live battery suite (ladder, ablations)",
    )
    eval_p.add_argument(
        "--conditions",
        default="",
        help="Comma-separated condition names (B0,B1,B2,B3,B4,abl_*) for live suites",
    )
    eval_p.add_argument("--seeds", type=int, default=1, help="Seeds for temperature>0 tasks")
    eval_p.add_argument(
        "--tasks",
        default="",
        help="Comma-separated battery suites to run (e.g. constraints,memory)",
    )
    eval_p.add_argument("--out", default="docs/results", help="Results output directory")
    eval_p.add_argument("--model", default="", help="Agent model for live suites")
    eval_p.add_argument("--judge-model", default="qwen3.6-27b", help="Judge model (must differ from agent)")
    eval_p.add_argument("--run-id", default="", help="Run id (default: date_suite_rand)")
    eval_p.add_argument(
        "--live",
        action="store_true",
        help="Required (with CONSCIO_EVAL_LIVE=1) to run paid live suites",
    )

    service_p = sub.add_parser("service", help="Manage the long-running Conscio service")
    service_sub = service_p.add_subparsers(dest="service_command")
    service_sub.add_parser("init", help="Create ~/.conscio/config.toml")
    service_sub.add_parser("start", help="Start the authenticated FastAPI service")
    service_sub.add_parser("status", help="Show service status")
    service_sub.add_parser("stop", help="Stop the service")

    chat_p = sub.add_parser("chat", help="Send a message to the running service")
    chat_p.add_argument("message", nargs="+", help="Message to send")
    influence_p = sub.add_parser("influence", help="Influence the running service")
    influence_sub = influence_p.add_subparsers(dest="influence_kind")
    influence_goal_p = influence_sub.add_parser("goal", help="Submit a goal influence")
    influence_goal_p.add_argument("content", nargs="+")
    influence_constraint_p = influence_sub.add_parser("constraint", help="Submit a constraint influence")
    influence_constraint_p.add_argument("content", nargs="+")
    sub.add_parser("pause", help="Pause autonomous action")
    sub.add_parser("resume", help="Resume autonomous action")
    sub.add_parser("goals", help="Show service goals")
    sub.add_parser("influences", help="Show service influences")
    projects_p = sub.add_parser("projects", help="Show service projects")
    projects_p.add_argument("project_id", nargs="?", default=None)
    sub.add_parser("tick", help="Run one autonomous service tick")
    sub.add_parser("trace", help="Show recent cognitive trace")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(_run_interactive(args.name, args.persona, args.model, args.offline))
    elif args.command == "ask":
        text = " ".join(args.input)
        asyncio.run(_run_ask(text, args.name, args.persona, args.model, args.quiet, args.offline))
    elif args.command == "history":
        asyncio.run(_show_history())
    elif args.command == "search":
        query = " ".join(args.query)
        asyncio.run(_search_memory(query))
    elif args.command == "daemon":
        asyncio.run(_run_daemon_dry_run(args.events))
    elif args.command == "eval":
        if args.suite in LIVE_SUITES:
            asyncio.run(_run_live_eval(args))
        else:
            asyncio.run(_run_eval(args.suite))
    elif args.command == "service":
        if args.service_command == "init":
            _service_init()
        elif args.service_command == "start":
            _service_start()
        elif args.service_command == "status":
            asyncio.run(_service_status())
        elif args.service_command == "stop":
            asyncio.run(_client_control("stop"))
        else:
            service_p.print_help()
            sys.exit(1)
    elif args.command == "chat":
        asyncio.run(_client_chat(" ".join(args.message)))
    elif args.command == "influence":
        if args.influence_kind in {"goal", "constraint"}:
            asyncio.run(_client_influence(args.influence_kind, " ".join(args.content)))
        else:
            influence_p.print_help()
            sys.exit(1)
    elif args.command == "pause":
        asyncio.run(_client_control("pause"))
    elif args.command == "resume":
        asyncio.run(_client_control("resume"))
    elif args.command == "goals":
        asyncio.run(_client_goals())
    elif args.command == "influences":
        asyncio.run(_client_influences())
    elif args.command == "projects":
        asyncio.run(_client_projects(args.project_id))
    elif args.command == "tick":
        asyncio.run(_client_tick())
    elif args.command == "trace":
        asyncio.run(_client_trace())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
