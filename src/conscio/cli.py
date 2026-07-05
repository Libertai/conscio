from __future__ import annotations

import argparse
import asyncio
import json
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
from rich.table import Table
from rich.tree import Tree

from conscio import __version__
from conscio.config import DEFAULT_HOME, load_config, write_default_config
from conscio.core.agent import ConsciousAgent
from conscio.core.cognition import InputEvent
from conscio.core.runtime import CognitiveRuntime
from conscio.eval import LIVE_SUITES, run_eval_suite
from conscio.memory.lifecycle import (
    DatabaseCorruptError,
    create_home_backup,
    export_database,
    import_database,
    migrate,
    preflight_database,
    restore_home_backup,
    schema_status,
)
from conscio.memory.store import MemoryStore

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


def _attach_stream_printer(agent: ConsciousAgent) -> None:
    """Print tokens live during local episodes; the result panel that follows
    is the authoritative rendering."""

    def _on_stream_event(data: dict[str, Any]) -> None:
        event = data.get("event")
        if event == "token":
            console.print(data.get("text", ""), end="", soft_wrap=True, style="dim")
        elif event in ("discard", "final"):
            console.print()

    agent.runtime.executor.on_stream_event = _on_stream_event


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
    _attach_stream_printer(agent)
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
    if not quiet:
        _attach_stream_printer(agent)
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
    table.add_row("agent tokens", f"{result.meta.total_prompt_tokens + result.meta.total_completion_tokens:,}")
    if result.meta.total_judge_calls:
        table.add_row("judge tokens", f"{result.meta.judge_prompt_tokens + result.meta.judge_completion_tokens:,}")
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


async def _client_chat_stream(message: str) -> None:
    """Live-token variant of _client_chat; falls back to /message on 404
    (older service without the streaming endpoint)."""
    cfg = load_config()
    if not cfg.api_key:
        raise SystemExit("No API key configured. Run `conscio service init` first.")
    async with httpx.AsyncClient(base_url=cfg.base_url, timeout=None) as client:
        async with client.stream(
            "POST",
            "/message/stream",
            headers=_service_headers(cfg.api_key),
            json={"content": message},
        ) as response:
            if response.status_code == 404:
                await _client_chat(message)
                return
            response.raise_for_status()
            event_name = ""
            streamed = False
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event_name = line[len("event: "):].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[len("data: "):])
                if event_name == "chat.token":
                    streamed = True
                    console.print(payload.get("text", ""), end="", soft_wrap=True)
                elif event_name == "chat.discard":
                    console.print("\n[dim]… running tools …[/]")
                elif event_name == "message.result":
                    if streamed:
                        console.print()
                    console.print(Panel(Markdown(payload.get("output", "")), title="Conscio", border_style="green"))
                    return
                elif event_name == "message.error":
                    raise SystemExit(f"agent error ({payload.get('status')}): {payload.get('detail')}")


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
    console.print(
        Panel(
            Markdown(data.get("output", "")),
            title=f"Autonomy: {data.get('selected_action', '')}",
            border_style="green",
        )
    )


async def _client_trace() -> None:
    data = await _client_request("GET", "/trace")
    console.print(Panel(data.get("trace", ""), title="Cognitive Trace", border_style="dim"))


def _service_init(profile: str = "research") -> None:
    path = write_default_config(profile=profile)
    console.print(f"[green]Config ready:[/] {path}")
    if profile.replace("-", "_") == "autonomous_vm":
        console.print("[dim]Autonomous VM profile enabled: shell/code/web tools are part of the agent premises.[/]")
    else:
        console.print("[dim]Unsafe autonomy is disabled until config.toml sets unsafe_autonomy = true.[/]")


def _service_start() -> None:
    import uvicorn

    from conscio.api import create_app

    cfg = load_config()
    try:
        cfg.validate_public_bind()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        preflight_database(cfg)
    except DatabaseCorruptError as exc:
        console.print(f"[red]{exc}[/]")
        # Exit code 3 is reserved for corrupt state: RestartPreventExitStatus=3
        # in the systemd units stops the crash loop.
        raise SystemExit(3) from exc
    app = create_app(config=cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port)


def _service_doctor() -> None:
    cfg = load_config()
    checks: list[tuple[str, str, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, "ok" if ok else "fail", detail))

    try:
        cfg.validate_public_bind()
        add("public_bind", True, f"{cfg.host}:{cfg.port}")
    except ValueError as exc:
        add("public_bind", False, str(exc))

    try:
        cfg.ensure_layout()
        probe = cfg.home / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        add("home_writable", True, str(cfg.home))
    except OSError as exc:
        add("home_writable", False, str(exc))

    try:
        cfg.working_directory.mkdir(parents=True, exist_ok=True)
        probe = cfg.working_directory / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        add("workspace_writable", True, str(cfg.working_directory))
    except OSError as exc:
        add("workspace_writable", False, str(exc))

    status = schema_status(cfg.db_path)
    if status.exists:
        add("database_schema", status.ok, f"version={status.version} missing={status.missing_core or 'none'}")
    else:
        add("database_schema", True, "state.db does not exist yet; it will be created on start")

    try:
        preflight_database(cfg)
        add("database_integrity", True, "quick_check ok" if status.exists else "state.db does not exist yet")
    except DatabaseCorruptError as exc:
        add("database_integrity", False, str(exc))

    static_index = Path(__file__).resolve().parent / "static" / "index.html"
    add("web_assets", static_index.is_file(), str(static_index))
    add("model_backend", bool(cfg.llm_base_url), cfg.llm_base_url or "no live model configured")
    add("agent_profile", True, f"{cfg.agent.profile} premises={cfg.agent.premises or 'none'}")

    table = Table(title="Conscio Doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    failed = False
    for name, status_text, detail in checks:
        failed = failed or status_text == "fail"
        style = "green" if status_text == "ok" else "red"
        table.add_row(name, f"[{style}]{status_text}[/]", detail)
    console.print(table)
    if failed:
        raise SystemExit(1)


def _db_path(args: argparse.Namespace) -> Path:
    return Path(args.db).expanduser() if getattr(args, "db", "") else load_config().db_path


async def _db_migrate(args: argparse.Namespace) -> None:
    status = await migrate(_db_path(args))
    console.print(f"[green]Schema ready:[/] version={status.version} db={status.db_path}")


def _db_schema(args: argparse.Namespace) -> None:
    status = schema_status(_db_path(args))
    table = Table(title="Conscio DB Schema")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("db", str(status.db_path))
    table.add_row("exists", str(status.exists))
    table.add_row("version", str(status.version))
    table.add_row("ok", str(status.ok))
    table.add_row("missing_core", ", ".join(status.missing_core) or "none")
    table.add_row("tables", str(len(status.tables)))
    console.print(table)


def _db_backup(args: argparse.Namespace) -> None:
    cfg = load_config()
    archive = create_home_backup(cfg)
    console.print(f"[green]Backup written:[/] {archive}")


def _db_restore(args: argparse.Namespace) -> None:
    cfg = load_config()
    restore_home_backup(cfg, args.archive, force=args.force)
    console.print(f"[green]Backup restored:[/] {args.archive}")


def _db_export(args: argparse.Namespace) -> None:
    out = export_database(_db_path(args), args.out)
    console.print(f"[green]Export written:[/] {out}")


async def _db_import(args: argparse.Namespace) -> None:
    await import_database(args.input, _db_path(args), replace=args.replace)
    console.print(f"[green]Import complete:[/] {args.input}")


def _config_file_path() -> Path:
    return Path(os.environ.get("CONSCIO_CONFIG", DEFAULT_HOME / "config.toml")).expanduser()


def _toml_array(items: list[str]) -> str:
    return "[" + ", ".join(json.dumps(str(item)) for item in items) + "]"


def _upsert_section_array(text: str, section: str, key: str, values: list[str]) -> str:
    lines = text.splitlines()
    header = f"[{section}]"
    replacement = f"{key} = {_toml_array(values)}"
    start = None
    for index, line in enumerate(lines):
        if line.strip() == header:
            start = index
            break
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([header, replacement])
        return "\n".join(lines) + "\n"

    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = index
            break
    for index in range(start + 1, end):
        stripped = lines[index].strip()
        if stripped.startswith(f"{key} ") or stripped.startswith(f"{key}="):
            lines[index] = replacement
            return "\n".join(lines) + "\n"
    lines.insert(end, replacement)
    return "\n".join(lines) + "\n"


def _write_tool_policy(*, allowed: list[str] | None = None, denied: list[str] | None = None) -> Path:
    path = _config_file_path()
    if not path.exists():
        write_default_config(path)
    text = path.read_text(encoding="utf-8")
    if allowed is not None:
        text = _upsert_section_array(text, "tools", "allowed", sorted(dict.fromkeys(allowed)))
    if denied is not None:
        text = _upsert_section_array(text, "tools", "denied", sorted(dict.fromkeys(denied)))
    path.write_text(text, encoding="utf-8")
    return path


def _tools_list() -> None:
    cfg = load_config(_config_file_path())
    table = Table(title="Tool Policy")
    table.add_column("List", style="cyan")
    table.add_column("Tools")
    table.add_row("allowed", ", ".join(cfg.allowed_tools) or "(empty)")
    table.add_row("denied", ", ".join(cfg.denied_tools) or "(empty)")
    table.add_row("unsafe_autonomy", str(cfg.unsafe_autonomy))
    table.add_row("working_directory", str(cfg.working_directory))
    console.print(table)


def _tools_deny(args: argparse.Namespace) -> None:
    cfg = load_config(_config_file_path())
    names = [str(name) for name in args.names]
    name_set = set(names)
    denied = sorted(dict.fromkeys([*cfg.denied_tools, *names]))
    allowed = [name for name in cfg.allowed_tools if name not in name_set]
    path = _write_tool_policy(allowed=allowed, denied=denied)
    console.print(f"[green]Denied tools updated:[/] {', '.join(names)} ({path})")


def _tools_allow(args: argparse.Namespace) -> None:
    cfg = load_config(_config_file_path())
    names = [str(name) for name in args.names]
    name_set = set(names)
    allowed = sorted(dict.fromkeys([*cfg.allowed_tools, *names]))
    denied = [name for name in cfg.denied_tools if name not in name_set]
    path = _write_tool_policy(allowed=allowed, denied=denied)
    console.print(f"[green]Allowed tools updated:[/] {', '.join(names)} ({path})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="conscio — a conscious autonomous agent runtime",
    )
    parser.add_argument("--version", action="version", version=f"conscio {__version__}")
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
    service_init_p = service_sub.add_parser("init", help="Create ~/.conscio/config.toml")
    service_init_p.add_argument(
        "--profile",
        default="research",
        choices=["research", "autonomous-vm", "autonomous_vm"],
        help="Config profile to generate",
    )
    service_sub.add_parser("start", help="Start the authenticated FastAPI service")
    service_sub.add_parser("status", help="Show service status")
    service_sub.add_parser("doctor", help="Validate local service configuration and runtime prerequisites")
    service_sub.add_parser("stop", help="Stop the service")

    db_p = sub.add_parser("db", help="Manage the local Conscio state database")
    db_sub = db_p.add_subparsers(dest="db_command")
    db_schema_p = db_sub.add_parser("schema", help="Show schema status")
    db_schema_p.add_argument("--db", default="", help="Override database path")
    db_migrate_p = db_sub.add_parser("migrate", help="Create/update additive schema metadata")
    db_migrate_p.add_argument("--db", default="", help="Override database path")
    db_sub.add_parser("backup", help="Create a timestamped home backup archive")
    db_restore_p = db_sub.add_parser("restore", help="Restore a home backup archive")
    db_restore_p.add_argument("archive")
    db_restore_p.add_argument("--force", action="store_true", help="Restore even if the service lock exists")
    db_export_p = db_sub.add_parser("export", help="Export logical database rows to JSON")
    db_export_p.add_argument("--db", default="", help="Override database path")
    db_export_p.add_argument("--out", required=True, help="Output JSON path")
    db_import_p = db_sub.add_parser("import", help="Import logical database rows from JSON")
    db_import_p.add_argument("input")
    db_import_p.add_argument("--db", default="", help="Override database path")
    db_import_p.add_argument("--replace", action="store_true", help="Replace matching tables before import")

    tools_p = sub.add_parser("tools", help="Inspect or update tool policy")
    tools_sub = tools_p.add_subparsers(dest="tools_command")
    tools_sub.add_parser("list", help="Show configured tool allow/deny lists")
    tools_deny_p = tools_sub.add_parser("deny", help="Deny one or more tools in config.toml")
    tools_deny_p.add_argument("names", nargs="+")
    tools_allow_p = tools_sub.add_parser("allow", help="Allow one or more tools in config.toml")
    tools_allow_p.add_argument("names", nargs="+")

    chat_p = sub.add_parser("chat", help="Send a message to the running service")
    chat_p.add_argument("message", nargs="+", help="Message to send")
    chat_p.add_argument("--stream", action="store_true", help="Stream tokens live over /message/stream")
    influence_p = sub.add_parser("influence", help="Influence the running service")
    influence_sub = influence_p.add_subparsers(dest="influence_kind")
    influence_goal_p = influence_sub.add_parser("goal", help="Submit a goal influence")
    influence_goal_p.add_argument("content", nargs="+")
    influence_constraint_p = influence_sub.add_parser("constraint", help="Submit a constraint influence")
    influence_constraint_p.add_argument("content", nargs="+")
    sub.add_parser("pause", help="Pause autonomous action")
    sub.add_parser("resume", help="Resume autonomous action")
    sub.add_parser("cancel", help="Cancel the episode the service is currently running")
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
            _service_init(args.profile)
        elif args.service_command == "start":
            _service_start()
        elif args.service_command == "status":
            asyncio.run(_service_status())
        elif args.service_command == "doctor":
            _service_doctor()
        elif args.service_command == "stop":
            asyncio.run(_client_control("stop"))
        else:
            service_p.print_help()
            sys.exit(1)
    elif args.command == "db":
        if args.db_command == "schema":
            _db_schema(args)
        elif args.db_command == "migrate":
            asyncio.run(_db_migrate(args))
        elif args.db_command == "backup":
            _db_backup(args)
        elif args.db_command == "restore":
            _db_restore(args)
        elif args.db_command == "export":
            _db_export(args)
        elif args.db_command == "import":
            asyncio.run(_db_import(args))
        else:
            db_p.print_help()
            sys.exit(1)
    elif args.command == "tools":
        if args.tools_command == "list":
            _tools_list()
        elif args.tools_command == "deny":
            _tools_deny(args)
        elif args.tools_command == "allow":
            _tools_allow(args)
        else:
            tools_p.print_help()
            sys.exit(1)
    elif args.command == "chat":
        if args.stream:
            asyncio.run(_client_chat_stream(" ".join(args.message)))
        else:
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
    elif args.command == "cancel":
        asyncio.run(_client_control("cancel"))
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
