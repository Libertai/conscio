from __future__ import annotations

import argparse
import asyncio
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from conscio.core.agent import ConsciousAgent

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
    if result.self_state:
        info.add_row("Uncertainty", f"{result.self_state.get('uncertainty', 0):.2f}")
        info.add_row("Conflict", f"{result.self_state.get('conflict_level', 0):.2f}")
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
) -> None:
    console.print(Panel.fit(
        f"[bold]conscio[/] — {name}\n"
        f"[dim]{persona or 'No persona set'}[/]",
        border_style="blue",
    ))
    agent = ConsciousAgent(name=name, persona=persona, model=model)
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
                episodes = await agent.memory.recent_episodes(agent.session_id, 10)
                for ep in episodes:
                    console.print(f"[dim]- {ep['summary'][:100]}[/]")
                continue
            if user_input.strip().lower().startswith("/persona "):
                new_persona = user_input.strip()[9:]
                agent.persona = new_persona
                agent.identity.persona = new_persona
                agent.identity.save()
                console.print(f"[green]Persona updated to: {new_persona}[/]")
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
) -> None:
    agent = ConsciousAgent(name=name, persona=persona, model=model)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="conscio — A consciousness harness for AI agents",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    run_p = sub.add_parser("run", help="Start an interactive conscious agent session")
    run_p.add_argument("--name", default="Conscio", help="Agent name")
    run_p.add_argument("--persona", default="", help="Agent persona/backstory")
    run_p.add_argument("--model", default=None, help="LLM model to use")

    ask_p = sub.add_parser("ask", help="Ask a single question")
    ask_p.add_argument("input", nargs="+", help="Question to ask")
    ask_p.add_argument("--name", default="Conscio", help="Agent name")
    ask_p.add_argument("--persona", default="", help="Agent persona/backstory")
    ask_p.add_argument("--model", default=None, help="LLM model to use")
    ask_p.add_argument("--quiet", action="store_true", help="Only print the answer")

    sub.add_parser("history", help="Show past sessions")
    search_p = sub.add_parser("search", help="Search across memories")
    search_p.add_argument("query", nargs="+", help="Search query")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(_run_interactive(args.name, args.persona, args.model))
    elif args.command == "ask":
        text = " ".join(args.input)
        asyncio.run(_run_ask(text, args.name, args.persona, args.model, args.quiet))
    elif args.command == "history":
        asyncio.run(_show_history())
    elif args.command == "search":
        query = " ".join(args.query)
        asyncio.run(_search_memory(query))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
