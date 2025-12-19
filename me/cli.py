"""
CLI interface for the self-model agent.

Provides command-line access to run the agent, manage MCPs,
view memory, and control agent execution.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click

from me.agent.core import Agent, AgentConfig
from me.agent.memory import Memory
from me.agent.mcp import MCPRegistry, COMMON_MCPS


def run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@click.group()
@click.version_option(version="0.1.0", prog_name="me")
def main():
    """
    me - Self-model agent CLI

    Upload "me" into a computer through a self-aware agent system.
    """
    pass


@main.command()
@click.argument("prompt", nargs=-1, required=False)
@click.option("--name", "-n", default="me", help="Agent name")
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--max-turns", "-t", type=int, default=None, help="Maximum conversation turns")
@click.option("--refresh-rate", "-r", type=int, default=100, help="Refresh rate in ms for process polling")
@click.option("--parent", "-p", default=None, help="Parent agent ID (for child agents)")
@click.option("--cwd", "-d", type=click.Path(exists=True, path_type=Path), default=None, help="Working directory")
@click.option("--interactive", "-i", is_flag=True, help="Run in interactive mode")
@click.option("--tui/--no-tui", default=True, help="Use TUI for interactive mode (default: enabled)")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose tool call output")
def run(
    prompt: tuple[str, ...],
    name: str,
    model: str | None,
    max_turns: int | None,
    refresh_rate: int,
    parent: str | None,
    cwd: Path | None,
    interactive: bool,
    tui: bool,
    verbose: bool,
):
    """Run the agent with a prompt."""
    prompt_text = " ".join(prompt) if prompt else None

    if not prompt_text and not interactive:
        click.echo("Error: Either provide a prompt or use --interactive mode")
        sys.exit(1)

    config = AgentConfig(
        name=name,
        model=model,
        max_turns=max_turns,
        refresh_rate_ms=refresh_rate,
        parent_id=parent,
        cwd=cwd or Path.cwd(),
    )

    agent = Agent(config)

    if interactive:
        if tui:
            # Use Textual TUI
            from me.tui import run_tui
            run_async(run_tui(agent, prompt_text))
        else:
            # Use simple CLI
            run_async(_interactive_session(agent, prompt_text, verbose=verbose))
    else:
        run_async(_single_run(agent, prompt_text, verbose=verbose))


def _format_tool_call(block, verbose: bool = False) -> str:
    """Format a tool call for display."""
    if not verbose:
        return f"[Tool: {block.name}]"

    # Verbose output with parameters
    parts = [f"[Tool: {block.name}"]
    if block.input:
        params = []
        for key, value in list(block.input.items())[:5]:
            if isinstance(value, str):
                if len(value) > 60:
                    value = value[:60] + "..."
                value = value.replace("\n", "\\n")
            params.append(f"{key}={value}")
        if params:
            parts.append(f"({', '.join(params)})")
    parts.append("]")
    return "".join(parts)


async def _single_run(agent: Agent, prompt: str, verbose: bool = False):
    """Run a single agent interaction."""
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ResultMessage

    click.echo(f"[{agent.config.agent_id}] Starting agent run...")
    click.echo()

    async for message in agent.run(prompt):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    click.echo(block.text)
                elif isinstance(block, ToolUseBlock):
                    click.echo(f"\n{_format_tool_call(block, verbose)}", nl=False)

        elif isinstance(message, ResultMessage):
            click.echo()
            click.echo(f"[{agent.config.agent_id}] Run completed in {message.duration_ms}ms")
            if message.total_cost_usd:
                click.echo(f"Cost: ${message.total_cost_usd:.4f}")


async def _interactive_session(agent: Agent, initial_prompt: str | None, verbose: bool = False):
    """Run an interactive agent session (simple CLI mode)."""
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ResultMessage

    click.echo(f"[{agent.config.agent_id}] Interactive session started")
    click.echo("Type 'exit' or 'quit' to end the session")
    click.echo("Use --tui for better editing experience")
    click.echo()

    first_run = True

    while True:
        if first_run and initial_prompt:
            prompt = initial_prompt
            first_run = False
        else:
            try:
                prompt = click.prompt("you", type=str)
            except click.Abort:
                break

        if prompt.lower() in ("exit", "quit", "/exit", "/quit"):
            break

        if prompt.startswith("/"):
            # Handle special commands
            if prompt == "/body":
                import json
                click.echo(json.dumps(agent.body.to_dict(), indent=2))
            elif prompt == "/memory":
                stats = agent.memory.stats()
                click.echo(f"Memories: {stats['total_memories']}, Runs: {stats['total_runs']}")
            elif prompt == "/mcps":
                mcps = agent.mcp_registry.list()
                for mcp in mcps:
                    status = "enabled" if mcp["enabled"] else "disabled"
                    click.echo(f"  {mcp['name']}: {mcp['command']} ({status})")
            elif prompt == "/help":
                click.echo("Commands:")
                click.echo("  /body   - Show body metadata")
                click.echo("  /memory - Show memory stats")
                click.echo("  /mcps   - List MCP servers")
                click.echo("  /help   - Show this help")
                click.echo("  /exit   - Exit session")
            else:
                click.echo(f"Unknown command: {prompt}")
            continue

        # Run the agent
        click.echo()
        async for message in agent.run(prompt):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        click.echo(block.text, nl=False)
                    elif isinstance(block, ToolUseBlock):
                        click.echo(f"\n{_format_tool_call(block, verbose)}", nl=False)
            elif isinstance(message, ResultMessage):
                click.echo()

        click.echo()

    click.echo(f"\n[{agent.config.agent_id}] Session ended")


@main.group()
def mcp():
    """Manage MCP servers."""
    pass


@mcp.command("list")
def mcp_list():
    """List registered MCP servers."""
    registry = MCPRegistry()
    mcps = registry.list()

    if not mcps:
        click.echo("No MCP servers registered")
        return

    for mcp in mcps:
        status = click.style("enabled", fg="green") if mcp["enabled"] else click.style("disabled", fg="red")
        click.echo(f"  {mcp['name']}: {mcp['command']} [{status}]")
        if mcp["args"]:
            click.echo(f"    args: {' '.join(mcp['args'])}")


@mcp.command("add")
@click.argument("name")
@click.argument("command")
@click.argument("args", nargs=-1)
def mcp_add(name: str, command: str, args: tuple[str, ...]):
    """Register a new MCP server."""
    registry = MCPRegistry()
    run_async(registry.register(name, command, list(args)))
    click.echo(f"Registered MCP: {name}")


@mcp.command("remove")
@click.argument("name")
def mcp_remove(name: str):
    """Unregister an MCP server."""
    registry = MCPRegistry()
    if run_async(registry.unregister(name)):
        click.echo(f"Removed MCP: {name}")
    else:
        click.echo(f"MCP not found: {name}")


@mcp.command("enable")
@click.argument("name")
def mcp_enable(name: str):
    """Enable an MCP server."""
    registry = MCPRegistry()
    if run_async(registry.enable(name)):
        click.echo(f"Enabled MCP: {name}")
    else:
        click.echo(f"MCP not found: {name}")


@mcp.command("disable")
@click.argument("name")
def mcp_disable(name: str):
    """Disable an MCP server."""
    registry = MCPRegistry()
    if run_async(registry.disable(name)):
        click.echo(f"Disabled MCP: {name}")
    else:
        click.echo(f"MCP not found: {name}")


@mcp.command("common")
def mcp_common():
    """Show commonly available MCP servers."""
    click.echo("Common MCP servers you can register:")
    for name, info in COMMON_MCPS.items():
        click.echo(f"\n  {name}")
        click.echo(f"    {info['description']}")
        click.echo(f"    Command: {info['command']} {' '.join(info['args'])}")


@main.group()
def memory():
    """Manage agent memory."""
    pass


@memory.command("stats")
def memory_stats():
    """Show memory statistics."""
    mem = Memory(Path.home() / ".me" / "memory")
    stats = mem.stats()
    click.echo(f"Total memories: {stats['total_memories']}")
    click.echo(f"Total runs: {stats['total_runs']}")
    click.echo(f"Storage: {stats['storage_dir']}")


@memory.command("runs")
@click.option("--limit", "-l", type=int, default=10, help="Number of runs to show")
def memory_runs(limit: int):
    """List recent agent runs."""
    mem = Memory(Path.home() / ".me" / "memory")
    runs = run_async(mem.list_runs(limit=limit))

    if not runs:
        click.echo("No runs found")
        return

    for run in runs:
        click.echo(f"\n  {run['run_id']}")
        click.echo(f"    Agent: {run['agent_id']}")
        click.echo(f"    Started: {run['started_at']}")
        click.echo(f"    Memories: {run['memory_count']}")


@memory.command("search")
@click.argument("query")
@click.option("--limit", "-l", type=int, default=5, help="Number of results")
def memory_search(query: str, limit: int):
    """Search memories semantically."""
    mem = Memory(Path.home() / ".me" / "memory")
    results = run_async(mem.recall(query, limit=limit))

    if not results:
        click.echo("No memories found")
        return

    for m in results:
        click.echo(f"\n[{m['id']}] (score: {m['score']:.2f})")
        click.echo(f"  {m['content'][:200]}{'...' if len(m['content']) > 200 else ''}")
        click.echo(f"  Tags: {', '.join(m['tags'])}")


@memory.command("clear")
@click.option("--run-id", "-r", default=None, help="Clear specific run only")
@click.confirmation_option(prompt="Are you sure you want to clear memory?")
def memory_clear(run_id: str | None):
    """Clear agent memory."""
    mem = Memory(Path.home() / ".me" / "memory")

    if run_id:
        if run_async(mem.delete_run(run_id)):
            click.echo(f"Cleared run: {run_id}")
        else:
            click.echo(f"Run not found: {run_id}")
    else:
        # Clear all - remove the directory
        import shutil
        storage_dir = Path.home() / ".me" / "memory"
        if storage_dir.exists():
            shutil.rmtree(storage_dir)
        click.echo("Memory cleared")


@main.group()
def daemon():
    """Manage agent daemons (Agentic OS)."""
    pass


@daemon.command("list")
@click.option("--agent", "-a", default=None, help="Agent ID (uses default if not specified)")
def daemon_list(agent: str | None):
    """List all daemons and their status."""
    from me.agent.body import BodyDirectory
    from me.agent.unconscious import UnconsciousDirectory, UnconsciousRunner, install_default_pipelines

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found. Run 'me run <prompt>' first.")
        return

    body_root = base_dir / "agents" / agent_id
    body_dir = BodyDirectory(base_dir, agent_id)
    udir = UnconsciousDirectory(body_root)

    # Ensure default pipelines exist
    if not list(udir.list_pipelines()):
        install_default_pipelines(udir)

    runner = UnconsciousRunner(udir, body_root, Path.cwd())
    runner.load_pipelines()

    daemons = runner.list_daemons()
    if not daemons:
        click.echo("No daemons registered")
        return

    click.echo(f"Runlevel: {runner._runlevel.value}")
    click.echo()

    for d in daemons:
        state = d["state"]
        if state == "running":
            state_str = click.style(state, fg="green")
        elif state == "failed":
            state_str = click.style(state, fg="red")
        elif state == "disabled":
            state_str = click.style(state, fg="yellow")
        else:
            state_str = state

        click.echo(f"  {d['name']:<25} {state_str:<15} [{d['group']}] runlevel={d['runlevel']}")


@daemon.command("start")
@click.argument("name")
@click.option("--agent", "-a", default=None, help="Agent ID")
def daemon_start(name: str, agent: str | None):
    """Start a daemon."""
    from me.agent.unconscious import UnconsciousDirectory, UnconsciousRunner, install_default_pipelines

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body_root = base_dir / "agents" / agent_id
    udir = UnconsciousDirectory(body_root)

    if not list(udir.list_pipelines()):
        install_default_pipelines(udir)

    runner = UnconsciousRunner(udir, body_root, Path.cwd())
    runner.load_pipelines()

    success = run_async(runner.start_daemon(name))
    if success:
        click.echo(f"Started daemon: {name}")
    else:
        click.echo(f"Failed to start daemon: {name}")


@daemon.command("stop")
@click.argument("name")
@click.option("--agent", "-a", default=None, help="Agent ID")
def daemon_stop(name: str, agent: str | None):
    """Stop a daemon."""
    from me.agent.unconscious import UnconsciousDirectory, UnconsciousRunner, install_default_pipelines

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body_root = base_dir / "agents" / agent_id
    udir = UnconsciousDirectory(body_root)

    if not list(udir.list_pipelines()):
        install_default_pipelines(udir)

    runner = UnconsciousRunner(udir, body_root, Path.cwd())
    runner.load_pipelines()

    success = run_async(runner.stop_daemon(name))
    if success:
        click.echo(f"Stopped daemon: {name}")
    else:
        click.echo(f"Failed to stop daemon: {name}")


@daemon.command("enable")
@click.argument("name")
@click.option("--agent", "-a", default=None, help="Agent ID")
def daemon_enable(name: str, agent: str | None):
    """Enable a daemon for boot."""
    from me.agent.unconscious import UnconsciousDirectory, UnconsciousRunner, install_default_pipelines

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body_root = base_dir / "agents" / agent_id
    udir = UnconsciousDirectory(body_root)

    if not list(udir.list_pipelines()):
        install_default_pipelines(udir)

    runner = UnconsciousRunner(udir, body_root, Path.cwd())
    runner.load_pipelines()

    success = runner.enable_daemon(name)
    if success:
        click.echo(f"Enabled daemon: {name}")
    else:
        click.echo(f"Failed to enable daemon: {name}")


@daemon.command("disable")
@click.argument("name")
@click.option("--agent", "-a", default=None, help="Agent ID")
def daemon_disable(name: str, agent: str | None):
    """Disable a daemon from boot."""
    from me.agent.unconscious import UnconsciousDirectory, UnconsciousRunner, install_default_pipelines

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body_root = base_dir / "agents" / agent_id
    udir = UnconsciousDirectory(body_root)

    if not list(udir.list_pipelines()):
        install_default_pipelines(udir)

    runner = UnconsciousRunner(udir, body_root, Path.cwd())
    runner.load_pipelines()

    success = runner.disable_daemon(name)
    if success:
        click.echo(f"Disabled daemon: {name}")
    else:
        click.echo(f"Failed to disable daemon: {name}")


@daemon.command("journal")
@click.option("--agent", "-a", default=None, help="Agent ID")
@click.option("--daemon-name", "-d", default=None, help="Filter by daemon")
@click.option("--level", "-l", default=None, help="Minimum log level")
@click.option("--limit", "-n", type=int, default=20, help="Number of entries")
def daemon_journal(agent: str | None, daemon_name: str | None, level: str | None, limit: int):
    """View daemon journal entries."""
    from me.agent.unconscious import UnconsciousDirectory, LogLevel

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body_root = base_dir / "agents" / agent_id
    udir = UnconsciousDirectory(body_root)

    min_level = LogLevel(level) if level else None
    entries = udir.query_journal(daemon=daemon_name, min_level=min_level, limit=limit)

    if not entries:
        click.echo("No journal entries")
        return

    for entry in entries:
        level_str = entry.level.value.upper()
        if entry.level == LogLevel.ERROR:
            level_str = click.style(level_str, fg="red")
        elif entry.level == LogLevel.WARN:
            level_str = click.style(level_str, fg="yellow")

        ts = entry.timestamp.strftime("%H:%M:%S")
        click.echo(f"{ts} [{level_str}] {entry.daemon}: {entry.message}")


@daemon.command("runlevel")
@click.argument("level", required=False)
@click.option("--agent", "-a", default=None, help="Agent ID")
def daemon_runlevel(level: str | None, agent: str | None):
    """Show or change the runlevel."""
    from me.agent.unconscious import UnconsciousDirectory, UnconsciousRunner, Runlevel, install_default_pipelines

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body_root = base_dir / "agents" / agent_id
    udir = UnconsciousDirectory(body_root)

    if not list(udir.list_pipelines()):
        install_default_pipelines(udir)

    runner = UnconsciousRunner(udir, body_root, Path.cwd())
    runner.load_pipelines()

    if level is None:
        click.echo(f"Current runlevel: {runner._runlevel.value}")
        click.echo("\nAvailable runlevels:")
        for rl in Runlevel:
            click.echo(f"  {rl.value}")
    else:
        try:
            new_level = Runlevel(level)
            run_async(runner.change_runlevel(new_level))
            click.echo(f"Changed runlevel to: {new_level.value}")
        except ValueError:
            click.echo(f"Invalid runlevel: {level}")
            click.echo(f"Valid values: {', '.join(r.value for r in Runlevel)}")


@daemon.command("boot")
@click.option("--agent", "-a", default=None, help="Agent ID")
@click.option("--level", "-l", default="normal", help="Target runlevel")
def daemon_boot(agent: str | None, level: str):
    """Boot the daemon system."""
    from me.agent.unconscious import UnconsciousDirectory, UnconsciousRunner, Runlevel, install_default_pipelines

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body_root = base_dir / "agents" / agent_id
    udir = UnconsciousDirectory(body_root)

    if not list(udir.list_pipelines()):
        install_default_pipelines(udir)

    runner = UnconsciousRunner(udir, body_root, Path.cwd())
    runner.load_pipelines()

    try:
        target = Runlevel(level)
        run_async(runner.boot(target))
        click.echo(f"Booted to runlevel: {target.value}")
    except ValueError:
        click.echo(f"Invalid runlevel: {level}")


@daemon.command("shutdown")
@click.option("--agent", "-a", default=None, help="Agent ID")
def daemon_shutdown(agent: str | None):
    """Shutdown the daemon system."""
    from me.agent.unconscious import UnconsciousDirectory, UnconsciousRunner, install_default_pipelines

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body_root = base_dir / "agents" / agent_id
    udir = UnconsciousDirectory(body_root)

    if not list(udir.list_pipelines()):
        install_default_pipelines(udir)

    runner = UnconsciousRunner(udir, body_root, Path.cwd())
    runner.load_pipelines()

    run_async(runner.shutdown())
    click.echo("Daemon system shutdown complete")


# =============================================================================
# Skills CLI
# =============================================================================

@main.group()
def skills():
    """Manage agent skills (learned capabilities)."""
    pass


@skills.command("list")
@click.option("--agent", "-a", default=None, help="Agent ID")
@click.option("--state", "-s", type=click.Choice(["learned", "developing", "atrophied"]), help="Filter by state")
def skills_list(agent: str | None, state: str | None):
    """List agent skills."""
    from me.agent.body import BodyDirectory
    from me.agent.skills import SkillState

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found. Run 'me run <prompt>' first.")
        return

    body = BodyDirectory(base_dir, agent_id)
    body.initialize()
    manager = body.skills

    filter_state = SkillState(state) if state else None
    skill_list = manager.list_skills(filter_state)

    if not skill_list:
        click.echo("No skills acquired yet.")
        click.echo("Use 'me skills create <name>' or the codify_skill tool to create skills.")
        return

    click.echo(f"\nSkills for agent {agent_id}:")
    click.echo("-" * 50)

    current_state = None
    for skill in sorted(skill_list, key=lambda s: (s.state.value, -s.proficiency)):
        if skill.state != current_state:
            current_state = skill.state
            state_color = {"learned": "green", "developing": "yellow", "atrophied": "red"}.get(current_state.value, "white")
            click.echo(f"\n{click.style(current_state.value.upper(), fg=state_color)}:")

        active_indicator = click.style(" (ACTIVE)", fg="cyan") if skill.is_active else ""
        prof_bar = "█" * int(skill.proficiency * 10) + "░" * (10 - int(skill.proficiency * 10))
        click.echo(f"  {skill.name}: [{prof_bar}] {skill.proficiency:.0%}{active_indicator}")
        if skill.description:
            click.echo(f"    {skill.description[:60]}")


@skills.command("show")
@click.argument("name")
@click.option("--agent", "-a", default=None, help="Agent ID")
def skills_show(name: str, agent: str | None):
    """Show skill details."""
    from me.agent.body import BodyDirectory

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body = BodyDirectory(base_dir, agent_id)
    body.initialize()
    manager = body.skills

    skill = manager.get_skill(name)
    if not skill:
        click.echo(f"Skill not found: {name}")
        return

    meta = skill.metadata
    click.echo(f"\n{'='*60}")
    click.echo(f"Skill: {meta.name}")
    click.echo(f"{'='*60}")
    click.echo(f"Description: {meta.description}")
    click.echo(f"Version: {meta.version}")
    click.echo(f"State: {meta.state.value}")
    click.echo(f"Proficiency: {meta.proficiency:.0%}")
    click.echo(f"Uses: {meta.use_count} ({meta.success_count} successes, {meta.failure_count} failures)")
    click.echo(f"Active: {meta.is_active}")
    click.echo(f"Tags: {', '.join(meta.tags) if meta.tags else 'none'}")
    click.echo(f"Domains: {', '.join(meta.domains) if meta.domains else 'none'}")
    click.echo(f"Learned from: {meta.learned_from or 'unknown'}")
    click.echo(f"Created: {meta.created_at.isoformat() if meta.created_at else 'unknown'}")
    click.echo(f"Last used: {meta.last_used.isoformat() if meta.last_used else 'never'}")

    if skill.scripts:
        click.echo(f"\nScripts: {', '.join(skill.scripts)}")

    if skill.steps:
        click.echo(f"\nSteps:")
        for i, step in enumerate(skill.steps, 1):
            click.echo(f"  {i}. {step}")

    if skill.metadata.daemons:
        click.echo(f"\nDaemons:")
        for d in skill.metadata.daemons:
            click.echo(f"  - {d.get('name', 'unnamed')}: {d.get('trigger', 'on_change')}")

    click.echo(f"\n--- Instructions ---")
    click.echo(skill.instructions[:2000])
    if len(skill.instructions) > 2000:
        click.echo("... (truncated)")


@skills.command("activate")
@click.argument("name")
@click.option("--agent", "-a", default=None, help="Agent ID")
def skills_activate(name: str, agent: str | None):
    """Activate a skill."""
    from me.agent.body import BodyDirectory

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body = BodyDirectory(base_dir, agent_id)
    body.initialize()
    manager = body.skills

    skill = manager.activate_skill(name)
    if skill:
        click.echo(f"Activated skill: {name}")
        click.echo(f"Proficiency: {skill.metadata.proficiency:.0%}")
    else:
        click.echo(f"Skill not found: {name}")


@skills.command("deactivate")
@click.argument("name")
@click.option("--agent", "-a", default=None, help="Agent ID")
def skills_deactivate(name: str, agent: str | None):
    """Deactivate a skill."""
    from me.agent.body import BodyDirectory

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body = BodyDirectory(base_dir, agent_id)
    body.initialize()
    manager = body.skills

    if manager.deactivate_skill(name):
        click.echo(f"Deactivated skill: {name}")
    else:
        click.echo(f"Skill not found or not active: {name}")


@skills.command("create")
@click.argument("name")
@click.option("--description", "-d", prompt="Description", help="Skill description")
@click.option("--tags", "-t", default="", help="Comma-separated tags")
@click.option("--agent", "-a", default=None, help="Agent ID")
def skills_create(name: str, description: str, tags: str, agent: str | None):
    """Create a new skill interactively."""
    from me.agent.body import BodyDirectory

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body = BodyDirectory(base_dir, agent_id)
    body.initialize()
    manager = body.skills

    click.echo("\nEnter skill instructions (markdown). End with Ctrl+D or empty line:")
    lines = []
    try:
        while True:
            line = input()
            if not line and lines:
                break
            lines.append(line)
    except EOFError:
        pass

    instructions = "\n".join(lines)
    if not instructions.strip():
        instructions = f"## {name}\n\nInstructions for {name} skill."

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    skill = manager.codify_skill(
        name=name,
        description=description,
        instructions=instructions,
        tags=tag_list,
    )

    click.echo(f"\nCreated skill: {skill.metadata.name}")
    click.echo(f"State: developing")
    click.echo(f"Path: {manager.skills_dir / 'developing' / name}")


@skills.command("delete")
@click.argument("name")
@click.option("--agent", "-a", default=None, help="Agent ID")
@click.confirmation_option(prompt="Are you sure you want to delete this skill?")
def skills_delete(name: str, agent: str | None):
    """Delete a skill."""
    from me.agent.body import BodyDirectory

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body = BodyDirectory(base_dir, agent_id)
    body.initialize()
    manager = body.skills

    if manager.delete_skill(name):
        click.echo(f"Deleted skill: {name}")
    else:
        click.echo(f"Skill not found: {name}")


@skills.command("install")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--agent", "-a", default=None, help="Agent ID")
def skills_install(path: Path, agent: str | None):
    """Install a skill from an external directory."""
    from me.agent.body import BodyDirectory

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body = BodyDirectory(base_dir, agent_id)
    body.initialize()
    manager = body.skills

    skill = manager.install_skill(path)
    if skill:
        click.echo(f"Installed skill: {skill.metadata.name}")
        click.echo(f"State: developing")
    else:
        click.echo(f"Failed to install skill. Ensure {path}/SKILL.md exists.")


@skills.command("install-templates")
@click.option("--agent", "-a", default=None, help="Agent ID")
def skills_install_templates(agent: str | None):
    """Install default skill templates."""
    from me.agent.body import BodyDirectory
    from me.agent.skill_templates import install_all_templates

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body = BodyDirectory(base_dir, agent_id)
    body.initialize()
    manager = body.skills

    installed = install_all_templates(manager)
    if installed:
        click.echo(f"Installed {len(installed)} skill templates:")
        for name in installed:
            click.echo(f"  - {name}")
    else:
        click.echo("No templates to install or all already installed.")


@skills.command("stats")
@click.option("--agent", "-a", default=None, help="Agent ID")
def skills_stats(agent: str | None):
    """Show skill statistics."""
    from me.agent.body import BodyDirectory

    base_dir = Path.home() / ".me"
    agent_id = agent or _get_default_agent(base_dir)
    if not agent_id:
        click.echo("No agent found.")
        return

    body = BodyDirectory(base_dir, agent_id)
    body.initialize()
    manager = body.skills

    stats = manager.get_statistics()
    click.echo(f"\nSkill Statistics for {agent_id}:")
    click.echo("-" * 40)
    click.echo(f"Total skills: {stats['total_skills']}")
    click.echo(f"  Learned: {stats['learned']}")
    click.echo(f"  Developing: {stats['developing']}")
    click.echo(f"  Atrophied: {stats['atrophied']}")
    click.echo(f"Currently active: {stats['active_now']}")
    click.echo(f"Total uses: {stats['total_uses']}")
    click.echo(f"Average proficiency: {stats['avg_proficiency']:.0%}")

    if stats['domains']:
        click.echo(f"\nDomains: {', '.join(stats['domains'])}")

    if stats['by_type']:
        click.echo("\nBy type:")
        for skill_type, count in stats['by_type'].items():
            if count > 0:
                click.echo(f"  {skill_type}: {count}")


def _get_default_agent(base_dir: Path) -> str | None:
    """Get the most recently used agent ID."""
    agents_dir = base_dir / "agents"
    if not agents_dir.exists():
        return None

    agents = [(d, d.stat().st_mtime) for d in agents_dir.iterdir() if d.is_dir()]
    if not agents:
        return None

    # Return most recently modified
    agents.sort(key=lambda x: x[1], reverse=True)
    return agents[0][0].name


@main.command()
def body():
    """Show current body/system metadata."""
    import json
    from me.agent.body import BodyDirectory
    from pathlib import Path

    base_dir = Path.home() / ".me"
    # List existing agents or show system info
    agents_dir = base_dir / "agents"
    if agents_dir.exists():
        agents = [d.name for d in agents_dir.iterdir() if d.is_dir()]
        if agents:
            click.echo(f"Agents: {len(agents)}")
            for agent_id in agents[:10]:
                body_dir = BodyDirectory(base_dir, agent_id)
                identity = body_dir.identity
                if identity:
                    click.echo(f"  {agent_id}: {identity.name} (gen {identity.generation})")
                    click.echo(f"    Steps: {body_dir.get_step_count()}")
            return

    # No agents yet, show system info
    from me.agent.body import SystemInfo
    sys_info = SystemInfo.collect()
    click.echo(f"System: {sys_info.hostname}")
    click.echo(f"Platform: {sys_info.platform} {sys_info.platform_version}")
    click.echo(f"No agents created yet. Run 'me run <prompt>' to create one.")


if __name__ == "__main__":
    main()
