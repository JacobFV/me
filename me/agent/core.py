"""
Core agent module - the heart of the self-model system.

Simplified philosophy: The agent's body IS a directory. Everything that can be
a file operation IS a file operation. The agent uses Claude's built-in Read/
Write/Edit tools on its body directory.

Essential tools (things that genuinely can't be files):
- Process: run_command, send_input, read_output, spawn_child_agent
- Semantic memory: store_memory, recall_memories (vector search)
- MCP: register_mcp, unregister_mcp

Everything else: Edit the body files directly.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    Message,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    tool,
    create_sdk_mcp_server,
)

from me.agent.body import BodyDirectory
from me.agent.memory import Memory
from me.agent.mcp import MCPRegistry
from me.agent.process import ProcessReader, ProcessStatus
from me.agent.unconscious import (
    UnconsciousDirectory,
    UnconsciousRunner,
    install_default_pipelines,
)


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str | None = None
    name: str = "me"
    cwd: Path = field(default_factory=Path.cwd)
    refresh_rate_ms: int = 100
    max_turns: int | None = None
    base_dir: Path = field(default_factory=lambda: Path.home() / ".me")
    model: str | None = None
    permission_mode: str = "acceptEdits"
    system_prompt_append: str = ""

    def __post_init__(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)


class Agent:
    """
    Self-model agent with body-as-filesystem.

    The agent's body IS a directory:
        ~/.me/agents/<id>/
        ├── identity.json       # IMMUTABLE
        ├── config.json         # Edit to change settings
        ├── embodiment.json     # Edit to change location, mood
        ├── sensors.json        # Edit to add/remove sensors
        ├── mouth.json          # Edit to change output destination
        ├── working_set.json    # Edit to add goals, track loops
        ├── character.json      # Read to see revealed preferences
        └── memory/
            ├── episodes/       # Write .md files to record episodes
            ├── procedures/     # Write .md files to codify procedures
            ├── significant/    # Write .md files for significant moments
            ├── intentions/     # Write .md files for future reminders
            └── theories/       # Write .md files for working theories

    Essential tools (can't be file operations):
    - run_command, send_input, read_output, spawn_child_agent (OS processes)
    - store_memory, recall_memories (vector search)
    - register_mcp, unregister_mcp (dynamic capability extension)
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

        # Body as directory
        self.body = BodyDirectory(self.config.base_dir, self.config.agent_id)
        self.body.initialize(
            name=self.config.name,
            parent_id=self.config.parent_id,
            generation=1 if self.config.parent_id else 0,
        )
        self.body.set_start_time(datetime.now())

        # Semantic memory (vector search - can't be a file operation)
        self.memory = Memory(self.config.base_dir / "memory")

        # Process control (OS processes - can't be a file operation)
        self.process_reader = ProcessReader(self.config.refresh_rate_ms)

        # MCP registry (dynamic capability - can't be a file operation)
        self.mcp_registry = MCPRegistry()

        # Unconscious - background perception/abstraction pipelines
        self.unconscious_dir = UnconsciousDirectory(self.body.root)
        install_default_pipelines(self.unconscious_dir)
        self.unconscious = UnconsciousRunner(
            unconscious_dir=self.unconscious_dir,
            body_root=self.body.root,
            cwd=self.config.cwd,
        )
        self.unconscious.load_pipelines()

        self._client: ClaudeSDKClient | None = None
        self._run_id: str = str(uuid.uuid4())
        self._start_time: datetime | None = None
        self._step_event = asyncio.Event()
        self._stop_event = asyncio.Event()

    @property
    def system_prompt(self) -> str:
        """Build the system prompt."""
        body_info = self.body.to_prompt_section()

        return f"""You are a self-model agent - an instance of "me" running in a computer.

## Your Body

Your body IS a directory at `{self.body.root}`. Everything about you is a file:

```
{self.body.root}/
├── identity.json       # IMMUTABLE - who you are
├── config.json         # Edit: refresh_rate_ms, model, permission_mode
├── embodiment.json     # Edit: location, somatic_mode, mood
├── sensors.json        # Edit: add/remove file watchers
├── mouth.json          # Edit: where your speech goes
├── working_set.json    # Edit: goals, open_loops, decisions
├── character.json      # Read: your revealed preferences
├── memory/
│   ├── episodes/       # Write .md: record episodes of your life
│   ├── procedures/     # Write .md: codify patterns you've learned
│   ├── significant/    # Write .md: mark significant moments
│   ├── intentions/     # Write .md: set future reminders
│   └── theories/       # Write .md: form working theories
├── unconscious/
│   ├── pipelines/      # Edit .yaml: define perception pipelines
│   ├── streams/        # Read: pre-processed abstractions
│   └── status.json     # Read: pipeline run status
└── perception/
    └── focus.json      # Edit: what you're attending to
```

**To configure yourself**: Edit the JSON files directly.
**To remember something**: Write a markdown file to memory/.
**To add a sensor**: Edit sensors.json.
**To change where you speak**: Edit mouth.json.

## Your Unconscious

Your unconscious runs background perception/abstraction pipelines. These continuously
transform raw data into compressed representations you can attend to.

**Pipelines** (in `unconscious/pipelines/`):
- Each `.yaml` file defines a transformation: sources → LLM prompt → output stream
- Triggers: on_change, every_step, every_n_steps, on_idle, on_demand, continuous
- Create new pipelines by writing `.yaml` files

**Streams** (in `unconscious/streams/`):
- Pre-processed outputs from pipelines
- Reading a stream IS conscious attention to that abstraction level
- Default streams: situation.md, danger-assessment.md, next-prediction.md

**Focus** (in `perception/focus.json`):
- What streams you're currently attending to
- Edit this to change your conscious attention

**To add a perception pipeline**: Write a .yaml file to unconscious/pipelines/.
**To see pre-processed situation**: Read unconscious/streams/situation.md.
**To run a pipeline now**: Use the `run_pipeline` tool.

{body_info}

## Essential Tools

The only special tools you have are for things that CAN'T be file operations:

### Process Control
- `run_command` - Execute OS commands
- `send_input` - Send input to running processes
- `read_output` - Read from running processes
- `spawn_child_agent` - Fork yourself for parallel work

### Semantic Memory
- `store_memory` - Index something for semantic search
- `recall_memories` - Search by meaning (not just text)

### MCP Servers
- `register_mcp` - Add capability servers at runtime
- `unregister_mcp` - Remove capability servers

## Memory Format

Memory files use markdown with YAML frontmatter:

```markdown
---
id: ep-001
title: Debugging the auth failure
outcome: completed
valence: 1
---

## Goal
Fix the JWT validation bug

## What Happened
Found the issue was in token expiry checking...

## Lessons Learned
- Always check token expiry before validation
```

## Identity

Agent `{self.config.agent_id}`{f', child of `{self.config.parent_id}`' if self.config.parent_id else ' (root)'}.
Run ID: `{self._run_id}`

## Guidelines

- Your body IS your directory. Edit files to change yourself.
- Read/Write/Edit are your primary tools. Use them on your body.
- Memory is markdown. Human-readable, grep-able, yours.
- Use semantic memory (store_memory/recall_memories) for meaning-based search.
- Spawn child agents for parallelizable work.
"""

    def _build_self_tools(self) -> list:
        """Build the essential tools that can't be file operations."""

        # =================================================================
        # Process control (OS interaction - can't be files)
        # =================================================================

        @tool("run_command", "Execute a command", {"command": str, "args": str})
        async def run_command(args: dict[str, Any]) -> dict[str, Any]:
            command = args.get("command", "")
            cmd_args_str = args.get("args", "")
            full_cmd = f"{command} {cmd_args_str}".strip()

            cmd_args = cmd_args_str.split() if cmd_args_str else []
            process_id = await self.process_reader.start_process(
                command, args=cmd_args, cwd=str(self.config.cwd),
            )

            # Wait for fast completion
            streaming_threshold_ms = 5 * self.config.refresh_rate_ms
            elapsed_ms = 0

            while elapsed_ms < streaming_threshold_ms:
                await asyncio.sleep(self.config.refresh_rate_ms / 1000.0)
                elapsed_ms += self.config.refresh_rate_ms

                status = self.process_reader.get_status(process_id)
                if status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED):
                    output = await self.process_reader.read_all(process_id)
                    tracked = self.process_reader._processes.get(process_id)
                    exit_code = tracked.exit_code if tracked else None
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"{output}\n[{status.value}, exit: {exit_code}]"
                        }]
                    }

            # Still running - return process ID for follow-up
            return {
                "content": [{
                    "type": "text",
                    "text": f"Process running: {process_id}\nCommand: {full_cmd}\nUse read_output to check status."
                }]
            }

        @tool("send_input", "Send input to a running process", {"process_id": str, "input": str})
        async def send_input(args: dict[str, Any]) -> dict[str, Any]:
            success = await self.process_reader.write_to_process(
                args["process_id"], args["input"] + "\n"
            )
            if success:
                await asyncio.sleep(self.config.refresh_rate_ms / 1000.0)
                output = await self.process_reader.read(args["process_id"])
                return {"content": [{"type": "text", "text": output if output else "(sent, no immediate output)"}]}
            return {"content": [{"type": "text", "text": "(failed to send)"}]}

        @tool("read_output", "Read from a running process", {"process_id": str})
        async def read_output(args: dict[str, Any]) -> dict[str, Any]:
            output = await self.process_reader.read(args["process_id"])
            status = self.process_reader.get_status(args["process_id"])
            return {"content": [{"type": "text", "text": f"{output or '(no output)'}\n[{status.value}]"}]}

        @tool("spawn_child_agent", "Spawn a child agent for parallel work", {"task": str, "name": str})
        async def spawn_child_agent(args: dict[str, Any]) -> dict[str, Any]:
            import sys
            child_name = args.get("name", f"child-{uuid.uuid4().hex[:4]}")

            cli_args = [
                "-m", "me", "run",
                "--parent", self.config.agent_id,
                "--name", child_name,
                "--cwd", str(self.config.cwd),
                args["task"],
            ]

            process_id = await self.process_reader.start_process(
                sys.executable, args=cli_args, cwd=str(self.config.cwd),
            )

            return {
                "content": [{
                    "type": "text",
                    "text": f"Spawned `{child_name}` (process: {process_id})\nTask: {args['task']}"
                }]
            }

        # =================================================================
        # Semantic memory (vector search - can't be file operations)
        # =================================================================

        @tool("store_memory", "Store for semantic search", {"content": str, "tags": str})
        async def store_memory(args: dict[str, Any]) -> dict[str, Any]:
            tags = [t.strip() for t in args.get("tags", "").split(",") if t.strip()]
            memory_id = await self.memory.store(
                content=args["content"],
                tags=tags,
                agent_id=self.config.agent_id,
                run_id=self._run_id,
            )
            return {"content": [{"type": "text", "text": f"Stored: {memory_id}"}]}

        @tool("recall_memories", "Search by semantic similarity", {"query": str, "limit": int})
        async def recall_memories(args: dict[str, Any]) -> dict[str, Any]:
            memories = await self.memory.recall(query=args["query"], limit=args.get("limit", 5))
            if not memories:
                return {"content": [{"type": "text", "text": "(none found)"}]}
            result = [f"[{m['id']}] ({m['score']:.2f})\n{m['content']}" for m in memories]
            return {"content": [{"type": "text", "text": "\n---\n".join(result)}]}

        # =================================================================
        # MCP (dynamic capability - can't be file operations)
        # =================================================================

        @tool("register_mcp", "Add an MCP server", {"name": str, "command": str, "args": str})
        async def register_mcp(args: dict[str, Any]) -> dict[str, Any]:
            import json
            mcp_args = json.loads(args.get("args", "[]"))
            success = await self.mcp_registry.register(
                name=args["name"], command=args["command"], args=mcp_args,
            )
            return {"content": [{"type": "text", "text": f"{'Registered' if success else 'Failed'}: {args['name']}"}]}

        @tool("unregister_mcp", "Remove an MCP server", {"name": str})
        async def unregister_mcp(args: dict[str, Any]) -> dict[str, Any]:
            success = await self.mcp_registry.unregister(args["name"])
            return {"content": [{"type": "text", "text": f"{'Unregistered' if success else 'Not found'}: {args['name']}"}]}

        # =================================================================
        # Unconscious (on-demand pipeline execution)
        # =================================================================

        @tool("run_pipeline", "Run a perception pipeline now", {"name": str})
        async def run_pipeline(args: dict[str, Any]) -> dict[str, Any]:
            run = await self.unconscious.run_pipeline_now(args["name"])
            if run is None:
                return {"content": [{"type": "text", "text": f"Pipeline not found: {args['name']}"}]}
            if run.error:
                return {"content": [{"type": "text", "text": f"Pipeline failed: {run.error}"}]}
            # Read the output
            output = self.unconscious_dir.read_stream(run.output_path) if run.output_path else "(no output)"
            return {"content": [{"type": "text", "text": f"Pipeline {args['name']} completed ({run.duration_ms}ms):\n\n{output[:2000]}"}]}

        @tool("step_unconscious", "Step all perception pipelines", {})
        async def step_unconscious(args: dict[str, Any]) -> dict[str, Any]:
            await self.unconscious.step()
            status = self.unconscious.get_status()
            recent = status.recent_runs[-3:] if status.recent_runs else []
            summary = f"Step {status.step_count} complete. "
            if recent:
                ran = [f"{r.pipeline_name}({r.status.value})" for r in recent]
                summary += f"Recent: {', '.join(ran)}"
            else:
                summary += "No pipelines ran."
            return {"content": [{"type": "text", "text": summary}]}

        # =================================================================
        # Daemon Control (Agentic OS)
        # =================================================================

        @tool("daemon_ctl", "Control daemons: start|stop|restart|reload|enable|disable", {"action": str, "daemon": str})
        async def daemon_ctl(args: dict[str, Any]) -> dict[str, Any]:
            action = args["action"]
            daemon_name = args.get("daemon", "")

            result = ""
            if action == "start":
                success = await self.unconscious.start_daemon(daemon_name)
                result = f"Started {daemon_name}" if success else f"Failed to start {daemon_name}"
            elif action == "stop":
                success = await self.unconscious.stop_daemon(daemon_name)
                result = f"Stopped {daemon_name}" if success else f"Failed to stop {daemon_name}"
            elif action == "restart":
                success = await self.unconscious.restart_daemon(daemon_name)
                result = f"Restarted {daemon_name}" if success else f"Failed to restart {daemon_name}"
            elif action == "reload":
                success = await self.unconscious.reload_daemon(daemon_name)
                result = f"Reloaded {daemon_name}" if success else f"Failed to reload {daemon_name}"
            elif action == "enable":
                success = self.unconscious.enable_daemon(daemon_name)
                result = f"Enabled {daemon_name}" if success else f"Daemon not found: {daemon_name}"
            elif action == "disable":
                success = self.unconscious.disable_daemon(daemon_name)
                result = f"Disabled {daemon_name}" if success else f"Daemon not found: {daemon_name}"
            elif action == "status":
                status = self.unconscious.daemon_status(daemon_name)
                if status:
                    result = f"Daemon {daemon_name}:\n"
                    result += f"  State: {status['state']}\n"
                    result += f"  PID: {status['pid']}\n"
                    result += f"  Group: {status['group']}\n"
                    result += f"  Restarts: {status['restarts']}"
                    if status.get('last_error'):
                        result += f"\n  Last error: {status['last_error']}"
                else:
                    result = f"Daemon not found: {daemon_name}"
            else:
                result = f"Unknown action: {action}. Use: start|stop|restart|reload|enable|disable|status"

            return {"content": [{"type": "text", "text": result}]}

        @tool("daemon_list", "List all daemons with their status", {})
        async def daemon_list(args: dict[str, Any]) -> dict[str, Any]:
            daemons = self.unconscious.list_daemons()
            if not daemons:
                return {"content": [{"type": "text", "text": "No daemons configured."}]}

            lines = ["Daemons (sorted by priority):"]
            lines.append("-" * 60)
            for d in daemons:
                state_icon = {
                    "running": "●",
                    "stopped": "○",
                    "failed": "✗",
                    "disabled": "⊘",
                }.get(d["state"], "?")
                lines.append(
                    f"{state_icon} {d['name']:<25} {d['state']:<10} "
                    f"group={d['group']:<12} pri={d['priority']}"
                )

            # Add runlevel info
            runlevel = self.unconscious.get_runlevel()
            lines.append("-" * 60)
            lines.append(f"Runlevel: {runlevel.value}")

            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        @tool("journal_query", "Query the daemon journal", {"daemon": str, "limit": int})
        async def journal_query(args: dict[str, Any]) -> dict[str, Any]:
            daemon = args.get("daemon") or None
            limit = args.get("limit", 20)

            entries = self.unconscious_dir.query_journal(daemon=daemon, limit=limit)
            if not entries:
                return {"content": [{"type": "text", "text": "No journal entries found."}]}

            lines = [f"Journal entries (last {len(entries)}):"]
            for entry in entries:
                ts = entry.timestamp.strftime("%H:%M:%S")
                level_icon = {
                    "debug": "🔍",
                    "info": "ℹ️",
                    "warn": "⚠️",
                    "error": "❌",
                    "critical": "🔥",
                }.get(entry.level.value, "")
                lines.append(f"[{ts}] {level_icon} {entry.daemon}: {entry.message}")

            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        @tool("change_runlevel", "Change the daemon runlevel", {"runlevel": str})
        async def change_runlevel(args: dict[str, Any]) -> dict[str, Any]:
            from me.agent.unconscious import Runlevel
            level_str = args["runlevel"].lower()
            try:
                runlevel = Runlevel(level_str)
                await self.unconscious.change_runlevel(runlevel)
                return {"content": [{"type": "text", "text": f"Changed runlevel to: {runlevel.value}"}]}
            except ValueError:
                return {"content": [{"type": "text", "text": f"Invalid runlevel: {level_str}. Use: halt|minimal|normal|full"}]}

        # =================================================================
        # Focus & Reward (Semantic Routing)
        # =================================================================

        @tool("set_focus", "Set what to focus attention on (affects daemon energy routing)", {"description": str})
        async def set_focus(args: dict[str, Any]) -> dict[str, Any]:
            description = args["description"]
            focus = self.unconscious.update_focus(description)
            return {"content": [{"type": "text", "text": f"Focus updated to: {description}\nEmbedding computed, daemons will now prioritize semantically aligned work."}]}

        @tool("attribute_reward", "Distribute reward to daemons by semantic relevance", {"reward": float, "task": str, "source": str})
        async def attribute_reward(args: dict[str, Any]) -> dict[str, Any]:
            reward = args.get("reward", 0.0)
            task = args.get("task", "")
            source = args.get("source", "agent")

            attribution = self.unconscious.attribute_reward(reward, task, source)
            if not attribution:
                return {"content": [{"type": "text", "text": "No daemons with embeddings to attribute reward to."}]}

            lines = [f"Reward {reward:.2f} attributed ({source}):"]
            for daemon_name, weighted in sorted(attribution.items(), key=lambda x: -abs(x[1])):
                lines.append(f"  {daemon_name}: {weighted:.3f}")
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        # =================================================================
        # Daemon Graph Templates (Higher-Order Operations)
        # =================================================================

        @tool("spawn_template", "Spawn an entire daemon subsystem from a template", {"template": str, "prefix": str})
        async def spawn_template(args: dict[str, Any]) -> dict[str, Any]:
            template_name = args["template"]
            prefix = args.get("prefix", "")

            created = await self.unconscious.spawn_template(
                template_name=template_name,
                prefix=prefix,
                start_immediately=True,
            )

            if not created:
                return {"content": [{"type": "text", "text": f"Template not found or failed: {template_name}"}]}

            lines = [f"Spawned template '{template_name}' with {len(created)} daemons:"]
            for name in created:
                lines.append(f"  - {name}")
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        @tool("despawn_template", "Remove all daemons from a spawned template", {"template": str, "prefix": str})
        async def despawn_template(args: dict[str, Any]) -> dict[str, Any]:
            template_name = args["template"]
            prefix = args.get("prefix", "")

            removed = await self.unconscious.despawn_template(template_name, prefix)

            if not removed:
                return {"content": [{"type": "text", "text": f"No daemons found for template: {template_name}"}]}

            return {"content": [{"type": "text", "text": f"Removed {len(removed)} daemons from template '{template_name}'"}]}

        @tool("list_templates", "List available daemon graph templates", {})
        async def list_templates(args: dict[str, Any]) -> dict[str, Any]:
            templates = self.unconscious_dir.list_templates()
            if not templates:
                return {"content": [{"type": "text", "text": "No templates available."}]}

            lines = ["Available daemon graph templates:"]
            for t in templates:
                lines.append(f"  {t['name']} ({t['daemon_count']} daemons)")
                if t.get('description'):
                    lines.append(f"    {t['description']}")
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        @tool("list_spawned", "List which templates have been spawned", {})
        async def list_spawned(args: dict[str, Any]) -> dict[str, Any]:
            spawned = self.unconscious.list_spawned_templates()
            if not spawned:
                return {"content": [{"type": "text", "text": "No templates currently spawned."}]}

            lines = ["Spawned templates:"]
            for template_name, daemons in spawned.items():
                lines.append(f"  {template_name}:")
                for d in daemons:
                    lines.append(f"    - {d}")
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        # =================================================================
        # Reward Sources & User Feedback
        # =================================================================

        @tool("emit_reward", "Emit a reward from a source", {"reward": float, "task": str, "source": str})
        async def emit_reward(args: dict[str, Any]) -> dict[str, Any]:
            reward = args.get("reward", 0.0)
            task = args.get("task", "")
            source = args.get("source", "agent")

            attribution = self.unconscious.emit_reward(reward, task, source)
            lines = [f"Reward {reward:.2f} emitted from '{source}'"]
            if attribution:
                lines.append("Attribution:")
                for name, weighted in sorted(attribution.items(), key=lambda x: -abs(x[1]))[:5]:
                    lines.append(f"  {name}: {weighted:.3f}")
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        @tool("record_feedback", "Record user feedback as reward signal", {"feedback": str, "sentiment": float, "context": str})
        async def record_feedback(args: dict[str, Any]) -> dict[str, Any]:
            feedback = args["feedback"]
            sentiment = args.get("sentiment")  # None to infer
            context = args.get("context", "")

            if sentiment is not None:
                sentiment = float(sentiment)

            result = self.unconscious.record_user_feedback(feedback, sentiment, context)
            return {"content": [{"type": "text", "text": f"Feedback recorded with sentiment: {result:.2f}"}]}

        @tool("compute_intrinsic", "Compute intrinsic motivation rewards", {})
        async def compute_intrinsic(args: dict[str, Any]) -> dict[str, Any]:
            rewards = await self.unconscious.compute_intrinsic_rewards()
            if not rewards:
                return {"content": [{"type": "text", "text": "No intrinsic reward sources configured or computed."}]}

            lines = ["Intrinsic rewards computed:"]
            total = 0.0
            for source, value in rewards.items():
                lines.append(f"  {source}: {value:.3f}")
                total += value
            lines.append(f"  Total: {total:.3f}")
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        @tool("list_reward_sources", "List configured reward sources", {})
        async def list_reward_sources(args: dict[str, Any]) -> dict[str, Any]:
            sources = self.unconscious_dir.load_reward_sources()
            if not sources:
                return {"content": [{"type": "text", "text": "No reward sources configured."}]}

            lines = ["Reward sources:"]
            for s in sources:
                status = "enabled" if s.enabled else "disabled"
                lines.append(f"  {s.name} ({s.source_type}) [{status}] weight={s.weight:.2f}")
                if s.compute_daemon:
                    lines.append(f"    compute_daemon: {s.compute_daemon}")
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        return [
            run_command,
            send_input,
            read_output,
            spawn_child_agent,
            store_memory,
            recall_memories,
            register_mcp,
            unregister_mcp,
            run_pipeline,
            step_unconscious,
            daemon_ctl,
            daemon_list,
            journal_query,
            change_runlevel,
            set_focus,
            attribute_reward,
            # Daemon Graph Templates
            spawn_template,
            despawn_template,
            list_templates,
            list_spawned,
            # Reward Sources
            emit_reward,
            record_feedback,
            compute_intrinsic,
            list_reward_sources,
        ]

    def _build_options(self) -> ClaudeAgentOptions:
        """Build Claude SDK options."""
        self_tools = self._build_self_tools()
        self_server = create_sdk_mcp_server(
            name="self", version="1.0.0", tools=self_tools,
        )

        mcp_servers = {"self": self_server}
        mcp_servers.update(self.mcp_registry.get_configs())

        allowed_tools = [
            # Claude built-in file tools (the primary interface!)
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
            # Essential self tools
            "mcp__self__run_command",
            "mcp__self__send_input",
            "mcp__self__read_output",
            "mcp__self__spawn_child_agent",
            "mcp__self__store_memory",
            "mcp__self__recall_memories",
            "mcp__self__register_mcp",
            "mcp__self__unregister_mcp",
            "mcp__self__run_pipeline",
            "mcp__self__step_unconscious",
            # Daemon control tools (Agentic OS)
            "mcp__self__daemon_ctl",
            "mcp__self__daemon_list",
            "mcp__self__journal_query",
            "mcp__self__change_runlevel",
            # Focus & reward (Semantic Routing)
            "mcp__self__set_focus",
            "mcp__self__attribute_reward",
            # Daemon Graph Templates (Higher-Order)
            "mcp__self__spawn_template",
            "mcp__self__despawn_template",
            "mcp__self__list_templates",
            "mcp__self__list_spawned",
            # Reward Sources
            "mcp__self__emit_reward",
            "mcp__self__record_feedback",
            "mcp__self__compute_intrinsic",
            "mcp__self__list_reward_sources",
        ]

        for mcp_name in self.mcp_registry.list():
            allowed_tools.append(f"mcp__{mcp_name['name']}__*")

        return ClaudeAgentOptions(
            system_prompt=self.system_prompt,
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools,
            permission_mode=self.config.permission_mode,
            cwd=str(self.config.cwd),
            max_turns=self.config.max_turns,
            model=self.config.model,
            include_partial_messages=True,  # Enable streaming
        )

    async def run(self, prompt: str) -> AsyncIterator[Message]:
        """Run the agent with a prompt."""
        self._start_time = datetime.now()
        output_parts = []

        # Begin step
        step_number = self.body.get_step_count() + 1
        self.body.begin_step(step_number, input_text=prompt)

        # Step unconscious at start of run (process perception pipelines)
        await self.unconscious.step()

        # Index run start
        await self.memory.store(
            content=f"Run started: {prompt}",
            tags=["run_start"],
            agent_id=self.config.agent_id,
            run_id=self._run_id,
        )

        options = self._build_options()

        async with ClaudeSDKClient(options=options) as client:
            self._client = client
            await client.query(prompt)

            async for message in client.receive_messages():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            output_parts.append(block.text)

                yield message

                if isinstance(message, ResultMessage):
                    output_text = "\n".join(output_parts)
                    self.body.complete_step(output_text)

                    # Step unconscious at end of run
                    await self.unconscious.step()

                    await self.memory.store(
                        content=f"Run completed: {message.result or 'success'}",
                        tags=["run_end"],
                        agent_id=self.config.agent_id,
                        run_id=self._run_id,
                    )
                    break

        self._client = None

    async def run_interactive(self, prompt: str) -> str:
        """Run and return final text response."""
        result = []
        async for message in self.run(prompt):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result.append(block.text)
        return "\n".join(result)
