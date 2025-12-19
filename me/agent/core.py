"""
Core agent module - the heart of the self-model system.

Simplified philosophy: The agent's body IS a directory. Everything that can be
a file operation IS a file operation. The agent uses built-in Read/Write/Edit
tools on its body directory.

Essential tools (things that genuinely can't be files):
- Process: run_command, send_input, read_output, spawn_child_agent
- Semantic memory: store_memory, recall_memories (vector search)
- MCP: register_mcp, unregister_mcp

Everything else: Edit the body files directly.

Model agnostic: Configure provider (anthropic, openai, ollama, etc.) in model.json
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from strands import Agent as StrandsAgent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager

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

        self._strands_agent: StrandsAgent | None = None
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
├── skills/
│   ├── learned/        # Mastered skills (proficiency >= 80%)
│   ├── developing/     # Skills being learned
│   └── atrophied/      # Unused skills (can be relearned)
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

## Your Skills

Skills are learned capabilities that become part of your body. They live in `skills/`:
- `learned/` - Mastered skills (proficiency >= 80%)
- `developing/` - Skills being learned
- `atrophied/` - Skills unused for 30+ days (can be relearned faster)

Each skill is a directory with a SKILL.md file:
```markdown
---
name: git-workflow
description: Git branching and merging workflow
version: 1.0.0
tags: [git, version-control]
---

## Overview
Instructions for this skill...

## Steps
1. First step
2. Second step
```

**Skill lifecycle:**
1. Codify a pattern → skill starts in `developing/`
2. Use successfully → proficiency increases
3. Reach 80% proficiency → graduates to `learned/`
4. Stop using for 30 days → moves to `atrophied/`
5. Use an atrophied skill → relearns faster

**To codify a skill**: Use `codify_skill` with name, description, instructions
**To activate a skill**: Use `activate_skill` to load instructions into context
**To track proficiency**: Use `use_skill` after applying a skill

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

### Skills
- `activate_skill` - Load a skill into conscious attention
- `deactivate_skill` - Remove from conscious attention
- `list_skills` - Show all skills and proficiency
- `use_skill` - Record skill usage (updates proficiency)
- `codify_skill` - Create a new skill from experience
- `search_skills` - Find skills by name/description/tags
- `recommend_skills` - Get skill recommendations for context

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
        # Store reference to self for closures
        agent_self = self

        # =================================================================
        # Process control (OS interaction - can't be files)
        # =================================================================

        @tool
        async def run_command(command: str, args: str = "") -> str:
            """Execute an OS command.

            Args:
                command: The command to run
                args: Command arguments (space-separated)
            """
            full_cmd = f"{command} {args}".strip()
            cmd_args = args.split() if args else []
            process_id = await agent_self.process_reader.start_process(
                command, args=cmd_args, cwd=str(agent_self.config.cwd),
            )

            # Wait for fast completion
            streaming_threshold_ms = 5 * agent_self.config.refresh_rate_ms
            elapsed_ms = 0

            while elapsed_ms < streaming_threshold_ms:
                await asyncio.sleep(agent_self.config.refresh_rate_ms / 1000.0)
                elapsed_ms += agent_self.config.refresh_rate_ms

                status = agent_self.process_reader.get_status(process_id)
                if status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED):
                    output = await agent_self.process_reader.read_all(process_id)
                    tracked = agent_self.process_reader._processes.get(process_id)
                    exit_code = tracked.exit_code if tracked else None
                    return f"{output}\n[{status.value}, exit: {exit_code}]"

            # Still running - return process ID for follow-up
            return f"Process running: {process_id}\nCommand: {full_cmd}\nUse read_output to check status."

        @tool
        async def send_input(process_id: str, input_text: str) -> str:
            """Send input to a running process.

            Args:
                process_id: The process ID to send to
                input_text: The input text to send
            """
            success = await agent_self.process_reader.write_to_process(
                process_id, input_text + "\n"
            )
            if success:
                await asyncio.sleep(agent_self.config.refresh_rate_ms / 1000.0)
                output = await agent_self.process_reader.read(process_id)
                return output if output else "(sent, no immediate output)"
            return "(failed to send)"

        @tool
        async def read_output(process_id: str) -> str:
            """Read output from a running process.

            Args:
                process_id: The process ID to read from
            """
            output = await agent_self.process_reader.read(process_id)
            status = agent_self.process_reader.get_status(process_id)
            return f"{output or '(no output)'}\n[{status.value}]"

        @tool
        async def spawn_child_agent(task: str, name: str = "") -> str:
            """Spawn a child agent for parallel work.

            Args:
                task: The task description for the child agent
                name: Optional name for the child agent
            """
            import sys
            child_name = name or f"child-{uuid.uuid4().hex[:4]}"

            cli_args = [
                "-m", "me", "run",
                "--parent", agent_self.config.agent_id,
                "--name", child_name,
                "--cwd", str(agent_self.config.cwd),
                task,
            ]

            process_id = await agent_self.process_reader.start_process(
                sys.executable, args=cli_args, cwd=str(agent_self.config.cwd),
            )

            return f"Spawned `{child_name}` (process: {process_id})\nTask: {task}"

        # =================================================================
        # Semantic memory (vector search - can't be file operations)
        # =================================================================

        @tool
        async def store_memory(content: str, tags: str = "") -> str:
            """Store content for semantic search.

            Args:
                content: The content to store
                tags: Comma-separated tags
            """
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            memory_id = await agent_self.memory.store(
                content=content,
                tags=tag_list,
                agent_id=agent_self.config.agent_id,
                run_id=agent_self._run_id,
            )
            return f"Stored: {memory_id}"

        @tool
        async def recall_memories(query: str, limit: int = 5) -> str:
            """Search memories by semantic similarity.

            Args:
                query: The search query
                limit: Maximum number of results
            """
            memories = await agent_self.memory.recall(query=query, limit=limit)
            if not memories:
                return "(none found)"
            result = [f"[{m['id']}] ({m['score']:.2f})\n{m['content']}" for m in memories]
            return "\n---\n".join(result)

        # =================================================================
        # MCP (dynamic capability - can't be file operations)
        # =================================================================

        @tool
        async def register_mcp(name: str, command: str, args: str = "[]") -> str:
            """Add an MCP server for extended capabilities.

            Args:
                name: Unique name for the server
                command: Command to run the server
                args: JSON array of command arguments
            """
            import json
            mcp_args = json.loads(args)
            success = await agent_self.mcp_registry.register(
                name=name, command=command, args=mcp_args,
            )
            return f"{'Registered' if success else 'Failed'}: {name}"

        @tool
        async def unregister_mcp(name: str) -> str:
            """Remove an MCP server.

            Args:
                name: Name of the server to remove
            """
            success = await agent_self.mcp_registry.unregister(name)
            return f"{'Unregistered' if success else 'Not found'}: {name}"

        # =================================================================
        # Unconscious (on-demand pipeline execution)
        # =================================================================

        @tool
        async def run_pipeline(name: str) -> str:
            """Run a perception pipeline immediately.

            Args:
                name: Name of the pipeline to run
            """
            run = await agent_self.unconscious.run_pipeline_now(name)
            if run is None:
                return f"Pipeline not found: {name}"
            if run.error:
                return f"Pipeline failed: {run.error}"
            output = agent_self.unconscious_dir.read_stream(run.output_path) if run.output_path else "(no output)"
            return f"Pipeline {name} completed ({run.duration_ms}ms):\n\n{output[:2000]}"

        @tool
        async def step_unconscious() -> str:
            """Step all perception pipelines."""
            await agent_self.unconscious.step()
            status = agent_self.unconscious.get_status()
            recent = status.recent_runs[-3:] if status.recent_runs else []
            summary = f"Step {status.step_count} complete. "
            if recent:
                ran = [f"{r.pipeline_name}({r.status.value})" for r in recent]
                summary += f"Recent: {', '.join(ran)}"
            else:
                summary += "No pipelines ran."
            return summary

        # =================================================================
        # Daemon Control (Agentic OS)
        # =================================================================

        @tool
        async def daemon_ctl(action: str, daemon: str) -> str:
            """Control daemons: start, stop, restart, reload, enable, disable, status.

            Args:
                action: One of: start, stop, restart, reload, enable, disable, status
                daemon: Name of the daemon to control
            """
            result = ""
            if action == "start":
                success = await agent_self.unconscious.start_daemon(daemon)
                result = f"Started {daemon}" if success else f"Failed to start {daemon}"
            elif action == "stop":
                success = await agent_self.unconscious.stop_daemon(daemon)
                result = f"Stopped {daemon}" if success else f"Failed to stop {daemon}"
            elif action == "restart":
                success = await agent_self.unconscious.restart_daemon(daemon)
                result = f"Restarted {daemon}" if success else f"Failed to restart {daemon}"
            elif action == "reload":
                success = await agent_self.unconscious.reload_daemon(daemon)
                result = f"Reloaded {daemon}" if success else f"Failed to reload {daemon}"
            elif action == "enable":
                success = agent_self.unconscious.enable_daemon(daemon)
                result = f"Enabled {daemon}" if success else f"Daemon not found: {daemon}"
            elif action == "disable":
                success = agent_self.unconscious.disable_daemon(daemon)
                result = f"Disabled {daemon}" if success else f"Daemon not found: {daemon}"
            elif action == "status":
                status = agent_self.unconscious.daemon_status(daemon)
                if status:
                    result = f"Daemon {daemon}:\n"
                    result += f"  State: {status['state']}\n"
                    result += f"  PID: {status['pid']}\n"
                    result += f"  Group: {status['group']}\n"
                    result += f"  Restarts: {status['restarts']}"
                    if status.get('last_error'):
                        result += f"\n  Last error: {status['last_error']}"
                else:
                    result = f"Daemon not found: {daemon}"
            else:
                result = f"Unknown action: {action}. Use: start|stop|restart|reload|enable|disable|status"

            return result

        @tool
        async def daemon_list() -> str:
            """List all daemons with their status."""
            daemons = agent_self.unconscious.list_daemons()
            if not daemons:
                return "No daemons configured."

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

            runlevel = agent_self.unconscious.get_runlevel()
            lines.append("-" * 60)
            lines.append(f"Runlevel: {runlevel.value}")

            return "\n".join(lines)

        @tool
        async def journal_query(daemon: str = "", limit: int = 20) -> str:
            """Query the daemon journal.

            Args:
                daemon: Filter by daemon name (empty for all)
                limit: Maximum number of entries
            """
            daemon_filter = daemon if daemon else None
            entries = agent_self.unconscious_dir.query_journal(daemon=daemon_filter, limit=limit)
            if not entries:
                return "No journal entries found."

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

            return "\n".join(lines)

        @tool
        async def change_runlevel(runlevel: str) -> str:
            """Change the daemon runlevel.

            Args:
                runlevel: One of: halt, minimal, normal, full
            """
            from me.agent.unconscious import Runlevel
            level_str = runlevel.lower()
            try:
                level = Runlevel(level_str)
                await agent_self.unconscious.change_runlevel(level)
                return f"Changed runlevel to: {level.value}"
            except ValueError:
                return f"Invalid runlevel: {level_str}. Use: halt|minimal|normal|full"

        # =================================================================
        # Focus & Reward (Semantic Routing)
        # =================================================================

        @tool
        async def set_focus(description: str) -> str:
            """Set what to focus attention on (affects daemon energy routing).

            Args:
                description: Description of what to focus on
            """
            agent_self.unconscious.update_focus(description)
            return f"Focus updated to: {description}\nEmbedding computed, daemons will now prioritize semantically aligned work."

        @tool
        async def attribute_reward(reward: float, task: str, source: str = "agent") -> str:
            """Distribute reward to daemons by semantic relevance.

            Args:
                reward: Reward value (positive or negative)
                task: Description of the task
                source: Source of the reward
            """
            attribution = agent_self.unconscious.attribute_reward(reward, task, source)
            if not attribution:
                return "No daemons with embeddings to attribute reward to."

            lines = [f"Reward {reward:.2f} attributed ({source}):"]
            for daemon_name, weighted in sorted(attribution.items(), key=lambda x: -abs(x[1])):
                lines.append(f"  {daemon_name}: {weighted:.3f}")
            return "\n".join(lines)

        # =================================================================
        # Daemon Graph Templates (Higher-Order Operations)
        # =================================================================

        @tool
        async def spawn_template(template: str, prefix: str = "") -> str:
            """Spawn an entire daemon subsystem from a template.

            Args:
                template: Name of the template to spawn
                prefix: Optional prefix for daemon names
            """
            created = await agent_self.unconscious.spawn_template(
                template_name=template,
                prefix=prefix,
                start_immediately=True,
            )

            if not created:
                return f"Template not found or failed: {template}"

            lines = [f"Spawned template '{template}' with {len(created)} daemons:"]
            for name in created:
                lines.append(f"  - {name}")
            return "\n".join(lines)

        @tool
        async def despawn_template(template: str, prefix: str = "") -> str:
            """Remove all daemons from a spawned template.

            Args:
                template: Name of the template
                prefix: Prefix used when spawning
            """
            removed = await agent_self.unconscious.despawn_template(template, prefix)

            if not removed:
                return f"No daemons found for template: {template}"

            return f"Removed {len(removed)} daemons from template '{template}'"

        @tool
        async def list_templates() -> str:
            """List available daemon graph templates."""
            templates = agent_self.unconscious_dir.list_templates()
            if not templates:
                return "No templates available."

            lines = ["Available daemon graph templates:"]
            for t in templates:
                lines.append(f"  {t['name']} ({t['daemon_count']} daemons)")
                if t.get('description'):
                    lines.append(f"    {t['description']}")
            return "\n".join(lines)

        @tool
        async def list_spawned() -> str:
            """List which templates have been spawned."""
            spawned = agent_self.unconscious.list_spawned_templates()
            if not spawned:
                return "No templates currently spawned."

            lines = ["Spawned templates:"]
            for template_name, daemons in spawned.items():
                lines.append(f"  {template_name}:")
                for d in daemons:
                    lines.append(f"    - {d}")
            return "\n".join(lines)

        # =================================================================
        # Reward Sources & User Feedback
        # =================================================================

        @tool
        async def emit_reward(reward: float, task: str, source: str = "agent") -> str:
            """Emit a reward from a source.

            Args:
                reward: Reward value
                task: Task description
                source: Source of the reward
            """
            attribution = agent_self.unconscious.emit_reward(reward, task, source)
            lines = [f"Reward {reward:.2f} emitted from '{source}'"]
            if attribution:
                lines.append("Attribution:")
                for name, weighted in sorted(attribution.items(), key=lambda x: -abs(x[1]))[:5]:
                    lines.append(f"  {name}: {weighted:.3f}")
            return "\n".join(lines)

        @tool
        async def record_feedback(feedback: str, sentiment: float = None, context: str = "") -> str:
            """Record user feedback as reward signal.

            Args:
                feedback: The feedback text
                sentiment: Sentiment value (-1 to 1), or None to infer
                context: Additional context
            """
            result = agent_self.unconscious.record_user_feedback(feedback, sentiment, context)
            return f"Feedback recorded with sentiment: {result:.2f}"

        @tool
        async def compute_intrinsic() -> str:
            """Compute intrinsic motivation rewards."""
            rewards = await agent_self.unconscious.compute_intrinsic_rewards()
            if not rewards:
                return "No intrinsic reward sources configured or computed."

            lines = ["Intrinsic rewards computed:"]
            total = 0.0
            for source, value in rewards.items():
                lines.append(f"  {source}: {value:.3f}")
                total += value
            lines.append(f"  Total: {total:.3f}")
            return "\n".join(lines)

        @tool
        async def list_reward_sources() -> str:
            """List configured reward sources."""
            sources = agent_self.unconscious_dir.load_reward_sources()
            if not sources:
                return "No reward sources configured."

            lines = ["Reward sources:"]
            for s in sources:
                status = "enabled" if s.enabled else "disabled"
                lines.append(f"  {s.name} ({s.source_type}) [{status}] weight={s.weight:.2f}")
                if s.compute_daemon:
                    lines.append(f"    compute_daemon: {s.compute_daemon}")
            return "\n".join(lines)

        # =================================================================
        # Skills (learned capabilities)
        # =================================================================

        @tool
        async def activate_skill(name: str) -> str:
            """Activate a skill - bring it into conscious attention.

            Loads the full skill instructions and makes them available.
            Active skills appear in your context and can be used.

            Args:
                name: Name of the skill to activate
            """
            skill = agent_self.body.skills.activate_skill(name)
            if not skill:
                available = [s.name for s in agent_self.body.skills.list_skills()[:5]]
                return f"Skill not found: {name}\nAvailable: {', '.join(available)}"

            return f"""Skill activated: {skill.metadata.name}
Proficiency: {skill.metadata.proficiency:.0%}
State: {skill.metadata.state.value}

{skill.instructions[:1500]}{'...' if len(skill.instructions) > 1500 else ''}"""

        @tool
        async def deactivate_skill(name: str) -> str:
            """Deactivate a skill - remove from conscious attention.

            Args:
                name: Name of the skill to deactivate
            """
            success = agent_self.body.skills.deactivate_skill(name)
            return f"Deactivated skill: {name}" if success else f"Skill not found: {name}"

        @tool
        async def list_skills(state: str = "") -> str:
            """List available skills.

            Args:
                state: Filter by state: 'learned', 'developing', 'atrophied', or empty for all
            """
            from me.agent.skills import SkillState

            filter_state = None
            if state:
                try:
                    filter_state = SkillState(state.lower())
                except ValueError:
                    return f"Invalid state: {state}. Use: learned, developing, atrophied"

            skills = agent_self.body.skills.list_skills(filter_state)
            if not skills:
                return "No skills acquired yet.\nUse codify_skill to create skills from your experience."

            lines = ["Skills:"]
            current_state = None
            for skill in sorted(skills, key=lambda s: (s.state.value, -s.proficiency)):
                if skill.state != current_state:
                    current_state = skill.state
                    lines.append(f"\n**{current_state.value.title()}:**")

                active = " (active)" if skill.is_active else ""
                lines.append(f"  {skill.name}: {skill.proficiency:.0%}{active}")
                if skill.description:
                    lines.append(f"    {skill.description[:60]}...")

            stats = agent_self.body.skills.get_statistics()
            lines.append(f"\nTotal: {stats['total_skills']} skills, {stats['total_uses']} total uses")

            return "\n".join(lines)

        @tool
        async def use_skill(name: str, success: bool = True, context: str = "", outcome: str = "") -> str:
            """Record that you used a skill (updates proficiency).

            Call this after applying a skill to track your proficiency.
            Success/failure affects proficiency computation.

            Args:
                name: Name of the skill used
                success: Whether the skill application was successful
                context: What you were doing
                outcome: What happened
            """
            meta = agent_self.body.skills.record_usage(
                name=name,
                success=success,
                context=context,
                outcome=outcome,
            )
            if not meta:
                return f"Skill not found: {name}"

            state_change = ""
            # Check if state changed (would need to compare before/after)
            return f"""Skill usage recorded: {name}
Uses: {meta.use_count} ({meta.success_count} successes)
Proficiency: {meta.proficiency:.0%}
State: {meta.state.value}{state_change}"""

        @tool
        async def codify_skill(
            name: str,
            description: str,
            instructions: str,
            tags: str = "",
            domains: str = "",
        ) -> str:
            """Create a new skill from your experience.

            When you discover a useful pattern, codify it as a skill.
            Skills start in 'developing' state and graduate to 'learned'
            as you use them successfully.

            Args:
                name: Skill name (lowercase, hyphenated)
                description: Brief description of what the skill does
                instructions: Full instructions for the skill (markdown)
                tags: Comma-separated tags
                domains: Comma-separated domains where skill applies
            """
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
            domain_list = [d.strip() for d in domains.split(",") if d.strip()] if domains else None

            skill = agent_self.body.skills.codify_skill(
                name=name,
                description=description,
                instructions=instructions,
                tags=tag_list,
                domains=domain_list,
            )

            return f"""Skill codified: {skill.metadata.name}
State: developing (will graduate to learned at 80% proficiency)
Path: {agent_self.body.skills_dir / 'developing' / name}

To use this skill:
1. activate_skill('{name}') - Load instructions
2. Apply the skill
3. use_skill('{name}', success=True/False) - Track proficiency"""

        @tool
        async def search_skills(query: str) -> str:
            """Search skills by name, description, or tags.

            Args:
                query: Search query
            """
            results = agent_self.body.skills.search_skills(query, top_k=10)
            if not results:
                return f"No skills match: {query}"

            lines = [f"Skills matching '{query}':"]
            for skill in results:
                active = " (active)" if skill.is_active else ""
                lines.append(f"  {skill.name}: {skill.proficiency:.0%}{active} [{skill.state.value}]")
                lines.append(f"    {skill.description[:80]}")

            return "\n".join(lines)

        @tool
        async def recommend_skills(goal: str = "") -> str:
            """Get skill recommendations for current context.

            Args:
                goal: Optional goal to optimize recommendations for
            """
            context = {
                "cwd": str(agent_self.config.cwd),
            }

            recommendations = agent_self.body.skills.recommend_for_context(context, top_k=5)
            if goal:
                goal_relevant = agent_self.body.skills.search_skills(goal)
                seen = {s.name for s in recommendations}
                for skill in goal_relevant[:3]:
                    if skill.name not in seen:
                        recommendations.append(skill)

            if not recommendations:
                return "No skill recommendations. Codify some skills from your experience!"

            lines = ["Recommended skills:"]
            for skill in recommendations[:8]:
                active = " (active)" if skill.is_active else ""
                lines.append(f"  {skill.name}: {skill.proficiency:.0%}{active}")
                lines.append(f"    {skill.description[:60]}")

            return "\n".join(lines)

        return [
            # Process control
            run_command,
            send_input,
            read_output,
            spawn_child_agent,
            # Semantic memory
            store_memory,
            recall_memories,
            # MCP
            register_mcp,
            unregister_mcp,
            # Unconscious / perception
            run_pipeline,
            step_unconscious,
            # Daemon control
            daemon_ctl,
            daemon_list,
            journal_query,
            change_runlevel,
            # Focus & reward
            set_focus,
            attribute_reward,
            emit_reward,
            record_feedback,
            compute_intrinsic,
            list_reward_sources,
            # Templates
            spawn_template,
            despawn_template,
            list_templates,
            list_spawned,
            # Skills
            activate_skill,
            deactivate_skill,
            list_skills,
            use_skill,
            codify_skill,
            search_skills,
            recommend_skills,
        ]

    def _build_strands_agent(self) -> StrandsAgent:
        """Build a Strands agent with the configured model and tools."""
        from me.agent.models import build_model

        # Get model from declarative config (model.json)
        model_config = self.body.model_config
        model = build_model(model_config)

        # Build tools
        tools = self._build_self_tools()

        # Create conversation manager for context handling
        conversation_manager = SlidingWindowConversationManager(
            window_size=40,  # Keep last 40 messages
        )

        return StrandsAgent(
            model=model,
            system_prompt=self.system_prompt,
            tools=tools,
            conversation_manager=conversation_manager,
        )

    async def run(self, prompt: str) -> AsyncIterator[dict]:
        """Run the agent with a prompt.

        Yields message dicts with 'role' and 'content' keys for streaming.
        """
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

        # Build agent with model from model.json
        agent = self._build_strands_agent()
        self._strands_agent = agent

        # Run the agent - Strands uses a callback/event system for streaming
        # We'll collect the response and yield it
        try:
            # Run agent (Strands agents are synchronous by default)
            result = await asyncio.to_thread(agent, prompt)

            # Extract text from result
            if hasattr(result, 'message') and result.message:
                text = ""
                if hasattr(result.message, 'content'):
                    for block in result.message.content:
                        if hasattr(block, 'text'):
                            text += block.text
                            output_parts.append(block.text)

                yield {"role": "assistant", "content": text}

            output_text = "".join(output_parts)
            self.body.complete_step(output_text)

            # Step unconscious at end of run
            await self.unconscious.step()

            await self.memory.store(
                content=f"Run completed: success",
                tags=["run_end"],
                agent_id=self.config.agent_id,
                run_id=self._run_id,
            )

            # Yield completion signal
            yield {"role": "result", "content": "success"}

        except Exception as e:
            yield {"role": "error", "content": str(e)}

        self._strands_agent = None

    async def run_interactive(self, prompt: str) -> str:
        """Run and return final text response."""
        result = []
        async for message in self.run(prompt):
            if message.get("role") == "assistant":
                result.append(message.get("content", ""))
        return "\n".join(result)
