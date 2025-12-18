"""
Unconscious - The Daemon Subsystem for an Agentic Operating System.

The agent's unconscious mind operates like daemons in an operating system:
continuously running background processes that transform raw data into
compressed, task-relevant representations. The central agent prompt acts
as the kernel, while these pipelines act as managed daemons.

Architecture (Agentic OS Metaphor):
    ┌─────────────────────────────────────────────────────────────────┐
    │  Traditional OS          →    Agentic OS                        │
    │  ─────────────────────────────────────────────────────────────  │
    │  Kernel                  →    Central agent prompt loop         │
    │  Daemons                 →    Background LLM pipelines          │
    │  Filesystem              →    Body directory (state as files)   │
    │  Scheduler               →    Pipeline priority + triggers      │
    │  IPC                     →    Pipeline outputs → other pipes    │
    │  Signals                 →    start/stop/reload/status          │
    │  Init system             →    Boot pipelines, runlevels         │
    │  Cgroups                 →    Token budgets per group           │
    │  Journal                 →    Structured daemon logs            │
    └─────────────────────────────────────────────────────────────────┘

Directory Structure:
    unconscious/
    ├── pipelines/              # Pipeline definitions (.json)
    ├── streams/                # Pipeline outputs (pre-conscious)
    ├── status.json             # Runner state
    ├── daemons.json            # Daemon states and metadata
    ├── groups.json             # Process group definitions
    ├── journal.jsonl           # Structured daemon log
    └── budgets.json            # Token budget tracking

Key concepts:
    - Daemon: A managed pipeline with full lifecycle (start/stop/restart)
    - Pipeline: Definition of how to transform source(s) into abstraction
    - ProcessGroup: Category of daemons with shared token budget
    - Stream: Output of a pipeline, available for conscious attention
    - Runlevel: System state determining which daemons run
    - Journal: Structured log of all daemon activity

The agent doesn't perceive raw reality—it perceives through configurable
abstraction layers. The agent can control its own daemons, adjusting what
background processing occurs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Callable, AsyncIterator

from pydantic import BaseModel, Field

# For template variable rendering - lazy import to avoid circular
_sentence_model = None

def _get_sentence_model():
    """Lazy-load sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    import numpy as np
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# =============================================================================
# Pipeline Configuration Models
# =============================================================================

class TriggerMode(str, Enum):
    """When a pipeline should run."""
    ON_CHANGE = "on_change"      # Source file(s) changed
    EVERY_STEP = "every_step"    # Every agent step
    EVERY_N_STEPS = "every_n_steps"  # Every N steps
    ON_IDLE = "on_idle"          # When agent has spare cycles
    ON_DEMAND = "on_demand"      # Only when explicitly requested
    CONTINUOUS = "continuous"    # As fast as possible (careful!)


class SourceMode(str, Enum):
    """How to read the source."""
    FULL = "full"                # Entire file
    TAIL = "tail"                # Last N lines
    HEAD = "head"                # First N lines
    DIFF = "diff"                # Changes since last run


class DaemonState(str, Enum):
    """Lifecycle state of a daemon (managed pipeline)."""
    STOPPED = "stopped"          # Not running, can be started
    STARTING = "starting"        # In the process of starting
    RUNNING = "running"          # Active and processing
    STOPPING = "stopping"        # In the process of stopping
    FAILED = "failed"            # Crashed or errored, needs attention
    DISABLED = "disabled"        # Administratively disabled


class Runlevel(str, Enum):
    """System runlevel determining which daemons run."""
    HALT = "halt"                # No daemons running
    MINIMAL = "minimal"          # Only critical daemons (danger detection)
    NORMAL = "normal"            # Standard operation
    FULL = "full"                # All daemons including expensive ones


class LogLevel(str, Enum):
    """Log level for journal entries."""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Resource Budget Models
# =============================================================================

class TokenBudget(BaseModel):
    """Token budget for a daemon or process group."""
    # Limits
    max_tokens_per_run: int = 1000       # Max tokens per single pipeline run
    max_tokens_per_step: int = 5000      # Max tokens per agent step
    max_tokens_per_hour: int = 50000     # Max tokens per hour

    # Usage tracking (reset periodically)
    used_this_step: int = 0
    used_this_hour: int = 0
    hour_started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def can_run(self, estimated_tokens: int = 500) -> bool:
        """Check if there's budget for a run."""
        return (
            estimated_tokens <= self.max_tokens_per_run and
            self.used_this_step + estimated_tokens <= self.max_tokens_per_step and
            self.used_this_hour + estimated_tokens <= self.max_tokens_per_hour
        )

    def deduct(self, tokens: int) -> None:
        """Deduct tokens from budget."""
        self.used_this_step += tokens
        self.used_this_hour += tokens

    def reset_step(self) -> None:
        """Reset step-level budget (called each agent step)."""
        self.used_this_step = 0

    def maybe_reset_hour(self) -> None:
        """Reset hour-level budget if hour has passed."""
        now = datetime.now(UTC)
        if (now - self.hour_started_at).total_seconds() >= 3600:
            self.used_this_hour = 0
            self.hour_started_at = now


class ProcessGroup(BaseModel):
    """
    A group of related daemons with shared resources.

    Like cgroups in Linux, process groups allow resource limits
    and management at a group level.
    """
    name: str
    description: str = ""

    # Members (pipeline names)
    pipelines: list[str] = Field(default_factory=list)

    # Shared budget
    budget: TokenBudget = Field(default_factory=TokenBudget)

    # Scheduling
    priority: int = 5            # Group-level priority (1=highest)
    max_concurrent: int = 2      # Max pipelines running simultaneously

    # State
    enabled: bool = True
    running_count: int = 0       # Currently running pipelines in group


# =============================================================================
# Journal (Structured Logging)
# =============================================================================

class JournalEntry(BaseModel):
    """A structured log entry for daemon activity."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    daemon: str                  # Pipeline/daemon name
    level: LogLevel = LogLevel.INFO
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Serialize to JSON line."""
        return json.dumps(self.model_dump(mode='json'), default=str)

    @classmethod
    def from_jsonl(cls, line: str) -> "JournalEntry":
        """Parse from JSON line."""
        return cls.model_validate(json.loads(line))


# =============================================================================
# Daemon (Managed Pipeline)
# =============================================================================

class Daemon(BaseModel):
    """
    A managed pipeline with full lifecycle.

    Wraps a Pipeline with daemon semantics: state machine,
    restart tracking, error handling, and lifecycle methods.
    """
    name: str                    # Pipeline name (matches pipeline.name)
    state: DaemonState = DaemonState.STOPPED
    pid: int = 0                 # Logical process ID (increments each start)

    # Lifecycle tracking
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    restarts: int = 0
    last_error: str | None = None
    consecutive_failures: int = 0

    # Configuration
    auto_restart: bool = True    # Restart on failure?
    max_restarts: int = 5        # Max restarts before giving up
    restart_delay_ms: int = 1000 # Delay before restart

    # Boot configuration
    enabled_on_boot: bool = True # Start on agent boot?
    boot_order: int = 50         # Lower = starts earlier


class DaemonProfile(BaseModel):
    """
    Semantic profile for a daemon.

    Tracks the daemon's embedding vector (representing its semantic domain),
    recent activity for embedding computation, and reward attribution history.
    """
    daemon_name: str
    embedding: list[float] = Field(default_factory=list)  # 384-dim for all-MiniLM
    embedding_updated: datetime | None = None

    # History for computing embeddings (most recent last)
    recent_prompts: list[str] = Field(default_factory=list)
    recent_outputs: list[str] = Field(default_factory=list)

    # Reward attribution
    total_reward: float = 0.0
    weighted_reward: float = 0.0
    reward_weight_history: list[float] = Field(default_factory=list)

    def add_activity(self, prompt: str, output: str, max_history: int = 10):
        """Record daemon activity for embedding computation."""
        self.recent_prompts.append(prompt[:2000])  # Truncate for storage
        self.recent_outputs.append(output[:2000])
        # Keep bounded history
        if len(self.recent_prompts) > max_history:
            self.recent_prompts = self.recent_prompts[-max_history:]
        if len(self.recent_outputs) > max_history:
            self.recent_outputs = self.recent_outputs[-max_history:]

    def get_embedding_text(self) -> str:
        """Get text to embed for this daemon's profile."""
        # Combine recent prompts and outputs
        texts = []
        for p, o in zip(self.recent_prompts[-5:], self.recent_outputs[-5:]):
            texts.append(f"Prompt: {p[:500]}\nOutput: {o[:500]}")
        return "\n---\n".join(texts) if texts else self.daemon_name


class PipelineTrigger(BaseModel):
    """Configuration for when a pipeline runs."""
    mode: TriggerMode = TriggerMode.ON_CHANGE
    n_steps: int = 5             # For EVERY_N_STEPS
    debounce_ms: int = 1000      # Minimum time between runs
    cooldown_ms: int = 0         # Forced wait after each run


class PipelineSource(BaseModel):
    """A source of data for a pipeline."""
    path: str                    # File path (supports {body}, {cwd} templates)
    mode: SourceMode = SourceMode.FULL
    lines: int = 100             # For TAIL/HEAD modes
    required: bool = True        # Fail if source missing?


class PipelineOutput(BaseModel):
    """An output stream from a pipeline."""
    path: str                    # Relative to streams/
    description: str = ""
    prompt_append: str = ""      # Additional prompt for this output level


class Pipeline(BaseModel):
    """
    A perception/abstraction pipeline definition.

    Pipelines transform source data into compressed representations
    that the agent can attend to instead of raw data.
    """
    name: str
    description: str = ""

    # Sources (what to process)
    sources: list[PipelineSource] = Field(default_factory=list)
    source: str | None = None    # Shorthand for single source
    source_mode: SourceMode = SourceMode.FULL
    source_lines: int = 100

    # Trigger (when to run)
    trigger: PipelineTrigger = Field(default_factory=PipelineTrigger)

    # Processing (how to transform)
    model: str = "haiku"         # LLM model (haiku=cheap, sonnet=better, opus=best)
    prompt: str                  # The transformation prompt
    max_tokens: int = 500
    temperature: float = 0.0     # Deterministic by default

    # Output (where to write)
    output: str                  # Primary output path (relative to streams/)
    outputs: list[PipelineOutput] = Field(default_factory=list)  # Multi-level outputs

    # Metadata
    enabled: bool = True
    priority: int = 5            # 1=highest, 10=lowest
    tags: list[str] = Field(default_factory=list)

    # Dependencies (for pipeline chaining)
    depends_on: list[str] = Field(default_factory=list)   # Pipelines that must run first
    feeds_into: list[str] = Field(default_factory=list)   # Pipelines to trigger after this

    # Process group and resource management
    group: str = "default"       # Process group name
    budget: TokenBudget | None = None  # Per-pipeline budget (overrides group)

    # Init system / boot
    runlevel: Runlevel = Runlevel.NORMAL  # Minimum runlevel to run
    boot_order: int = 50         # Lower = starts earlier (like S01, S99)

    # Special behaviors
    auto_apply_focus: bool = False  # If True, output is applied to focus.json

    def get_sources(self) -> list[PipelineSource]:
        """Get all sources, handling shorthand."""
        if self.sources:
            return self.sources
        if self.source:
            return [PipelineSource(
                path=self.source,
                mode=self.source_mode,
                lines=self.source_lines,
            )]
        return []

    @classmethod
    def from_json(cls, path: Path) -> "Pipeline":
        """Load pipeline from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = self.model_dump(mode='json', exclude_none=True)
        return json.dumps(data, indent=2, default=str)


# =============================================================================
# Pipeline Run State
# =============================================================================

class PipelineRunStatus(str, Enum):
    """Status of a pipeline run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineRun:
    """Record of a single pipeline execution."""
    pipeline_name: str
    status: PipelineRunStatus
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: int | None = None
    source_hashes: dict[str, str] = field(default_factory=dict)
    output_path: str | None = None
    error: str | None = None
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "source_hashes": self.source_hashes,
            "output_path": self.output_path,
            "error": self.error,
            "tokens_used": self.tokens_used,
        }


@dataclass
class UnconsciousStatus:
    """Overall status of the unconscious system."""
    running: bool = False
    step_count: int = 0
    pipelines_loaded: int = 0
    last_step_at: datetime | None = None
    recent_runs: list[PipelineRun] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "step_count": self.step_count,
            "pipelines_loaded": self.pipelines_loaded,
            "last_step_at": self.last_step_at.isoformat() if self.last_step_at else None,
            "recent_runs": [r.to_dict() for r in self.recent_runs[-20:]],
            "errors": self.errors[-10:],
        }


# =============================================================================
# Focus / Attention
# =============================================================================

class Focus(BaseModel):
    """
    What the agent is currently attending to.

    The agent's conscious perception is defined by what it focuses on.
    This is a file the agent can edit to change its attention.

    The focus also serves as the central routing signal for the daemon system:
    daemons semantically close to the focus receive more compute budget.
    """
    # Currently attended streams (unconscious outputs)
    streams: list[str] = Field(default_factory=list)

    # Currently attended raw sources (bypassing unconscious)
    raw: list[str] = Field(default_factory=list)

    # When focus was last changed
    changed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Attention budget (0-1, how much cognitive resource here)
    budget: float = 1.0

    # Auto-focus: streams that should always be included
    auto_include: list[str] = Field(default_factory=lambda: [
        "danger-assessment.md",
        "situation.md",
    ])

    # Semantic focus - the routing signal for daemon energy allocation
    description: str = ""  # Natural language description of current focus
    embedding: list[float] = Field(default_factory=list)  # Computed from description
    embedding_updated: datetime | None = None


# =============================================================================
# Unconscious Directory Manager
# =============================================================================

class UnconsciousDirectory:
    """
    Manages the unconscious directory structure.

    This is the file-based representation of the agent's daemon subsystem:
    - pipelines/: Pipeline (daemon) definitions
    - streams/: Pipeline outputs (pre-conscious)
    - status.json: Runner state
    - daemons.json: Daemon states and metadata
    - groups.json: Process group definitions
    - journal.jsonl: Structured daemon log
    - budgets.json: Token budget tracking
    """

    def __init__(self, body_root: Path):
        self.root = body_root / "unconscious"
        self.pipelines_dir = self.root / "pipelines"
        self.streams_dir = self.root / "streams"
        self.status_path = self.root / "status.json"
        self.daemons_path = self.root / "daemons.json"
        self.groups_path = self.root / "groups.json"
        self.journal_path = self.root / "journal.jsonl"
        self.budgets_path = self.root / "budgets.json"
        self.focus_path = body_root / "perception" / "focus.json"

        self._ensure_structure()

    def _ensure_structure(self):
        """Create directory structure."""
        self.pipelines_dir.mkdir(parents=True, exist_ok=True)
        self.streams_dir.mkdir(parents=True, exist_ok=True)
        self.focus_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Pipelines
    # =========================================================================

    def load_pipelines(self) -> list[Pipeline]:
        """Load all pipeline definitions."""
        pipelines = []
        for path in self.pipelines_dir.glob("*.json"):
            try:
                pipeline = Pipeline.from_json(path)
                if pipeline.enabled:
                    pipelines.append(pipeline)
            except Exception as e:
                # Log error but continue
                print(f"Error loading pipeline {path}: {e}")
        return sorted(pipelines, key=lambda p: p.priority)

    def save_pipeline(self, pipeline: Pipeline) -> Path:
        """Save a pipeline definition."""
        path = self.pipelines_dir / f"{pipeline.name}.json"
        path.write_text(pipeline.to_json())
        return path

    def get_pipeline(self, name: str) -> Pipeline | None:
        """Get a specific pipeline by name."""
        path = self.pipelines_dir / f"{name}.json"
        if path.exists():
            return Pipeline.from_json(path)
        return None

    def delete_pipeline(self, name: str) -> bool:
        """Delete a pipeline definition."""
        path = self.pipelines_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_pipelines(self) -> list[dict[str, Any]]:
        """List all pipelines with basic info."""
        result = []
        for path in self.pipelines_dir.glob("*.json"):
            try:
                pipeline = Pipeline.from_json(path)
                result.append({
                    "name": pipeline.name,
                    "description": pipeline.description,
                    "trigger": pipeline.trigger.mode.value,
                    "enabled": pipeline.enabled,
                    "priority": pipeline.priority,
                })
            except Exception:
                pass
        return sorted(result, key=lambda p: p.get("priority", 5))

    # =========================================================================
    # Streams
    # =========================================================================

    def write_stream(self, name: str, content: str) -> Path:
        """Write to a stream (pipeline output)."""
        path = self.streams_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def read_stream(self, name: str) -> str | None:
        """Read a stream."""
        path = self.streams_dir / name
        if path.exists():
            return path.read_text()
        return None

    def list_streams(self) -> list[dict[str, Any]]:
        """List all streams with metadata."""
        result = []
        for path in self.streams_dir.glob("**/*"):
            if path.is_file():
                stat = path.stat()
                result.append({
                    "name": str(path.relative_to(self.streams_dir)),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        return result

    def delete_stream(self, name: str) -> bool:
        """Delete a stream."""
        path = self.streams_dir / name
        if path.exists():
            path.unlink()
            return True
        return False

    # =========================================================================
    # Status
    # =========================================================================

    def load_status(self) -> UnconsciousStatus:
        """Load runner status."""
        if self.status_path.exists():
            try:
                with open(self.status_path) as f:
                    data = json.load(f)
                return UnconsciousStatus(
                    running=data.get("running", False),
                    step_count=data.get("step_count", 0),
                    pipelines_loaded=data.get("pipelines_loaded", 0),
                    # Don't reconstruct complex objects, just track basics
                )
            except Exception:
                pass
        return UnconsciousStatus()

    def save_status(self, status: UnconsciousStatus):
        """Save runner status."""
        with open(self.status_path, 'w') as f:
            json.dump(status.to_dict(), f, indent=2)

    # =========================================================================
    # Focus
    # =========================================================================

    def load_focus(self) -> Focus:
        """Load current attention focus."""
        if self.focus_path.exists():
            try:
                with open(self.focus_path) as f:
                    return Focus.model_validate(json.load(f))
            except Exception:
                pass
        return Focus()

    def save_focus(self, focus: Focus):
        """Save attention focus."""
        with open(self.focus_path, 'w') as f:
            json.dump(focus.model_dump(mode='json'), f, indent=2)

    # =========================================================================
    # Utilities
    # =========================================================================

    def expand_path(self, template: str, body_root: Path, cwd: Path) -> str:
        """Expand path templates like {body}, {cwd}."""
        return template.format(
            body=str(body_root),
            cwd=str(cwd),
            streams=str(self.streams_dir),
        )

    # =========================================================================
    # Daemons (Managed Pipelines)
    # =========================================================================

    def load_daemons(self) -> dict[str, Daemon]:
        """Load daemon states from file."""
        if self.daemons_path.exists():
            try:
                with open(self.daemons_path) as f:
                    data = json.load(f)
                return {
                    name: Daemon.model_validate(d)
                    for name, d in data.items()
                }
            except Exception:
                pass
        return {}

    def save_daemons(self, daemons: dict[str, Daemon]) -> None:
        """Save daemon states to file."""
        data = {
            name: d.model_dump(mode='json')
            for name, d in daemons.items()
        }
        with open(self.daemons_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def get_daemon(self, name: str) -> Daemon | None:
        """Get a specific daemon by name."""
        daemons = self.load_daemons()
        return daemons.get(name)

    def save_daemon(self, daemon: Daemon) -> None:
        """Save a single daemon state."""
        daemons = self.load_daemons()
        daemons[daemon.name] = daemon
        self.save_daemons(daemons)

    # =========================================================================
    # Process Groups
    # =========================================================================

    def load_groups(self) -> dict[str, ProcessGroup]:
        """Load process groups from file."""
        if self.groups_path.exists():
            try:
                with open(self.groups_path) as f:
                    data = json.load(f)
                return {
                    name: ProcessGroup.model_validate(g)
                    for name, g in data.items()
                }
            except Exception:
                pass
        return {}

    def save_groups(self, groups: dict[str, ProcessGroup]) -> None:
        """Save process groups to file."""
        data = {
            name: g.model_dump(mode='json')
            for name, g in groups.items()
        }
        with open(self.groups_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def get_group(self, name: str) -> ProcessGroup | None:
        """Get a specific process group by name."""
        groups = self.load_groups()
        return groups.get(name)

    def save_group(self, group: ProcessGroup) -> None:
        """Save a single process group."""
        groups = self.load_groups()
        groups[group.name] = group
        self.save_groups(groups)

    # =========================================================================
    # Journal (Structured Logging)
    # =========================================================================

    def log(
        self,
        daemon: str,
        level: LogLevel,
        message: str,
        **metadata: Any,
    ) -> JournalEntry:
        """Write a journal entry."""
        entry = JournalEntry(
            daemon=daemon,
            level=level,
            message=message,
            metadata=metadata,
        )
        # Append to journal file
        with open(self.journal_path, 'a') as f:
            f.write(entry.to_jsonl() + '\n')

        # Rotate if too large (keep last 1000 entries)
        self._maybe_rotate_journal()
        return entry

    def _maybe_rotate_journal(self, max_entries: int = 1000) -> None:
        """Rotate journal file if it exceeds max entries."""
        if not self.journal_path.exists():
            return

        try:
            lines = self.journal_path.read_text().strip().split('\n')
            if len(lines) > max_entries:
                # Keep last max_entries
                lines = lines[-max_entries:]
                self.journal_path.write_text('\n'.join(lines) + '\n')
        except Exception:
            pass

    def query_journal(
        self,
        daemon: str | None = None,
        level: LogLevel | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[JournalEntry]:
        """Query journal entries with filters."""
        if not self.journal_path.exists():
            return []

        entries = []
        try:
            for line in self.journal_path.read_text().strip().split('\n'):
                if not line:
                    continue
                try:
                    entry = JournalEntry.from_jsonl(line)

                    # Apply filters
                    if daemon and entry.daemon != daemon:
                        continue
                    if level and entry.level != level:
                        continue
                    if since and entry.timestamp < since:
                        continue

                    entries.append(entry)
                except Exception:
                    pass
        except Exception:
            pass

        # Return most recent first, limited
        return list(reversed(entries[-limit:]))

    # =========================================================================
    # Budgets
    # =========================================================================

    def load_budgets(self) -> dict[str, TokenBudget]:
        """Load budget tracking from file."""
        if self.budgets_path.exists():
            try:
                with open(self.budgets_path) as f:
                    data = json.load(f)
                return {
                    name: TokenBudget.model_validate(b)
                    for name, b in data.items()
                }
            except Exception:
                pass
        return {}

    def save_budgets(self, budgets: dict[str, TokenBudget]) -> None:
        """Save budget tracking to file."""
        data = {
            name: b.model_dump(mode='json')
            for name, b in budgets.items()
        }
        with open(self.budgets_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def get_budget(self, name: str) -> TokenBudget | None:
        """Get budget for a daemon or group."""
        budgets = self.load_budgets()
        return budgets.get(name)

    def update_budget(self, name: str, budget: TokenBudget) -> None:
        """Update budget for a daemon or group."""
        budgets = self.load_budgets()
        budgets[name] = budget
        self.save_budgets(budgets)

    # =========================================================================
    # Daemon Profiles (Semantic Embeddings)
    # =========================================================================

    @property
    def profiles_dir(self) -> Path:
        """Directory for daemon profiles."""
        return self.root / "profiles"

    def get_profile(self, daemon_name: str) -> DaemonProfile:
        """Get semantic profile for a daemon (creates if not exists)."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        path = self.profiles_dir / f"{daemon_name}.json"
        if path.exists():
            try:
                with open(path) as f:
                    return DaemonProfile.model_validate(json.load(f))
            except Exception:
                pass
        return DaemonProfile(daemon_name=daemon_name)

    def save_profile(self, profile: DaemonProfile) -> None:
        """Save daemon profile."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        path = self.profiles_dir / f"{profile.daemon_name}.json"
        with open(path, 'w') as f:
            json.dump(profile.model_dump(mode='json'), f, indent=2, default=str)

    def list_profiles(self) -> list[str]:
        """List all daemon profile names."""
        if not self.profiles_dir.exists():
            return []
        return [p.stem for p in self.profiles_dir.glob("*.json")]

    def update_profile_embedding(self, daemon_name: str) -> DaemonProfile:
        """Recompute embedding for a daemon profile."""
        profile = self.get_profile(daemon_name)
        text = profile.get_embedding_text()

        try:
            model = _get_sentence_model()
            embedding = model.encode(text, convert_to_numpy=True).tolist()
            profile.embedding = embedding
            profile.embedding_updated = datetime.now(UTC)
            self.save_profile(profile)
        except Exception as e:
            self.log(daemon_name, LogLevel.WARN, f"Failed to compute embedding: {e}")

        return profile


# =============================================================================
# Pipeline Runner (Daemon Manager)
# =============================================================================

class UnconsciousRunner:
    """
    The Daemon Manager for the Agentic Operating System.

    Manages pipelines as daemons with full lifecycle control:
    - Start/stop/restart/reload individual daemons
    - Boot/shutdown sequences with runlevels
    - Process groups with shared resource budgets
    - Dependency resolution for pipeline chaining
    - Token budget tracking and enforcement
    - Structured journal logging

    This is the agent's unconscious mind—continuously processing,
    compressing, predicting, and detecting patterns without requiring
    conscious attention.
    """

    def __init__(
        self,
        unconscious_dir: UnconsciousDirectory,
        body_root: Path,
        cwd: Path,
        llm_caller: Callable[[str, str, int, float], AsyncIterator[str]] | None = None,
    ):
        self.unconscious = unconscious_dir
        self.body_root = body_root
        self.cwd = cwd
        self._llm_caller = llm_caller

        # Pipeline/Daemon state
        self._pipelines: dict[str, Pipeline] = {}  # name -> Pipeline
        self._daemons: dict[str, Daemon] = {}      # name -> Daemon
        self._groups: dict[str, ProcessGroup] = {} # name -> ProcessGroup
        self._budgets: dict[str, TokenBudget] = {} # name -> TokenBudget (runtime)

        # Execution tracking
        self._source_hashes: dict[str, dict[str, str]] = {}  # pipeline -> {source: hash}
        self._last_runs: dict[str, datetime] = {}  # pipeline -> last run time
        self._step_count = 0
        self._running = False
        self._status = UnconsciousStatus()

        # Init system state
        self._runlevel: Runlevel = Runlevel.HALT
        self._pid_counter = 0  # Logical PID generator
        self._booted = False

    # =========================================================================
    # Initialization
    # =========================================================================

    def load_pipelines(self):
        """Load/reload all pipeline definitions."""
        pipelines = self.unconscious.load_pipelines()
        self._pipelines = {p.name: p for p in pipelines}
        self._status.pipelines_loaded = len(self._pipelines)

        # Load persisted daemon states
        self._daemons = self.unconscious.load_daemons()

        # Ensure daemon exists for each pipeline
        for name in self._pipelines:
            if name not in self._daemons:
                self._daemons[name] = Daemon(name=name)

        # Load process groups
        self._groups = self.unconscious.load_groups()
        if not self._groups:
            self._install_default_groups()

        # Load budgets
        self._budgets = self.unconscious.load_budgets()

        self._log("system", LogLevel.INFO, f"Loaded {len(self._pipelines)} pipelines")

    def _install_default_groups(self):
        """Install default process groups."""
        self._groups = {
            "perception": ProcessGroup(
                name="perception",
                description="Raw sensory processing and danger detection",
                priority=1,
                budget=TokenBudget(max_tokens_per_step=2000),
                max_concurrent=2,
            ),
            "abstraction": ProcessGroup(
                name="abstraction",
                description="Higher-level pattern recognition",
                priority=3,
                budget=TokenBudget(max_tokens_per_step=3000),
                max_concurrent=2,
            ),
            "maintenance": ProcessGroup(
                name="maintenance",
                description="Self-monitoring and cleanup",
                priority=7,
                budget=TokenBudget(max_tokens_per_step=1000),
                max_concurrent=1,
            ),
            "default": ProcessGroup(
                name="default",
                description="Default group for unassigned daemons",
                priority=5,
                budget=TokenBudget(max_tokens_per_step=2000),
                max_concurrent=2,
            ),
        }
        self.unconscious.save_groups(self._groups)

    def set_llm_caller(self, caller: Callable[[str, str, int, float], AsyncIterator[str]]):
        """Set the LLM calling function."""
        self._llm_caller = caller

    def _log(self, daemon: str, level: LogLevel, message: str, **metadata):
        """Write to journal."""
        self.unconscious.log(daemon, level, message, **metadata)

    # =========================================================================
    # Template Variable Rendering
    # =========================================================================

    def _resolve_template_var(self, var_path: str, daemon_name: str | None = None) -> str:
        """
        Resolve a dot-notation template variable.

        Supported namespaces:
            body.* - Body state files (embodiment, working_set, config, etc.)
            streams.* - Stream file contents
            daemon.* - Current daemon's profile (reward_weight, total_reward, etc.)
            focus.* - Current focus state

        Examples:
            {body.mood.confidence} -> 0.75
            {body.working_set.goals} -> [{"description": "..."}]
            {streams.situation} -> "The agent is currently..."
            {daemon.reward_weight} -> 0.65
            {focus.description} -> "Completing the authentication feature"
        """
        parts = var_path.split('.', 1)
        if len(parts) < 2:
            return f"{{{var_path}}}"  # Return unchanged if no namespace

        namespace, path = parts

        try:
            if namespace == "body":
                return self._resolve_body_var(path)
            elif namespace == "streams":
                return self._resolve_stream_var(path)
            elif namespace == "daemon" and daemon_name:
                return self._resolve_daemon_var(path, daemon_name)
            elif namespace == "focus":
                return self._resolve_focus_var(path)
            else:
                return f"{{{var_path}}}"  # Unknown namespace
        except Exception as e:
            self._log("system", LogLevel.DEBUG, f"Template var resolution failed: {var_path} - {e}")
            return f"{{{var_path}}}"  # Return unchanged on error

    def _resolve_body_var(self, path: str) -> str:
        """Resolve a body.* variable by reading JSON files."""
        # Map top-level names to files
        file_map = {
            "embodiment": "embodiment.json",
            "mood": "embodiment.json",  # mood is nested in embodiment
            "working_set": "working_set.json",
            "goals": "working_set.json",
            "character": "character.json",
            "config": "config.json",
            "identity": "identity.json",
            "sensors": "sensors.json",
        }

        parts = path.split('.')
        top_key = parts[0]

        # Find the file to read
        filename = file_map.get(top_key)
        if not filename:
            return f"{{body.{path}}}"

        file_path = self.body_root / filename
        if not file_path.exists():
            return ""

        try:
            with open(file_path) as f:
                data = json.load(f)

            # Navigate the path
            value = data
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return f"{{body.{path}}}"

            # Convert to string representation
            if isinstance(value, (dict, list)):
                return json.dumps(value, indent=2, default=str)
            return str(value)

        except Exception:
            return f"{{body.{path}}}"

    def _resolve_stream_var(self, path: str) -> str:
        """Resolve a streams.* variable by reading stream files."""
        # Handle dashes in stream names (e.g., danger-assessment)
        stream_name = path.replace('.', '/')
        if not stream_name.endswith('.md'):
            stream_name += '.md'

        content = self.unconscious.read_stream(stream_name)
        return content if content else ""

    def _resolve_daemon_var(self, path: str, daemon_name: str) -> str:
        """Resolve a daemon.* variable from the daemon's profile."""
        profile = self.unconscious.get_profile(daemon_name)

        if path == "reward_weight":
            if profile.reward_weight_history:
                return str(profile.reward_weight_history[-1])
            return "0.5"
        elif path == "total_reward":
            return str(profile.total_reward)
        elif path == "weighted_reward":
            return str(profile.weighted_reward)
        elif path == "name":
            return daemon_name
        else:
            return f"{{daemon.{path}}}"

    def _resolve_focus_var(self, path: str) -> str:
        """Resolve a focus.* variable."""
        focus = self.unconscious.load_focus()
        focus_dict = focus.model_dump(mode='json')

        parts = path.split('.')
        value = focus_dict
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return f"{{focus.{path}}}"

        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2, default=str)
        return str(value)

    def render_prompt(self, template: str, daemon_name: str | None = None) -> str:
        """
        Render a prompt template with file-backed variables.

        Uses {namespace.path} syntax for variable substitution:
            {body.mood.confidence} - body state
            {streams.situation} - stream contents
            {daemon.reward_weight} - daemon profile
            {focus.description} - current focus

        Unresolved variables are left as-is (for debugging).
        """
        import re

        def replace_var(match: re.Match) -> str:
            var_path = match.group(1)
            return self._resolve_template_var(var_path, daemon_name)

        # Match {word.path.to.value} but not {{ escaped }}
        pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_-]+)+)\}'
        return re.sub(pattern, replace_var, template)

    # =========================================================================
    # Daemon Lifecycle Control
    # =========================================================================

    async def start_daemon(self, name: str) -> bool:
        """Start a daemon (make it eligible to run)."""
        if name not in self._pipelines:
            self._log(name, LogLevel.ERROR, f"Pipeline not found: {name}")
            return False

        daemon = self._daemons.get(name)
        if not daemon:
            daemon = Daemon(name=name)
            self._daemons[name] = daemon

        if daemon.state == DaemonState.RUNNING:
            return True  # Already running

        if daemon.state == DaemonState.DISABLED:
            self._log(name, LogLevel.WARN, "Cannot start disabled daemon")
            return False

        # Transition to starting
        daemon.state = DaemonState.STARTING
        self._pid_counter += 1
        daemon.pid = self._pid_counter
        daemon.started_at = datetime.now(UTC)
        daemon.stopped_at = None
        daemon.last_error = None

        # Check dependencies are running
        pipeline = self._pipelines[name]
        for dep in pipeline.depends_on:
            dep_daemon = self._daemons.get(dep)
            if not dep_daemon or dep_daemon.state != DaemonState.RUNNING:
                self._log(name, LogLevel.WARN, f"Dependency not running: {dep}")
                # Start dependency first
                await self.start_daemon(dep)

        daemon.state = DaemonState.RUNNING
        self._save_daemon_state(daemon)
        self._log(name, LogLevel.INFO, f"Started daemon (pid={daemon.pid})")
        return True

    async def stop_daemon(self, name: str) -> bool:
        """Stop a daemon."""
        daemon = self._daemons.get(name)
        if not daemon:
            return False

        if daemon.state in (DaemonState.STOPPED, DaemonState.DISABLED):
            return True  # Already stopped

        daemon.state = DaemonState.STOPPING
        self._log(name, LogLevel.INFO, "Stopping daemon...")

        # Stop any dependents first
        pipeline = self._pipelines.get(name)
        if pipeline:
            for other_name, other_pipe in self._pipelines.items():
                if name in other_pipe.depends_on:
                    await self.stop_daemon(other_name)

        daemon.state = DaemonState.STOPPED
        daemon.stopped_at = datetime.now(UTC)
        self._save_daemon_state(daemon)
        self._log(name, LogLevel.INFO, "Daemon stopped")
        return True

    async def restart_daemon(self, name: str) -> bool:
        """Restart a daemon."""
        daemon = self._daemons.get(name)
        if daemon:
            daemon.restarts += 1

        await self.stop_daemon(name)
        await asyncio.sleep(0.1)  # Brief pause
        return await self.start_daemon(name)

    async def reload_daemon(self, name: str) -> bool:
        """Reload daemon configuration without full restart."""
        if name not in self._pipelines:
            return False

        # Re-read pipeline from disk
        pipeline = self.unconscious.get_pipeline(name)
        if pipeline:
            self._pipelines[name] = pipeline
            self._log(name, LogLevel.INFO, "Reloaded configuration")
            return True
        return False

    def enable_daemon(self, name: str) -> bool:
        """Enable daemon for boot."""
        daemon = self._daemons.get(name)
        if not daemon:
            daemon = Daemon(name=name)
            self._daemons[name] = daemon

        daemon.enabled_on_boot = True
        if daemon.state == DaemonState.DISABLED:
            daemon.state = DaemonState.STOPPED
        self._save_daemon_state(daemon)
        self._log(name, LogLevel.INFO, "Enabled daemon")
        return True

    def disable_daemon(self, name: str) -> bool:
        """Disable daemon (won't start on boot)."""
        daemon = self._daemons.get(name)
        if not daemon:
            return False

        daemon.enabled_on_boot = False
        daemon.state = DaemonState.DISABLED
        self._save_daemon_state(daemon)
        self._log(name, LogLevel.INFO, "Disabled daemon")
        return True

    def daemon_status(self, name: str) -> dict[str, Any] | None:
        """Get detailed status for a daemon."""
        daemon = self._daemons.get(name)
        pipeline = self._pipelines.get(name)
        if not daemon:
            return None

        return {
            "name": name,
            "state": daemon.state.value,
            "pid": daemon.pid,
            "started_at": daemon.started_at.isoformat() if daemon.started_at else None,
            "restarts": daemon.restarts,
            "last_error": daemon.last_error,
            "enabled_on_boot": daemon.enabled_on_boot,
            "group": pipeline.group if pipeline else "default",
            "priority": pipeline.priority if pipeline else 5,
            "runlevel": pipeline.runlevel.value if pipeline else "normal",
        }

    def list_daemons(self) -> list[dict[str, Any]]:
        """List all daemons with their status."""
        result = []
        for name in self._pipelines:
            status = self.daemon_status(name)
            if status:
                result.append(status)
        return sorted(result, key=lambda d: (d.get("priority", 5), d.get("name", "")))

    def _save_daemon_state(self, daemon: Daemon):
        """Persist daemon state."""
        self._daemons[daemon.name] = daemon
        self.unconscious.save_daemons(self._daemons)

    # =========================================================================
    # Boot / Shutdown (Init System)
    # =========================================================================

    async def boot(self, runlevel: Runlevel = Runlevel.NORMAL):
        """
        Boot the daemon subsystem to specified runlevel.

        Starts daemons in boot_order, respecting runlevel requirements.
        """
        if self._booted and runlevel == self._runlevel:
            return

        self._log("system", LogLevel.INFO, f"Booting to runlevel {runlevel.value}")
        self._runlevel = runlevel

        if runlevel == Runlevel.HALT:
            await self.shutdown()
            return

        # Get runlevel ordering (halt < minimal < normal < full)
        runlevel_order = [Runlevel.HALT, Runlevel.MINIMAL, Runlevel.NORMAL, Runlevel.FULL]
        current_level_idx = runlevel_order.index(runlevel)

        # Sort pipelines by boot order
        pipelines_to_start = []
        for name, pipeline in self._pipelines.items():
            daemon = self._daemons.get(name)
            if not daemon or not daemon.enabled_on_boot:
                continue

            # Check if pipeline's runlevel is at or below current
            pipe_level_idx = runlevel_order.index(pipeline.runlevel)
            if pipe_level_idx <= current_level_idx:
                pipelines_to_start.append((pipeline.boot_order, name))

        # Start in boot order
        for _, name in sorted(pipelines_to_start):
            await self.start_daemon(name)

        self._booted = True
        self._log("system", LogLevel.INFO, f"Boot complete, {len(pipelines_to_start)} daemons started")

    async def shutdown(self):
        """Shutdown all daemons in reverse boot order."""
        self._log("system", LogLevel.INFO, "Shutting down...")

        # Sort by reverse boot order
        pipelines_to_stop = [
            (self._pipelines[name].boot_order, name)
            for name in self._daemons
            if name in self._pipelines
        ]

        for _, name in sorted(pipelines_to_stop, reverse=True):
            await self.stop_daemon(name)

        self._runlevel = Runlevel.HALT
        self._booted = False
        self._log("system", LogLevel.INFO, "Shutdown complete")

    async def change_runlevel(self, runlevel: Runlevel):
        """Change to a different runlevel, starting/stopping daemons as needed."""
        if runlevel == self._runlevel:
            return

        self._log("system", LogLevel.INFO, f"Changing runlevel: {self._runlevel.value} -> {runlevel.value}")

        runlevel_order = [Runlevel.HALT, Runlevel.MINIMAL, Runlevel.NORMAL, Runlevel.FULL]
        old_idx = runlevel_order.index(self._runlevel)
        new_idx = runlevel_order.index(runlevel)

        if new_idx > old_idx:
            # Going up - start more daemons
            await self.boot(runlevel)
        else:
            # Going down - stop some daemons
            for name, pipeline in self._pipelines.items():
                pipe_idx = runlevel_order.index(pipeline.runlevel)
                if pipe_idx > new_idx:
                    await self.stop_daemon(name)

        self._runlevel = runlevel

    # =========================================================================
    # Dependency Resolution
    # =========================================================================

    def _resolve_run_order(self, pipelines: list[str]) -> list[str]:
        """
        Topological sort of pipelines based on dependencies.

        Returns pipelines in order such that dependencies come first.
        """
        # Build dependency graph
        graph: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = defaultdict(int)

        for name in pipelines:
            pipeline = self._pipelines.get(name)
            if not pipeline:
                continue

            in_degree.setdefault(name, 0)
            for dep in pipeline.depends_on:
                if dep in self._pipelines:
                    graph[dep].append(name)
                    in_degree[name] += 1

        # Kahn's algorithm
        queue = [n for n in pipelines if in_degree[n] == 0]
        result = []

        while queue:
            # Sort by priority within the queue
            queue.sort(key=lambda n: self._pipelines.get(n, Pipeline(name=n, prompt="", output="")).priority)
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(pipelines):
            self._log("system", LogLevel.WARN, "Circular dependency detected in pipelines")
            # Return what we can
            return result + [n for n in pipelines if n not in result]

        return result

    # =========================================================================
    # Resource Budget Management (Focus-Based Dynamic Allocation)
    # =========================================================================

    def _get_base_budget(self, pipeline: Pipeline) -> TokenBudget:
        """Get base budget for a pipeline (pipeline > group > default)."""
        # Pipeline-specific budget
        if pipeline.budget:
            return pipeline.budget

        # Group budget
        group = self._groups.get(pipeline.group)
        if group:
            # Use group's runtime budget
            budget_name = f"group:{group.name}"
            if budget_name not in self._budgets:
                self._budgets[budget_name] = group.budget.model_copy()
            return self._budgets[budget_name]

        # Default
        if "default" not in self._budgets:
            self._budgets["default"] = TokenBudget()
        return self._budgets["default"]

    def _compute_focus_similarity(self, daemon_name: str) -> float:
        """
        Compute similarity between daemon's semantic profile and current focus.

        Returns a value between 0 and 1, where:
            1.0 = perfect alignment with focus
            0.0 = no alignment (or missing embeddings)
        """
        focus = self.unconscious.load_focus()
        if not focus.embedding:
            return 1.0  # No focus embedding - treat all daemons equally

        profile = self.unconscious.get_profile(daemon_name)
        if not profile.embedding:
            return 0.5  # No profile embedding - use neutral similarity

        similarity = cosine_similarity(focus.embedding, profile.embedding)
        # Cosine similarity is [-1, 1], normalize to [0, 1]
        return (similarity + 1) / 2

    def _get_effective_budget(self, pipeline: Pipeline) -> TokenBudget:
        """
        Get effective budget for a pipeline, scaled by focus similarity.

        Daemons semantically aligned with the current focus get more budget.
        This implements "energy routing" - daemons "light up" when the agent
        is focused on their domain.
        """
        base = self._get_base_budget(pipeline)

        # Compute focus similarity scaling
        similarity = self._compute_focus_similarity(pipeline.name)

        # Scale budget by similarity, with floor at 10% to ensure daemons can still run
        scale = max(0.1, similarity)

        # Create scaled budget (don't modify the base)
        return TokenBudget(
            max_tokens_per_run=int(base.max_tokens_per_run * scale),
            max_tokens_per_step=int(base.max_tokens_per_step * scale),
            max_tokens_per_hour=int(base.max_tokens_per_hour * scale),
            used_this_step=base.used_this_step,
            used_this_hour=base.used_this_hour,
            hour_started_at=base.hour_started_at,
        )

    def _check_budget(self, pipeline: Pipeline) -> bool:
        """Check if pipeline has budget to run."""
        budget = self._get_effective_budget(pipeline)
        budget.maybe_reset_hour()
        return budget.can_run(pipeline.max_tokens)

    def _deduct_tokens(self, pipeline: Pipeline, tokens: int):
        """Deduct tokens from pipeline's budget."""
        budget = self._get_effective_budget(pipeline)
        budget.deduct(tokens)
        # Persist
        self.unconscious.save_budgets(self._budgets)

    def _check_group_concurrency(self, pipeline: Pipeline) -> bool:
        """Check if group allows another concurrent run."""
        group = self._groups.get(pipeline.group)
        if not group:
            return True
        return group.running_count < group.max_concurrent

    # =========================================================================
    # Reward Attribution (Semantic Credit Assignment)
    # =========================================================================

    def attribute_reward(
        self,
        reward: float,
        task_description: str,
        source: str = "unknown"
    ) -> dict[str, float]:
        """
        Distribute reward to daemons by semantic relevance.

        NOTE: The source of reward signals needs careful design to align with
        the agent's intentional architecture. This method provides the mechanism;
        the calling context provides the meaning.

        Args:
            reward: Scalar reward value (-1 to 1 typically)
            task_description: Natural language description of what was rewarded
            source: Where this reward came from (for logging/analysis)

        Returns:
            Dict mapping daemon names to their weighted reward amounts
        """
        # Compute task embedding
        try:
            model = _get_sentence_model()
            task_embedding = model.encode(task_description, convert_to_numpy=True).tolist()
        except Exception as e:
            self._log("system", LogLevel.WARN, f"Failed to embed task for reward: {e}")
            return {}

        attribution = {}

        for daemon_name in self._daemons:
            profile = self.unconscious.get_profile(daemon_name)
            if not profile.embedding:
                continue

            # Compute semantic similarity between task and daemon profile
            similarity = cosine_similarity(task_embedding, profile.embedding)
            # Normalize similarity to [0, 1] for weight
            weight = (similarity + 1) / 2

            # Compute weighted reward
            weighted_reward = reward * weight

            # Update profile
            profile.total_reward += reward
            profile.weighted_reward += weighted_reward
            profile.reward_weight_history.append(weight)
            # Keep bounded history
            if len(profile.reward_weight_history) > 100:
                profile.reward_weight_history = profile.reward_weight_history[-100:]

            self.unconscious.save_profile(profile)
            attribution[daemon_name] = weighted_reward

            # Log for training data extraction and introspection
            self._log(
                daemon_name,
                LogLevel.INFO,
                f"Reward ({source}): {reward:.3f} * {weight:.3f} = {weighted_reward:.3f}",
                source=source,
                task=task_description[:200],
                weight=weight,
                raw_reward=reward,
                weighted_reward=weighted_reward,
            )

        return attribution

    def update_focus(self, description: str) -> Focus:
        """
        Update the focus description and recompute its embedding.

        This is called when the agent wants to change what it's focusing on,
        which in turn affects daemon energy allocation.
        """
        focus = self.unconscious.load_focus()
        focus.description = description
        focus.changed_at = datetime.now(UTC)

        # Compute embedding
        try:
            model = _get_sentence_model()
            focus.embedding = model.encode(description, convert_to_numpy=True).tolist()
            focus.embedding_updated = datetime.now(UTC)
        except Exception as e:
            self._log("system", LogLevel.WARN, f"Failed to embed focus: {e}")

        self.unconscious.save_focus(focus)
        self._log("system", LogLevel.INFO, f"Focus updated: {description[:100]}")
        return focus

    def _maybe_apply_focus_update(
        self,
        daemon_name: str,
        new_focus_description: str,
        stability_window_minutes: float = 5.0,
    ) -> bool:
        """
        Apply focus update from a daemon, respecting manual changes.

        The focus-updater daemon should not override recent manual changes.
        If the agent directly updated focus within the stability window,
        we skip the automatic update.

        Args:
            daemon_name: Name of the daemon requesting the update
            new_focus_description: The new focus description
            stability_window_minutes: How long to respect manual changes

        Returns:
            True if focus was updated, False if skipped
        """
        focus = self.unconscious.load_focus()

        # Check if focus was recently changed manually
        minutes_since_change = (datetime.now(UTC) - focus.changed_at).total_seconds() / 60
        if minutes_since_change < stability_window_minutes:
            # Check if the change was from another daemon or manual
            # We consider any recent change as potentially manual
            self._log(
                daemon_name,
                LogLevel.DEBUG,
                f"Skipping auto-update, focus changed {minutes_since_change:.1f}m ago"
            )
            return False

        # Clean up the response (take first line, strip whitespace)
        new_description = new_focus_description.strip().split('\n')[0].strip()

        # Don't update if empty or unchanged
        if not new_description or new_description == focus.description:
            return False

        # Apply the update
        self.update_focus(new_description)
        self._log(
            daemon_name,
            LogLevel.INFO,
            f"Auto-updated focus: {new_description[:100]}"
        )
        return True

    def _increment_group_running(self, pipeline: Pipeline):
        """Increment running count for pipeline's group."""
        group = self._groups.get(pipeline.group)
        if group:
            group.running_count += 1

    def _decrement_group_running(self, pipeline: Pipeline):
        """Decrement running count for pipeline's group."""
        group = self._groups.get(pipeline.group)
        if group and group.running_count > 0:
            group.running_count -= 1

    # =========================================================================
    # Source Reading
    # =========================================================================

    def _expand_path(self, template: str) -> str:
        """Expand a path template."""
        return self.unconscious.expand_path(template, self.body_root, self.cwd)

    def _read_source(self, source: PipelineSource) -> str | None:
        """Read content from a source."""
        path_str = self._expand_path(source.path)

        # Handle glob patterns
        if '*' in path_str:
            paths = list(Path('/').glob(path_str.lstrip('/')))
            if not paths:
                return None if not source.required else ""
            # Concatenate multiple files
            contents = []
            for p in sorted(paths)[-10:]:  # Limit to last 10 matches
                try:
                    contents.append(f"--- {p} ---\n{p.read_text()[:5000]}")
                except Exception:
                    pass
            return "\n\n".join(contents)

        path = Path(path_str)
        if not path.exists():
            return None if not source.required else ""

        try:
            content = path.read_text()

            if source.mode == SourceMode.FULL:
                return content[:50000]  # Limit full reads
            elif source.mode == SourceMode.TAIL:
                lines = content.split('\n')
                return '\n'.join(lines[-source.lines:])
            elif source.mode == SourceMode.HEAD:
                lines = content.split('\n')
                return '\n'.join(lines[:source.lines])
            elif source.mode == SourceMode.DIFF:
                # For diff mode, we'd need to track previous content
                # For now, just return tail
                lines = content.split('\n')
                return '\n'.join(lines[-source.lines:])

            return content
        except Exception:
            return None if not source.required else ""

    def _hash_content(self, content: str) -> str:
        """Hash content for change detection."""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _sources_changed(self, pipeline: Pipeline) -> bool:
        """Check if any sources have changed since last run."""
        current_hashes = {}
        for source in pipeline.get_sources():
            content = self._read_source(source)
            if content is not None:
                current_hashes[source.path] = self._hash_content(content)

        previous_hashes = self._source_hashes.get(pipeline.name, {})
        changed = current_hashes != previous_hashes

        if changed:
            self._source_hashes[pipeline.name] = current_hashes

        return changed

    # =========================================================================
    # Pipeline Execution
    # =========================================================================

    async def _run_pipeline(self, pipeline: Pipeline) -> PipelineRun:
        """
        Execute a single pipeline with full daemon integration.

        Handles:
        - Group concurrency tracking
        - Token budget deduction
        - Journal logging
        - Daemon failure tracking
        - Triggering feeds_into pipelines
        """
        run = PipelineRun(
            pipeline_name=pipeline.name,
            status=PipelineRunStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

        # Track group concurrency
        self._increment_group_running(pipeline)
        self._log(pipeline.name, LogLevel.DEBUG, "Starting pipeline run")

        try:
            # Gather source content
            source_contents = []
            for source in pipeline.get_sources():
                content = self._read_source(source)
                if content is None and source.required:
                    run.status = PipelineRunStatus.SKIPPED
                    run.error = f"Required source missing: {source.path}"
                    self._log(pipeline.name, LogLevel.WARN, f"Skipped: {run.error}")
                    return run
                if content:
                    expanded_path = self._expand_path(source.path)
                    source_contents.append(f"### Source: {expanded_path}\n\n{content}")
                    run.source_hashes[source.path] = self._hash_content(content)

            if not source_contents:
                run.status = PipelineRunStatus.SKIPPED
                run.error = "No source content"
                self._log(pipeline.name, LogLevel.WARN, "Skipped: no source content")
                return run

            combined_sources = "\n\n---\n\n".join(source_contents)

            # Render the prompt template with file-backed variables
            rendered_prompt = self.render_prompt(pipeline.prompt, daemon_name=pipeline.name)

            # Build the full prompt
            full_prompt = f"""{rendered_prompt}

---

{combined_sources}"""

            # Call LLM
            if self._llm_caller:
                response_parts = []
                async for chunk in self._llm_caller(
                    pipeline.model,
                    full_prompt,
                    pipeline.max_tokens,
                    pipeline.temperature,
                ):
                    response_parts.append(chunk)
                response = "".join(response_parts)

                # Estimate tokens used (rough: ~4 chars per token)
                tokens_used = (len(full_prompt) + len(response)) // 4
                run.tokens_used = tokens_used
                self._deduct_tokens(pipeline, tokens_used)
            else:
                # No LLM caller - write a placeholder
                response = f"[Pipeline {pipeline.name} - LLM not configured]\n\nSources:\n{combined_sources[:500]}..."
                run.tokens_used = 0

            # Write primary output
            self.unconscious.write_stream(pipeline.output, response)
            run.output_path = pipeline.output

            # Write additional outputs if defined
            for additional_output in pipeline.outputs:
                if additional_output.prompt_append:
                    additional_prompt = f"{pipeline.prompt}\n\n{additional_output.prompt_append}"
                    if self._llm_caller:
                        add_response_parts = []
                        async for chunk in self._llm_caller(
                            pipeline.model,
                            f"{additional_prompt}\n\n---\n\n{combined_sources}",
                            pipeline.max_tokens // 2,
                            pipeline.temperature,
                        ):
                            add_response_parts.append(chunk)
                        add_response = "".join(add_response_parts)
                        # Track additional tokens
                        add_tokens = len(add_response) // 4
                        run.tokens_used += add_tokens
                        self._deduct_tokens(pipeline, add_tokens)
                    else:
                        add_response = f"[Additional output - LLM not configured]"
                    self.unconscious.write_stream(additional_output.path, add_response)

            run.status = PipelineRunStatus.COMPLETED
            run.completed_at = datetime.now(UTC)
            run.duration_ms = int((run.completed_at - run.started_at).total_seconds() * 1000)

            # Reset daemon failure counter on success
            daemon = self._daemons.get(pipeline.name)
            if daemon:
                daemon.consecutive_failures = 0
                self._save_daemon_state(daemon)

            # Update daemon profile with activity
            profile = self.unconscious.get_profile(pipeline.name)
            profile.add_activity(rendered_prompt, response)
            self.unconscious.save_profile(profile)

            # Update embedding periodically (every 5 runs to save compute)
            if len(profile.recent_outputs) % 5 == 0:
                self.unconscious.update_profile_embedding(pipeline.name)

            self._log(
                pipeline.name, LogLevel.INFO,
                f"Completed in {run.duration_ms}ms, {run.tokens_used} tokens",
                duration_ms=run.duration_ms,
                tokens=run.tokens_used,
            )

            # Handle auto_apply_focus (for focus-updater daemon)
            if pipeline.auto_apply_focus:
                self._maybe_apply_focus_update(pipeline.name, response)

        except Exception as e:
            run.status = PipelineRunStatus.FAILED
            run.error = str(e)
            run.completed_at = datetime.now(UTC)

            # Track daemon failure
            daemon = self._daemons.get(pipeline.name)
            if daemon:
                daemon.consecutive_failures += 1
                daemon.last_error = str(e)

                # Check if we should mark daemon as failed
                if daemon.consecutive_failures >= 3:
                    daemon.state = DaemonState.FAILED
                    self._log(
                        pipeline.name, LogLevel.ERROR,
                        f"Daemon marked FAILED after {daemon.consecutive_failures} consecutive failures",
                    )

                self._save_daemon_state(daemon)

            self._log(pipeline.name, LogLevel.ERROR, f"Failed: {e}")

        finally:
            # Always decrement group running count
            self._decrement_group_running(pipeline)

        self._last_runs[pipeline.name] = datetime.now(UTC)

        # Trigger feeds_into pipelines if successful
        if run.status == PipelineRunStatus.COMPLETED and pipeline.feeds_into:
            for downstream_name in pipeline.feeds_into:
                downstream = self._pipelines.get(downstream_name)
                if downstream and self._should_run(downstream):
                    self._log(pipeline.name, LogLevel.DEBUG, f"Triggering downstream: {downstream_name}")
                    await self._run_pipeline(downstream)

        return run

    def _should_run(self, pipeline: Pipeline) -> bool:
        """
        Determine if a pipeline should run now.

        Checks (in order):
        1. Pipeline enabled
        2. Daemon state is RUNNING
        3. Token budget available
        4. Group concurrency limit not exceeded
        5. Dependencies have run (if any)
        6. Trigger conditions met
        """
        # Check pipeline enabled
        if not pipeline.enabled:
            return False

        # Check daemon state
        daemon = self._daemons.get(pipeline.name)
        if not daemon or daemon.state != DaemonState.RUNNING:
            return False

        # Check token budget
        if not self._check_budget(pipeline):
            self._log(pipeline.name, LogLevel.DEBUG, "Budget exhausted, skipping")
            return False

        # Check group concurrency
        if not self._check_group_concurrency(pipeline):
            return False

        # Check dependencies have outputs available
        for dep_name in pipeline.depends_on:
            dep_pipeline = self._pipelines.get(dep_name)
            if dep_pipeline:
                # Check if dependency's output exists
                dep_output = self.unconscious.read_stream(dep_pipeline.output)
                if not dep_output:
                    return False

        trigger = pipeline.trigger
        now = datetime.now(UTC)

        # Check cooldown
        last_run = self._last_runs.get(pipeline.name)
        if last_run:
            elapsed_ms = (now - last_run).total_seconds() * 1000
            if elapsed_ms < trigger.cooldown_ms:
                return False
            if elapsed_ms < trigger.debounce_ms:
                return False

        if trigger.mode == TriggerMode.ON_CHANGE:
            return self._sources_changed(pipeline)

        elif trigger.mode == TriggerMode.EVERY_STEP:
            return True

        elif trigger.mode == TriggerMode.EVERY_N_STEPS:
            return self._step_count % trigger.n_steps == 0

        elif trigger.mode == TriggerMode.ON_IDLE:
            # Run if nothing else is running
            return True  # Simplified for now

        elif trigger.mode == TriggerMode.ON_DEMAND:
            return False  # Only run when explicitly requested

        elif trigger.mode == TriggerMode.CONTINUOUS:
            return True

        return False

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def step(self):
        """
        Process one step of the unconscious daemon manager.

        Steps:
        1. Increment step counter
        2. Reset step-level budgets
        3. Resolve pipeline run order (topological sort)
        4. Run eligible pipelines
        5. Save status
        """
        self._step_count += 1
        self._status.step_count = self._step_count
        self._status.last_step_at = datetime.now(UTC)

        # Reset step-level budgets
        for budget in self._budgets.values():
            budget.reset_step()

        # Get eligible pipelines and resolve run order
        eligible = [
            name for name, pipeline in self._pipelines.items()
            if self._should_run(pipeline)
        ]
        run_order = self._resolve_run_order(eligible)

        # Run pipelines in dependency order
        for name in run_order:
            pipeline = self._pipelines.get(name)
            if not pipeline:
                continue

            # Re-check eligibility (might have changed due to budget/concurrency)
            if not self._should_run(pipeline):
                continue

            run = await self._run_pipeline(pipeline)
            self._status.recent_runs.append(run)

            # Keep only recent runs
            if len(self._status.recent_runs) > 50:
                self._status.recent_runs = self._status.recent_runs[-50:]

            if run.status == PipelineRunStatus.FAILED:
                self._status.errors.append(f"{pipeline.name}: {run.error}")

        self.unconscious.save_status(self._status)

    async def run_pipeline_now(self, name: str) -> PipelineRun | None:
        """Run a specific pipeline immediately (on-demand), bypassing trigger checks."""
        pipeline = self._pipelines.get(name) or self.unconscious.get_pipeline(name)
        if not pipeline:
            self._log(name, LogLevel.ERROR, "Pipeline not found")
            return None

        # Ensure daemon exists and is startable
        daemon = self._daemons.get(name)
        if not daemon:
            daemon = Daemon(name=name)
            self._daemons[name] = daemon

        # Temporarily mark as running if not already
        original_state = daemon.state
        if daemon.state != DaemonState.RUNNING:
            daemon.state = DaemonState.RUNNING

        try:
            run = await self._run_pipeline(pipeline)
            return run
        finally:
            # Restore original state if it wasn't already running
            if original_state != DaemonState.RUNNING:
                daemon.state = original_state

    async def run_loop(self, step_event: asyncio.Event, stop_event: asyncio.Event):
        """
        Main unconscious daemon loop.

        Lifecycle:
        1. Load pipelines and daemon states
        2. Boot to NORMAL runlevel
        3. Process steps on events or idle timeout
        4. Shutdown on stop signal
        """
        self._running = True
        self._status.running = True
        self.load_pipelines()

        # Boot the daemon subsystem
        await self.boot(Runlevel.NORMAL)

        try:
            while not stop_event.is_set():
                try:
                    # Wait for step signal or timeout (for idle processing)
                    await asyncio.wait_for(step_event.wait(), timeout=5.0)
                    step_event.clear()
                except asyncio.TimeoutError:
                    # Idle - run on_idle pipelines
                    pass

                await self.step()

        finally:
            # Shutdown daemons gracefully
            await self.shutdown()
            self._running = False
            self._status.running = False
            self.unconscious.save_status(self._status)

    def get_status(self) -> UnconsciousStatus:
        """Get current status."""
        return self._status

    def get_runlevel(self) -> Runlevel:
        """Get current runlevel."""
        return self._runlevel

    def get_groups(self) -> dict[str, ProcessGroup]:
        """Get all process groups."""
        return self._groups.copy()

    def get_budget_status(self) -> dict[str, dict]:
        """Get budget status for all tracked budgets."""
        return {
            name: {
                "used_this_step": b.used_this_step,
                "used_this_hour": b.used_this_hour,
                "max_per_step": b.max_tokens_per_step,
                "max_per_hour": b.max_tokens_per_hour,
            }
            for name, b in self._budgets.items()
        }


# =============================================================================
# Default Pipelines
# =============================================================================

DEFAULT_PIPELINES = [
    # Critical perception - runs at MINIMAL runlevel
    Pipeline(
        name="danger-assessment",
        description="Assess potential dangers or risks",
        sources=[
            PipelineSource(path="{body}/embodiment.json"),
            PipelineSource(path="{body}/working_set.json"),
        ],
        trigger=PipelineTrigger(mode=TriggerMode.EVERY_N_STEPS, n_steps=5),
        model="haiku",
        prompt="""Assess any potential dangers or risks in the current situation.
Consider: resource limits, error patterns, permission issues, wrong environment.
Respond with threat level (none/low/medium/high/critical) and brief explanation.
If none, just say "none".""",
        output="danger-assessment.md",
        priority=1,
        group="perception",
        runlevel=Runlevel.MINIMAL,
        boot_order=10,
    ),

    # Standard perception - runs at NORMAL runlevel
    Pipeline(
        name="situation-summary",
        description="High-level summary of current situation",
        sources=[
            PipelineSource(path="{body}/working_set.json"),
            PipelineSource(path="{body}/embodiment.json"),
        ],
        trigger=PipelineTrigger(mode=TriggerMode.EVERY_STEP),
        model="haiku",
        prompt="""Summarize the agent's current situation in 2-3 sentences.
Include: current goals, location, and any notable state.
Be extremely concise.""",
        output="situation.md",
        priority=1,
        group="perception",
        runlevel=Runlevel.NORMAL,
        boot_order=20,
    ),

    # Abstraction - runs at NORMAL runlevel, depends on situation-summary
    Pipeline(
        name="next-step-prediction",
        description="Predict what the agent will likely do next",
        sources=[
            PipelineSource(path="{body}/working_set.json"),
            PipelineSource(path="{streams}/situation.md", required=False),
        ],
        trigger=PipelineTrigger(mode=TriggerMode.EVERY_N_STEPS, n_steps=3),
        model="haiku",
        prompt="""Based on current goals and situation, predict:
1. What will the agent probably do next?
2. What should it do next?
3. Where might it get stuck?
Be concise - 3 bullet points max.""",
        output="next-prediction.md",
        priority=3,
        group="abstraction",
        runlevel=Runlevel.NORMAL,
        boot_order=30,
        depends_on=["situation-summary"],
    ),

    # Meta-cognition - runs at FULL runlevel
    Pipeline(
        name="self-reflection",
        description="Reflect on recent behavior and patterns",
        sources=[
            PipelineSource(path="{body}/memory/episodes/*.md", required=False),
            PipelineSource(path="{body}/character.json"),
        ],
        trigger=PipelineTrigger(mode=TriggerMode.EVERY_N_STEPS, n_steps=10),
        model="haiku",
        prompt="""Briefly reflect on the agent's recent behavior.
Are there any patterns? Blind spots? Improvements to suggest?
One paragraph max.""",
        output="self-reflection.md",
        priority=7,
        group="maintenance",
        runlevel=Runlevel.FULL,
        boot_order=50,
    ),

    # Focus updater - automatically updates focus based on situation
    Pipeline(
        name="focus-updater",
        description="Automatically update the agent's focus based on current situation",
        sources=[
            PipelineSource(path="{body}/working_set.json"),
            PipelineSource(path="{body}/perception/focus.json"),
            PipelineSource(path="{streams}/situation.md", required=False),
            PipelineSource(path="{streams}/danger-assessment.md", required=False),
        ],
        trigger=PipelineTrigger(mode=TriggerMode.EVERY_N_STEPS, n_steps=5),
        model="haiku",
        prompt="""Review the agent's current situation and determine the appropriate focus.

Current focus: {focus.description}
Last changed: {focus.changed_at}

Consider: current goals, recent decisions, open loops, danger level, and overall situation.

Output ONLY a single sentence describing what the agent should focus on.
If the current focus is still appropriate, output it unchanged.
Do not explain - just output the focus sentence.""",
        output="focus-update.md",
        priority=6,
        group="abstraction",
        runlevel=Runlevel.NORMAL,
        boot_order=60,
        auto_apply_focus=True,
    ),
]


# Default process groups with their budgets
DEFAULT_GROUPS = {
    "perception": ProcessGroup(
        name="perception",
        description="Raw sensory processing and danger detection",
        priority=1,
        budget=TokenBudget(
            max_tokens_per_run=800,
            max_tokens_per_step=2000,
            max_tokens_per_hour=40000,
        ),
        max_concurrent=2,
    ),
    "abstraction": ProcessGroup(
        name="abstraction",
        description="Higher-level pattern recognition and prediction",
        priority=3,
        budget=TokenBudget(
            max_tokens_per_run=1000,
            max_tokens_per_step=3000,
            max_tokens_per_hour=50000,
        ),
        max_concurrent=2,
    ),
    "maintenance": ProcessGroup(
        name="maintenance",
        description="Self-monitoring, cleanup, and reflection",
        priority=7,
        budget=TokenBudget(
            max_tokens_per_run=600,
            max_tokens_per_step=1000,
            max_tokens_per_hour=20000,
        ),
        max_concurrent=1,
    ),
    "default": ProcessGroup(
        name="default",
        description="Default group for unassigned daemons",
        priority=5,
        budget=TokenBudget(
            max_tokens_per_run=800,
            max_tokens_per_step=2000,
            max_tokens_per_hour=30000,
        ),
        max_concurrent=2,
    ),
}


def install_default_pipelines(unconscious_dir: UnconsciousDirectory):
    """Install default pipelines and groups if none exist."""
    existing = unconscious_dir.list_pipelines()
    if not existing:
        for pipeline in DEFAULT_PIPELINES:
            unconscious_dir.save_pipeline(pipeline)

    # Install default groups if none exist
    existing_groups = unconscious_dir.load_groups()
    if not existing_groups:
        unconscious_dir.save_groups(DEFAULT_GROUPS)
