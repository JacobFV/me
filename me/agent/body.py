"""
Body - The agent's body as a virtual filesystem.

The agent's body IS a directory. Every aspect of the agent's state is a file
that can be read, written, edited, listed, and deleted using standard file
operations. No special tools needed.

This module defines:
1. Pydantic models for all body state
2. A BodyDirectory that maps these models to files (using filepydantic)
3. Markdown-based memory with JSON frontmatter

Structure:
    ~/.me/agents/<id>/
    ├── identity.json           # IMMUTABLE - who this agent is
    ├── config.json             # Session settings (refresh rate, model, etc.)
    ├── current -> steps/0005   # Symlink to current step
    ├── memory/
    │   ├── episodes/           # Episode markdown files
    │   ├── procedures/         # Procedure markdown files
    │   ├── significant/        # Significant moment files
    │   ├── intentions/         # Future intention files
    │   └── theories/           # Working theory files
    ├── unconscious/            # Background perception pipelines
    │   ├── pipelines/          # Pipeline definitions (.json)
    │   ├── streams/            # Pipeline outputs (pre-conscious)
    │   └── status.json         # Runner state
    └── steps/
        └── NNNN/               # Step snapshots (each is complete state)
            ├── embodiment.json # Location, capabilities, mood
            ├── character.json  # Values, tendencies, revealed preferences
            ├── working_set.json# Goals, open loops, decisions
            ├── mouth.json      # Where speech goes
            ├── sensors.json    # Sensor definitions
            ├── focus.json      # Current attention focus
            ├── input.txt       # User input for this step
            └── output.txt      # Agent output for this step

Philosophy:
- The filesystem IS the agent's body
- Read/Write/Edit are the only tools needed
- Memory is markdown (human-readable, grep-able)
- State is JSON (machine-parseable, editable)
- Each step is a complete snapshot - enables replay and debugging
- current/ symlink gives access to latest state
- Unconscious pipelines run in background, outputs available in streams/
- Conscious attention is choosing which files to read
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any

import psutil
from pydantic import BaseModel, Field
from filepydantic import FileDirectory, FileModel


# =============================================================================
# System Info (for body awareness)
# =============================================================================

@dataclass
class SystemInfo:
    """Static system information."""
    hostname: str
    platform: str
    platform_version: str
    architecture: str
    python_version: str
    cpu_count: int
    total_memory_gb: float

    @classmethod
    def collect(cls) -> "SystemInfo":
        mem = psutil.virtual_memory()
        return cls(
            hostname=socket.gethostname(),
            platform=platform.system(),
            platform_version=platform.release(),
            architecture=platform.machine(),
            python_version=platform.python_version(),
            cpu_count=os.cpu_count() or 1,
            total_memory_gb=round(mem.total / (1024**3), 2),
        )


@dataclass
class RuntimeInfo:
    """Dynamic runtime information."""
    pid: int
    cwd: str
    user: str
    cpu_percent: float
    memory_percent: float
    uptime_seconds: float
    timestamp: str

    @classmethod
    def collect(cls, start_time: datetime | None = None) -> "RuntimeInfo":
        process = psutil.Process()
        uptime = 0.0
        if start_time:
            uptime = (datetime.now() - start_time).total_seconds()
        return cls(
            pid=os.getpid(),
            cwd=os.getcwd(),
            user=os.getenv("USER", "unknown"),
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            uptime_seconds=uptime,
            timestamp=datetime.now().isoformat(),
        )


# =============================================================================
# Core Body Models (Pydantic)
# =============================================================================

class AgentIdentity(BaseModel):
    """Immutable identity - set at creation, never changes."""
    agent_id: str
    name: str
    created_at: datetime
    parent_id: str | None = None
    generation: int = 0


class ModelProvider(str, Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    LITELLM = "litellm"


class ModelConfig(BaseModel):
    """
    Model configuration - declarative, file-based.

    Edit ~/.me/agents/<id>/model.json to change models:

    ```json
    {
      "provider": "anthropic",
      "model_id": "claude-sonnet-4-20250514",
      "temperature": 0.7,
      "max_tokens": 4096
    }
    ```

    Or for local Ollama:
    ```json
    {
      "provider": "ollama",
      "model_id": "llama3",
      "host": "http://localhost:11434"
    }
    ```
    """
    provider: ModelProvider = ModelProvider.ANTHROPIC
    model_id: str = "claude-sonnet-4-20250514"

    # Common parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float | None = None

    # Provider-specific
    api_key: str | None = None  # Falls back to env vars
    host: str | None = None  # For Ollama
    base_url: str | None = None  # For OpenAI-compatible endpoints

    # Advanced
    stop_sequences: list[str] = Field(default_factory=list)
    extra_params: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Mutable configuration."""
    refresh_rate_ms: int = 100
    permission_mode: str = "acceptEdits"
    max_turns: int | None = None


class SomaticMode(str, Enum):
    PLAN = "plan"
    ACT = "act"
    INTERACT = "interact"


class Mood(BaseModel):
    """Emergent mood - discovered, not decided."""
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    frustration: float = Field(0.0, ge=0.0, le=1.0)
    curiosity: float = Field(0.5, ge=0.0, le=1.0)
    momentum: float = Field(0.0, ge=-1.0, le=1.0)


class Embodiment(BaseModel):
    """Physical state - where the agent is and what it can do."""
    location: str = "~"
    location_hash: str = ""
    capabilities: list[str] = Field(default_factory=list)
    somatic_mode: SomaticMode = SomaticMode.ACT
    mood: Mood = Field(default_factory=Mood)
    max_steps: int | None = None
    current_step: int = 0


class SensorDefinition(BaseModel):
    """A sensor watching something."""
    name: str
    source: str
    type: str = "file"
    refresh_mode: str = "on_demand"
    interval_ms: int | None = None
    tail_lines: int = 50
    enabled: bool = True


class SensorsConfig(BaseModel):
    """All sensor definitions."""
    sensors: dict[str, SensorDefinition] = Field(default_factory=dict)


class MouthMode(str, Enum):
    RESPONSE = "response"
    FILE = "file"
    STREAM = "stream"
    TEE = "tee"


class MouthConfig(BaseModel):
    """Where the agent's speech goes."""
    mode: MouthMode = MouthMode.RESPONSE
    target: str | None = None
    append: bool = True
    can_change: bool = True


class Goal(BaseModel):
    """A goal in the working set."""
    id: str
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str = "active"
    parent_id: str | None = None


class Decision(BaseModel):
    """A recorded decision."""
    id: str
    summary: str
    reasoning: str
    alternatives_considered: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reversible: bool = True


class OpenLoop(BaseModel):
    """Something unfinished."""
    id: str
    description: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: str = ""


class WorkingSet(BaseModel):
    """The agent's current focus."""
    goals: list[Goal] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    open_loops: list[OpenLoop] = Field(default_factory=list)
    last_actions: list[str] = Field(default_factory=list)


class ActionPattern(BaseModel):
    """A behavioral pattern."""
    situation: str
    action_taken: str
    count: int = 1
    last_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Character(BaseModel):
    """Who the agent has become through choices."""
    risk_tolerance: float = Field(0.5, ge=0.0, le=1.0)
    exploration_tendency: float = Field(0.5, ge=0.0, le=1.0)
    thoroughness: float = Field(0.5, ge=0.0, le=1.0)
    persistence: float = Field(0.5, ge=0.0, le=1.0)
    action_patterns: list[ActionPattern] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    total_decisions: int = 0


class Focus(BaseModel):
    """
    What the agent is currently attending to.

    Core focus fields are defined here. The unconscious module extends this
    with semantic routing fields (embedding, etc.) for daemon energy allocation.
    """
    streams: list[str] = Field(default_factory=list)
    raw: list[str] = Field(default_factory=list)
    description: str = ""
    changed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    budget: float = 1.0
    auto_include: list[str] = Field(default_factory=lambda: ["danger-assessment.md", "situation.md"])
    # Semantic fields (optional, populated by unconscious system)
    embedding: list[float] = Field(default_factory=list)
    embedding_updated: datetime | None = None


# =============================================================================
# Memory Models (Markdown with JSON frontmatter)
# =============================================================================

class EmotionalValence(int, Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class Episode(BaseModel):
    """An episode in the agent's life."""
    id: str
    title: str
    goal: str | None = None
    outcome: str = "in_progress"
    valence: EmotionalValence = EmotionalValence.NEUTRAL
    significance: str = ""
    lessons: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    location: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    follows_from: str | None = None
    tags: list[str] = Field(default_factory=list)


class Procedure(BaseModel):
    """A codified procedure."""
    id: str
    name: str
    description: str
    when_to_use: str
    steps: list[str]
    watch_out_for: list[str] = Field(default_factory=list)
    doesnt_work_when: list[str] = Field(default_factory=list)
    learned_from: str | None = None
    times_used: int = 0
    times_succeeded: int = 0
    notes: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SignificantMoment(BaseModel):
    """A moment marked as significant."""
    id: str
    title: str
    what_happened: str
    why_significant: str
    moment_type: str
    insight: str = ""
    tags: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Intention(BaseModel):
    """A future intention."""
    id: str
    reminder: str
    reason: str
    trigger_description: str
    trigger_keywords: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    triggered: bool = False
    triggered_at: datetime | None = None
    resolved: bool = False
    resolved_at: datetime | None = None
    resolution_notes: str = ""


class Theory(BaseModel):
    """A working theory."""
    id: str
    domain: str
    claim: str
    reasoning: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    status: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# =============================================================================
# Markdown Serialization (using JSON frontmatter)
# =============================================================================

def model_to_markdown(model: BaseModel, body_text: str = "") -> str:
    """Convert a Pydantic model to markdown with JSON frontmatter."""
    data = model.model_dump(mode='json')
    json_str = json.dumps(data, indent=2, default=str)
    return f"```json\n{json_str}\n```\n\n{body_text}"


def markdown_to_model(content: str, model_class: type[BaseModel]) -> tuple[BaseModel, str]:
    """Parse markdown with JSON frontmatter into a Pydantic model."""
    match = re.match(r'^```json\n(.*?)\n```\n?(.*)', content, re.DOTALL)
    if not match:
        raise ValueError("No JSON frontmatter found")
    json_str, body = match.groups()
    data = json.loads(json_str)
    return model_class.model_validate(data), body.strip()


def episode_to_markdown(ep: Episode) -> str:
    """Convert episode to markdown."""
    parts = []
    if ep.goal:
        parts.append(f"## Goal\n\n{ep.goal}")
    if ep.significance:
        parts.append(f"## Significance\n\n{ep.significance}")
    if ep.lessons:
        parts.append("## Lessons Learned\n\n" + "\n".join(f"- {l}" for l in ep.lessons))
    if ep.open_questions:
        parts.append("## Open Questions\n\n" + "\n".join(f"- {q}" for q in ep.open_questions))
    return model_to_markdown(ep, "\n\n".join(parts))


def procedure_to_markdown(proc: Procedure) -> str:
    """Convert procedure to markdown."""
    parts = [
        f"## When to Use\n\n{proc.when_to_use}",
        "## Steps\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(proc.steps)),
    ]
    if proc.watch_out_for:
        parts.append("## Watch Out For\n\n" + "\n".join(f"- {w}" for w in proc.watch_out_for))
    if proc.doesnt_work_when:
        parts.append("## Doesn't Work When\n\n" + "\n".join(f"- {d}" for d in proc.doesnt_work_when))
    if proc.notes:
        parts.append(f"## Notes\n\n{proc.notes}")
    return model_to_markdown(proc, "\n\n".join(parts))


def theory_to_markdown(theory: Theory) -> str:
    """Convert theory to markdown."""
    parts = [
        f"## Claim\n\n{theory.claim}",
        f"## Reasoning\n\n{theory.reasoning}",
    ]
    if theory.supporting_evidence:
        parts.append("## Supporting Evidence\n\n" + "\n".join(f"- {e}" for e in theory.supporting_evidence))
    if theory.contradicting_evidence:
        parts.append("## Contradicting Evidence\n\n" + "\n".join(f"- {e}" for e in theory.contradicting_evidence))
    if theory.predictions:
        parts.append("## Predictions\n\n" + "\n".join(f"- {p}" for p in theory.predictions))
    return model_to_markdown(theory, "\n\n".join(parts))


# =============================================================================
# Body Directory Manager (using filepydantic)
# =============================================================================

class BodyDirectory:
    """
    Manages the agent's body as a directory of files.

    The filesystem IS the agent. Read/Write/Edit are all you need.
    Uses filepydantic for reactive JSON file management.

    State files (embodiment, character, working_set, etc.) live in step
    directories. The `current` symlink points to the active step, so
    `current/embodiment.json` is always the current state.
    """

    # Mutable state files that live in each step directory
    STEP_STATE_FILES = [
        ("embodiment", Embodiment),
        ("character", Character),
        ("working_set", WorkingSet),
        ("mouth", MouthConfig),
        ("sensors", SensorsConfig),
        ("focus", Focus),
    ]

    def __init__(self, base_dir: Path, agent_id: str):
        self.base_dir = base_dir
        self.agent_id = agent_id
        self.root = base_dir / "agents" / agent_id
        self._start_time: datetime | None = None
        self._system_info = SystemInfo.collect()

        # Create directory structure
        self._ensure_structure()

        # FileDirectory for top-level immutable files (identity, config)
        self._files = FileDirectory(self.root, create=True)

        # FileDirectory for step-based mutable state (via current symlink)
        self._current_dir = self.root / "current"
        self._step_files: FileDirectory | None = None
        if self._current_dir.exists():
            self._step_files = FileDirectory(self._current_dir, create=False)
            self._register_step_files()

    def _ensure_structure(self):
        """Create the directory structure."""
        for subdir in [
            "",
            "memory",
            "memory/episodes",
            "memory/procedures",
            "memory/significant",
            "memory/intentions",
            "memory/theories",
            "unconscious",
            "unconscious/pipelines",
            "unconscious/streams",
            "steps",
        ]:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    def _register_step_files(self):
        """Register state file models with the step FileDirectory."""
        if not self._step_files:
            return
        for name, model_class in self.STEP_STATE_FILES:
            if not self._step_files.get(name):
                self._step_files.register(name, model_class, default=model_class())

    def _write_step_state(self, step_dir: Path, defaults: dict[str, BaseModel] | None = None):
        """Write all state files to a step directory."""
        defaults = defaults or {}
        for name, model_class in self.STEP_STATE_FILES:
            model = defaults.get(name, model_class())
            path = step_dir / f"{name}.json"
            path.write_text(model.model_dump_json(indent=2))

    def _copy_step_state(self, from_step: Path, to_step: Path):
        """Copy state files from one step to another."""
        for name, _ in self.STEP_STATE_FILES:
            src = from_step / f"{name}.json"
            dst = to_step / f"{name}.json"
            if src.exists():
                dst.write_text(src.read_text())
            else:
                # Write default if source doesn't exist
                model_class = dict(self.STEP_STATE_FILES)[name]
                dst.write_text(model_class().model_dump_json(indent=2))

    # =========================================================================
    # System/Runtime Info
    # =========================================================================

    @property
    def system(self) -> SystemInfo:
        return self._system_info

    @property
    def runtime(self) -> RuntimeInfo:
        return RuntimeInfo.collect(self._start_time)

    def set_start_time(self, time: datetime):
        self._start_time = time

    # =========================================================================
    # Top-level Files (immutable or session-scoped)
    # =========================================================================

    @property
    def identity(self) -> AgentIdentity | None:
        fm = self._files.get("identity")
        return fm.model if fm else None

    @property
    def config(self) -> AgentConfig:
        fm = self._files.get("config")
        return fm.model if fm else AgentConfig()

    @config.setter
    def config(self, value: AgentConfig):
        fm = self._files.get("config")
        if fm:
            fm.model = value

    @property
    def model_config(self) -> ModelConfig:
        """
        Get model configuration.

        The agent reads model settings from model.json, allowing you to
        change models by editing the file:

        ```json
        {
          "provider": "ollama",
          "model_id": "llama3",
          "host": "http://localhost:11434"
        }
        ```
        """
        fm = self._files.get("model")
        return fm.model if fm else ModelConfig()

    @model_config.setter
    def model_config(self, value: ModelConfig):
        fm = self._files.get("model")
        if fm:
            fm.model = value

    # =========================================================================
    # Step-based State Files (mutable, versioned per step)
    # =========================================================================

    @property
    def embodiment(self) -> Embodiment:
        if self._step_files:
            fm = self._step_files.get("embodiment")
            if fm:
                return fm.model
        return Embodiment()

    @embodiment.setter
    def embodiment(self, value: Embodiment):
        if self._step_files:
            fm = self._step_files.get("embodiment")
            if fm:
                fm.model = value

    @property
    def character(self) -> Character:
        if self._step_files:
            fm = self._step_files.get("character")
            if fm:
                return fm.model
        return Character()

    @character.setter
    def character(self, value: Character):
        if self._step_files:
            fm = self._step_files.get("character")
            if fm:
                fm.model = value

    @property
    def sensors(self) -> SensorsConfig:
        if self._step_files:
            fm = self._step_files.get("sensors")
            if fm:
                return fm.model
        return SensorsConfig()

    @sensors.setter
    def sensors(self, value: SensorsConfig):
        if self._step_files:
            fm = self._step_files.get("sensors")
            if fm:
                fm.model = value

    @property
    def mouth(self) -> MouthConfig:
        if self._step_files:
            fm = self._step_files.get("mouth")
            if fm:
                return fm.model
        return MouthConfig()

    @mouth.setter
    def mouth(self, value: MouthConfig):
        if self._step_files:
            fm = self._step_files.get("mouth")
            if fm:
                fm.model = value

    @property
    def working_set(self) -> WorkingSet:
        if self._step_files:
            fm = self._step_files.get("working_set")
            if fm:
                return fm.model
        return WorkingSet()

    @working_set.setter
    def working_set(self, value: WorkingSet):
        if self._step_files:
            fm = self._step_files.get("working_set")
            if fm:
                fm.model = value

    @property
    def focus(self) -> Focus:
        if self._step_files:
            fm = self._step_files.get("focus")
            if fm:
                return fm.model
        return Focus()

    @focus.setter
    def focus(self, value: Focus):
        if self._step_files:
            fm = self._step_files.get("focus")
            if fm:
                fm.model = value

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(
        self,
        name: str = "me",
        parent_id: str | None = None,
        generation: int = 0,
    ) -> AgentIdentity:
        """Initialize a new agent with step 0000."""
        # Check if already initialized
        identity_path = self.root / "identity.json"
        if identity_path.exists():
            fm = FileModel(identity_path, AgentIdentity, frozen=True)
            # Ensure step files are set up
            if not self._step_files and self._current_dir.exists():
                self._step_files = FileDirectory(self._current_dir, create=False)
                self._register_step_files()
            return fm.model

        # Create identity (frozen - cannot change after creation)
        identity = AgentIdentity(
            agent_id=self.agent_id,
            name=name,
            created_at=datetime.now(UTC),
            parent_id=parent_id,
            generation=generation,
        )
        self._files.register("identity", AgentIdentity, default=identity, frozen=True)

        # Register config at top level
        self._files.register("config", AgentConfig, default=AgentConfig())

        # Register model config at top level (declarative model configuration)
        self._files.register("model", ModelConfig, default=ModelConfig())

        # Create step 0000 with initial state
        step_dir = self.root / "steps" / "0000"
        step_dir.mkdir(parents=True, exist_ok=True)
        self._write_step_state(step_dir)

        # Create current symlink
        current = self.root / "current"
        if current.exists() or current.is_symlink():
            current.unlink()
        current.symlink_to("steps/0000")

        # Set up step files
        self._step_files = FileDirectory(self._current_dir, create=False)
        self._register_step_files()

        return identity

    # =========================================================================
    # Memory Files (Markdown)
    # =========================================================================

    def _slugify(self, text: str) -> str:
        slug = re.sub(r'[^a-z0-9]+', '-', text.lower())
        return slug.strip('-')[:50]

    def _timestamp_prefix(self, dt: datetime | None = None) -> str:
        dt = dt or datetime.now(UTC)
        return dt.strftime("%Y-%m-%d-%H%M")

    def save_episode(self, episode: Episode) -> Path:
        slug = self._slugify(episode.title)
        prefix = self._timestamp_prefix(episode.started_at)
        path = self.root / "memory" / "episodes" / f"{prefix}-{slug}.md"
        path.write_text(episode_to_markdown(episode))
        return path

    def save_procedure(self, procedure: Procedure) -> Path:
        slug = self._slugify(procedure.name)
        path = self.root / "memory" / "procedures" / f"{slug}.md"
        path.write_text(procedure_to_markdown(procedure))
        return path

    def save_significant(self, moment: SignificantMoment) -> Path:
        slug = self._slugify(moment.title)
        prefix = self._timestamp_prefix(moment.timestamp)
        path = self.root / "memory" / "significant" / f"{prefix}-{slug}.md"
        body = f"## What Happened\n\n{moment.what_happened}\n\n## Why Significant\n\n{moment.why_significant}"
        if moment.insight:
            body += f"\n\n## Insight\n\n{moment.insight}"
        path.write_text(model_to_markdown(moment, body))
        return path

    def save_intention(self, intention: Intention) -> Path:
        slug = self._slugify(intention.reminder)
        path = self.root / "memory" / "intentions" / f"{slug}.md"
        body = f"## Reminder\n\n{intention.reminder}\n\n## Reason\n\n{intention.reason}\n\n## Trigger\n\n{intention.trigger_description}"
        path.write_text(model_to_markdown(intention, body))
        return path

    def save_theory(self, theory: Theory) -> Path:
        domain_slug = self._slugify(theory.domain)
        claim_slug = self._slugify(theory.claim)
        path = self.root / "memory" / "theories" / f"{domain_slug}-{claim_slug}.md"
        path.write_text(theory_to_markdown(theory))
        return path

    def list_episodes(self) -> list[Path]:
        return sorted((self.root / "memory" / "episodes").glob("*.md"), reverse=True)

    def list_procedures(self) -> list[Path]:
        return sorted((self.root / "memory" / "procedures").glob("*.md"))

    def list_significant(self) -> list[Path]:
        return sorted((self.root / "memory" / "significant").glob("*.md"), reverse=True)

    def list_intentions(self) -> list[Path]:
        return sorted((self.root / "memory" / "intentions").glob("*.md"))

    def list_theories(self) -> list[Path]:
        return sorted((self.root / "memory" / "theories").glob("*.md"))

    # =========================================================================
    # Step History
    # =========================================================================

    def begin_step(self, step_number: int, input_text: str = "") -> Path:
        """
        Begin a new step by creating a step directory with state copied from previous step.

        This creates the step directory, copies all state files from the previous step
        (or uses defaults if this is the first step), updates the current symlink,
        and sets up the step FileDirectory for reading/writing state.
        """
        step_dir = self.root / "steps" / f"{step_number:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Copy state from previous step (if exists) or write defaults
        prev_step_num = step_number - 1
        prev_step_dir = self.root / "steps" / f"{prev_step_num:04d}"
        if prev_step_dir.exists():
            self._copy_step_state(prev_step_dir, step_dir)
        else:
            self._write_step_state(step_dir)

        # Write input
        if input_text:
            (step_dir / "input.txt").write_text(input_text)

        # Update current symlink
        current = self.root / "current"
        if current.exists() or current.is_symlink():
            current.unlink()
        current.symlink_to(f"steps/{step_number:04d}")

        # Set up step files for new step
        self._step_files = FileDirectory(self._current_dir, create=False)
        self._register_step_files()

        return step_dir

    def complete_step(self, output_text: str = ""):
        """
        Complete the current step by writing output.

        State files are already in the step directory (written via properties),
        so this just writes the output text.
        """
        current = self.root / "current"
        if not current.exists():
            return
        step_dir = current.resolve()
        if output_text:
            (step_dir / "output.txt").write_text(output_text)

    def get_step_count(self) -> int:
        steps_dir = self.root / "steps"
        if not steps_dir.exists():
            return 0
        return len([d for d in steps_dir.iterdir() if d.is_dir() and d.name.isdigit()])

    def get_current_step_number(self) -> int:
        """Get the current step number from the symlink."""
        current = self.root / "current"
        if not current.exists() and not current.is_symlink():
            return 0
        target = os.readlink(current)
        # Extract step number from "steps/NNNN"
        step_name = Path(target).name
        try:
            return int(step_name)
        except ValueError:
            return 0

    # =========================================================================
    # Full State
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.model_dump(mode='json') if self.identity else None,
            "config": self.config.model_dump(mode='json'),
            "embodiment": self.embodiment.model_dump(mode='json'),
            "character": self.character.model_dump(mode='json'),
            "sensors": self.sensors.model_dump(mode='json'),
            "mouth": self.mouth.model_dump(mode='json'),
            "working_set": self.working_set.model_dump(mode='json'),
            "focus": self.focus.model_dump(mode='json'),
            "system": asdict(self.system),
            "runtime": asdict(self.runtime),
            "step_count": self.get_step_count(),
            "current_step": self.get_current_step_number(),
            "memory": {
                "episodes": len(self.list_episodes()),
                "procedures": len(self.list_procedures()),
                "significant": len(self.list_significant()),
                "intentions": len(self.list_intentions()),
                "theories": len(self.list_theories()),
            }
        }

    def to_prompt_section(self) -> str:
        """Format body state for system prompt."""
        s = self.system
        r = self.runtime
        i = self.identity
        e = self.embodiment
        m = e.mood

        identity_section = ""
        if i:
            identity_section = f"""### Identity
- Agent ID: {i.agent_id}
- Name: {i.name}
{f'- Parent: {i.parent_id}' if i.parent_id else '- Role: Root Agent'}
- Generation: {i.generation}
"""

        return f"""### System
- Host: {s.hostname}
- Platform: {s.platform} {s.platform_version} ({s.architecture})
- CPUs: {s.cpu_count}, Memory: {s.total_memory_gb}GB

### Runtime
- PID: {r.pid}
- Working Directory: {r.cwd}
- User: {r.user}

{identity_section}
### Embodiment
- Location: {e.location}
- Mode: {e.somatic_mode.value}
- Mood: confidence={m.confidence:.2f}, frustration={m.frustration:.2f}, curiosity={m.curiosity:.2f}

### Body Directory
- Path: {self.root}
- Steps: {self.get_step_count()}
- Memory: {len(self.list_episodes())} episodes, {len(self.list_procedures())} procedures
"""
