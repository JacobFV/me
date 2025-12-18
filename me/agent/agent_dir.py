"""
Agent Directory - The body as filesystem.

Every agent has a directory that IS its body - a virtual filesystem
mapping the agent's existence into navigable structure. The directory
is not a cache or log; it IS the agent.

Structure:
    ~/.me/agents/<agent-id>/
    ├── identity.json       # Immutable: agent_id, name, lineage (WHO I AM)
    ├── character.json      # Slowly changing: values, tendencies (WHO I'VE BECOME)
    ├── config.json         # Mutable: settings, preferences (HOW I OPERATE)
    ├── current -> steps/N  # Symlink to latest step
    ├── session/            # Ephemeral: current session state
    │   ├── working_set.json
    │   └── open_loops.json
    ├── steps/
    │   └── NNNN/
    │       ├── state.json  # Embodiment snapshot
    │       ├── input.txt   # What was received
    │       ├── output.txt  # What was produced
    │       └── sensors/    # Sensor readings
    └── sensors/
        └── config.json     # Sensor configuration

Philosophy: Copy the directory, and you've copied the agent.
The filesystem IS the agent's identity and history.

The distinction between identity, character, and session matters:
- Identity: Never changes. Who you ARE.
- Character: Changes slowly. Who you've BECOME through your choices.
- Session: Ephemeral. What you're doing NOW.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from me.agent.core import AgentConfig


@dataclass
class AgentIdentity:
    """
    Immutable identity - set at creation, never changes.

    This is who the agent IS, not how it operates.
    """
    agent_id: str
    name: str
    created_at: str
    parent_id: str | None = None
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentIdentity":
        return cls(**data)


@dataclass
class StepRecord:
    """A single step in the agent's history."""
    step_number: int
    timestamp: str
    input_text: str | None = None
    output_text: str | None = None
    state_snapshot: dict[str, Any] | None = None


class AgentDirectory:
    """
    Manages the agent's directory structure.

    The directory is the agent's body - it contains:
    - Identity (immutable)
    - Configuration (mutable)
    - Step history (append-only)
    - Sensor configuration
    """

    def __init__(self, base_dir: Path, agent_id: str, name: str = "me"):
        self.base_dir = base_dir
        self.agent_id = agent_id
        self.name = name
        self._step_count = 0

        # Ensure base directory exists
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Root directory for this agent."""
        return self.base_dir / "agents" / self.agent_id

    @property
    def identity_path(self) -> Path:
        return self.root / "identity.json"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def character_path(self) -> Path:
        """Character - who the agent has become through choices."""
        return self.root / "character.json"

    @property
    def session_dir(self) -> Path:
        """Session state - ephemeral, what's happening now."""
        return self.root / "session"

    @property
    def steps_dir(self) -> Path:
        return self.root / "steps"

    @property
    def current_link(self) -> Path:
        return self.root / "current"

    @property
    def sensors_dir(self) -> Path:
        return self.root / "sensors"

    def initialize(self, parent_id: str | None = None, generation: int = 0) -> AgentIdentity:
        """
        Initialize the agent directory structure.

        Creates identity.json (immutable) and sets up directories.
        Should only be called once when agent is first created.
        """
        # Create directory structure
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.sensors_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Check if identity already exists
        if self.identity_path.exists():
            return self.load_identity()

        # Create identity (immutable)
        identity = AgentIdentity(
            agent_id=self.agent_id,
            name=self.name,
            created_at=datetime.now().isoformat(),
            parent_id=parent_id,
            generation=generation,
        )

        with open(self.identity_path, "w") as f:
            json.dump(identity.to_dict(), f, indent=2)

        # Create initial config
        if not self.config_path.exists():
            self.save_config({
                "refresh_rate_ms": 100,
                "permission_mode": "acceptEdits",
                "model": None,
            })

        return identity

    def load_identity(self) -> AgentIdentity | None:
        """Load the agent's identity."""
        if not self.identity_path.exists():
            return None
        with open(self.identity_path) as f:
            return AgentIdentity.from_dict(json.load(f))

    def load_config(self) -> dict[str, Any]:
        """Load the agent's configuration."""
        if not self.config_path.exists():
            return {}
        with open(self.config_path) as f:
            return json.load(f)

    def save_config(self, config: dict[str, Any]):
        """Save the agent's configuration."""
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def update_config(self, updates: dict[str, Any]):
        """Update specific config fields."""
        config = self.load_config()
        config.update(updates)
        self.save_config(config)

    def _get_next_step_number(self) -> int:
        """Get the next step number."""
        if not self.steps_dir.exists():
            return 1

        existing = [
            int(d.name) for d in self.steps_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
        if not existing:
            return 1
        return max(existing) + 1

    def _get_step_dir(self, step_number: int) -> Path:
        """Get the directory for a specific step."""
        return self.steps_dir / f"{step_number:04d}"

    def current_step_dir(self) -> Path | None:
        """Get the current step directory (follows symlink)."""
        if not self.current_link.exists():
            return None
        return self.current_link.resolve()

    def begin_step(self, input_text: str | None = None) -> Path:
        """
        Begin a new step.

        Creates the step directory and writes input.txt.
        Returns the step directory path.
        """
        step_num = self._get_next_step_number()
        step_dir = self._get_step_dir(step_num)
        step_dir.mkdir(parents=True, exist_ok=True)

        # Write input
        if input_text:
            with open(step_dir / "input.txt", "w") as f:
                f.write(input_text)

        # Update current symlink
        self._update_current_link(step_dir)

        self._step_count = step_num
        return step_dir

    def _update_current_link(self, step_dir: Path):
        """Update the 'current' symlink to point to the given step."""
        # Remove existing symlink
        if self.current_link.exists() or self.current_link.is_symlink():
            self.current_link.unlink()

        # Create relative symlink
        relative_path = step_dir.relative_to(self.root)
        self.current_link.symlink_to(relative_path)

    def complete_step(
        self,
        output_text: str | None = None,
        state_snapshot: dict[str, Any] | None = None,
    ):
        """
        Complete the current step.

        Writes output.txt and state.json.
        """
        step_dir = self.current_step_dir()
        if not step_dir:
            return

        # Write output
        if output_text:
            with open(step_dir / "output.txt", "w") as f:
                f.write(output_text)

        # Write state snapshot
        if state_snapshot:
            with open(step_dir / "state.json", "w") as f:
                json.dump(state_snapshot, f, indent=2)

    def get_step(self, step_number: int) -> StepRecord | None:
        """Load a specific step's record."""
        step_dir = self._get_step_dir(step_number)
        if not step_dir.exists():
            return None

        record = StepRecord(
            step_number=step_number,
            timestamp="",  # TODO: get from state.json
        )

        input_path = step_dir / "input.txt"
        if input_path.exists():
            record.input_text = input_path.read_text()

        output_path = step_dir / "output.txt"
        if output_path.exists():
            record.output_text = output_path.read_text()

        state_path = step_dir / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                record.state_snapshot = json.load(f)
                record.timestamp = record.state_snapshot.get("timestamp", "")

        return record

    def list_steps(self, limit: int = 10) -> list[int]:
        """List recent step numbers."""
        if not self.steps_dir.exists():
            return []

        steps = sorted([
            int(d.name) for d in self.steps_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ], reverse=True)

        return steps[:limit]

    def get_step_count(self) -> int:
        """Get total number of steps."""
        if not self.steps_dir.exists():
            return 0
        return len([
            d for d in self.steps_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])

    def export(self, destination: Path) -> bool:
        """
        Export the entire agent directory.

        This creates a complete copy of the agent.
        """
        try:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(self.root, destination, symlinks=True)
            return True
        except Exception:
            return False

    @classmethod
    def import_agent(cls, source: Path, base_dir: Path) -> "AgentDirectory | None":
        """
        Import an agent from an exported directory.

        Returns the AgentDirectory instance if successful.
        """
        # Load identity to get agent_id
        identity_path = source / "identity.json"
        if not identity_path.exists():
            return None

        with open(identity_path) as f:
            identity_data = json.load(f)

        agent_id = identity_data.get("agent_id")
        if not agent_id:
            return None

        # Copy to agents directory
        dest = base_dir / "agents" / agent_id
        if dest.exists():
            # Agent already exists
            return cls(base_dir, agent_id)

        try:
            shutil.copytree(source, dest, symlinks=True)
            return cls(base_dir, agent_id, identity_data.get("name", "me"))
        except Exception:
            return None

    def to_dict(self) -> dict[str, Any]:
        """Get summary of agent directory state."""
        identity = self.load_identity()
        config = self.load_config()

        return {
            "root": str(self.root),
            "agent_id": self.agent_id,
            "identity": identity.to_dict() if identity else None,
            "config": config,
            "step_count": self.get_step_count(),
            "current_step": self.current_step_dir().name if self.current_step_dir() else None,
        }
