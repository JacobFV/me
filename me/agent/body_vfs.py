"""
Body VFS - The agent's body as a virtual filesystem.

The agent's body IS a set of files. When the agent reads/writes these files,
it reads/writes its own configuration, sensors, mouth, etc. No special tools
needed - just Edit and Write to your own body files.

## File Structure

```
~/.me/agents/<agent-id>/
├── identity.json           # IMMUTABLE - who this agent is
├── config.json             # Agent configuration (refresh_rate, permissions, etc.)
├── sensors.json            # Sensor definitions
├── mouth.json              # Mouth configuration
├── embodiment.json         # Location, capabilities, somatic state
├── working_set.json        # Goals, tasks, open loops
├── current -> steps/NNNN   # Symlink to current step
└── steps/
    └── NNNN/               # Step N's complete state
        ├── identity.json   # Copy (for reproducibility)
        ├── config.json
        ├── sensors.json
        ├── mouth.json
        ├── embodiment.json
        ├── working_set.json
        ├── input.txt       # What was received this step
        ├── output.txt      # What was produced this step
        └── readings/       # Sensor readings at this step
            ├── self.txt
            └── ...
```

## Step Lifecycle

1. **INITIALIZE** - Load state from `steps/NNNN-1/` (or defaults if step 1)
   - Parse all JSON files into in-memory dataclasses
   - Set up the virtual layer

2. **VIRTUAL LAYER** - Agent sees body files at root level
   - Read `~/.me/agents/<id>/config.json` → returns current in-memory state as JSON
   - Write `~/.me/agents/<id>/config.json` → validates JSON, updates in-memory, queues for flush

3. **PERFORM STEP** - Agent runs, may edit its own files
   - Standard Edit/Write tools work on body paths
   - Writes are validated (must be valid JSON for .json files)
   - Invalid writes return errors to the agent
   - identity.json writes return PermissionError

4. **FLUSH** - Write final state to `steps/NNNN/`
   - All body files are copied to step directory
   - Symlink `current` updated
   - This IS the agent's state at step N

## Integration with Agent Core

The body VFS integrates with the agent like this:

```python
from me.agent.body_vfs import BodyVFS
from me.agent.file_intercept import FileInterceptor, ToolWrapper

class Agent:
    def __init__(self, config):
        # Create body VFS
        self.body_vfs = BodyVFS(
            base_dir=Path.home() / ".me",
            agent_id=config.agent_id,
        )

        # Register callbacks for state changes
        self.body_vfs.on_config_change(self._handle_config_change)
        self.body_vfs.on_sensors_change(self._handle_sensors_change)
        self.body_vfs.on_mouth_change(self._handle_mouth_change)

        # Create file interceptor
        self.interceptor = FileInterceptor(self.body_vfs)

    def _handle_config_change(self, new_config):
        # Update in-memory config
        self.config.refresh_rate_ms = new_config.get("refresh_rate_ms", 100)
        # etc...

    def _handle_sensors_change(self, new_sensors):
        # Rebuild sensorium from new config
        self.sensorium.rebuild_from_config(new_sensors)

    async def run(self, prompt: str):
        # Initialize step
        step_number = self.body_vfs.get_step_count() + 1
        self.body_vfs.initialize_step(step_number, input_text=prompt)

        # Run agent loop (file operations are intercepted)
        # ...

        # Flush step
        self.body_vfs.flush_step()
```

## What This Enables

Instead of:
```python
# Old way - special tools for everything
await set_mouth(mode="file", target="/tmp/out.txt")
await add_sensor(name="logs", source="/var/log/app.log")
await set_refresh_rate(rate_ms=50)
```

The agent can now:
```python
# New way - just edit your own files
Edit("~/.me/agents/abc123/mouth.json", ...)
Edit("~/.me/agents/abc123/sensors.json", ...)
Edit("~/.me/agents/abc123/config.json", ...)
```

This is:
1. **Simpler** - No special tools to learn
2. **Transparent** - `cat` the file to see the config
3. **Unix-y** - Standard file operations work
4. **Reproducible** - Each step has complete state snapshot
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from me.agent.sensorium import Sensorium, Mouth, MouthMode
    from me.agent.embodiment import Embodiment


class BodyFileType(Enum):
    """Types of files in the body VFS."""
    IMMUTABLE = "immutable"     # identity.json - reads ok, writes error
    CONFIG = "config"           # config.json - reads/writes update in-memory
    SENSORS = "sensors"         # sensors.json - reads/writes update sensorium
    MOUTH = "mouth"             # mouth.json - reads/writes update mouth
    EMBODIMENT = "embodiment"   # embodiment.json - reads/writes update embodiment
    WORKING_SET = "working_set" # working_set.json - reads/writes update working set
    TEXT = "text"               # input.txt, output.txt - plain text
    DIRECTORY = "directory"     # directories


@dataclass
class BodyFile:
    """Metadata about a body file."""
    name: str
    file_type: BodyFileType
    writable: bool = True
    validator: Optional[Callable[[dict], tuple[bool, str]]] = None


# Define the body file schema
BODY_FILES = {
    "identity.json": BodyFile("identity.json", BodyFileType.IMMUTABLE, writable=False),
    "config.json": BodyFile("config.json", BodyFileType.CONFIG),
    "sensors.json": BodyFile("sensors.json", BodyFileType.SENSORS),
    "mouth.json": BodyFile("mouth.json", BodyFileType.MOUTH),
    "embodiment.json": BodyFile("embodiment.json", BodyFileType.EMBODIMENT),
    "working_set.json": BodyFile("working_set.json", BodyFileType.WORKING_SET),
}


@dataclass
class WriteResult:
    """Result of a write operation."""
    success: bool
    error: Optional[str] = None
    bytes_written: int = 0


class BodyVFS:
    """
    Virtual filesystem layer for the agent's body.

    This class intercepts reads and writes to the agent's body files,
    translating them to/from in-memory state.

    Usage:
        body_vfs = BodyVFS(agent_dir, agent_id)
        body_vfs.initialize_step(previous_step=41)

        # During step, intercept file operations:
        content = body_vfs.read("config.json")
        result = body_vfs.write("config.json", new_content)

        # At step end:
        body_vfs.flush_step()
    """

    def __init__(
        self,
        base_dir: Path,
        agent_id: str,
    ):
        self.base_dir = base_dir
        self.agent_id = agent_id
        self.root = base_dir / "agents" / agent_id

        # In-memory state (populated during initialize_step)
        self._identity: dict[str, Any] = {}
        self._config: dict[str, Any] = {}
        self._sensors: dict[str, Any] = {}
        self._mouth: dict[str, Any] = {}
        self._embodiment: dict[str, Any] = {}
        self._working_set: dict[str, Any] = {}

        # Step tracking
        self._current_step: int = 0
        self._step_input: str = ""
        self._step_output_parts: list[str] = []

        # Callbacks for state changes (set by Agent)
        self._on_config_change: Optional[Callable[[dict], None]] = None
        self._on_sensors_change: Optional[Callable[[dict], None]] = None
        self._on_mouth_change: Optional[Callable[[dict], None]] = None
        self._on_embodiment_change: Optional[Callable[[dict], None]] = None
        self._on_working_set_change: Optional[Callable[[dict], None]] = None

        # Ensure directory structure exists
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "steps").mkdir(exist_ok=True)

    # =========================================================================
    # Path handling
    # =========================================================================

    def is_body_path(self, path: Path | str) -> bool:
        """Check if a path is within this agent's body directory."""
        try:
            path = Path(path).resolve()
            return path.is_relative_to(self.root)
        except (ValueError, OSError):
            return False

    def relative_path(self, path: Path | str) -> str:
        """Get the path relative to the agent root."""
        path = Path(path).resolve()
        return str(path.relative_to(self.root))

    def get_file_type(self, rel_path: str) -> Optional[BodyFileType]:
        """Get the file type for a relative path."""
        # Check if it's a known body file
        if rel_path in BODY_FILES:
            return BODY_FILES[rel_path].file_type

        # Check if it's in steps directory
        if rel_path.startswith("steps/"):
            return BodyFileType.TEXT

        # Check if it's a directory
        full_path = self.root / rel_path
        if full_path.is_dir():
            return BodyFileType.DIRECTORY

        return None

    # =========================================================================
    # Step lifecycle
    # =========================================================================

    def initialize_step(
        self,
        step_number: int,
        input_text: str = "",
        seed_state: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a new step.

        Loads state from the previous step (or seed_state if provided).
        Creates the virtual layer for this step.
        """
        self._current_step = step_number
        self._step_input = input_text
        self._step_output_parts = []

        # Determine where to load state from
        if seed_state:
            # Use provided seed state
            self._load_from_dict(seed_state)
        elif step_number > 1:
            # Load from previous step
            prev_step_dir = self.root / "steps" / f"{step_number - 1:04d}"
            if prev_step_dir.exists():
                self._load_from_step_dir(prev_step_dir)
            else:
                # Fall back to root files (legacy or first run)
                self._load_from_root()
        else:
            # First step - load from root or create defaults
            self._load_from_root()

        # Create step directory
        step_dir = self._current_step_dir()
        step_dir.mkdir(parents=True, exist_ok=True)

        # Write input
        if input_text:
            (step_dir / "input.txt").write_text(input_text)

    def _current_step_dir(self) -> Path:
        """Get the directory for the current step."""
        return self.root / "steps" / f"{self._current_step:04d}"

    def _load_from_step_dir(self, step_dir: Path) -> None:
        """Load state from a step directory."""
        self._identity = self._read_json_file(step_dir / "identity.json", {})
        self._config = self._read_json_file(step_dir / "config.json", {})
        self._sensors = self._read_json_file(step_dir / "sensors.json", {})
        self._mouth = self._read_json_file(step_dir / "mouth.json", {})
        self._embodiment = self._read_json_file(step_dir / "embodiment.json", {})
        self._working_set = self._read_json_file(step_dir / "working_set.json", {})

    def _load_from_root(self) -> None:
        """Load state from root-level files (legacy/init)."""
        self._identity = self._read_json_file(self.root / "identity.json", {})
        self._config = self._read_json_file(self.root / "config.json", self._default_config())
        self._sensors = self._read_json_file(self.root / "sensors.json", {"sensors": {}})
        self._mouth = self._read_json_file(self.root / "mouth.json", self._default_mouth())
        self._embodiment = self._read_json_file(self.root / "embodiment.json", {})
        self._working_set = self._read_json_file(self.root / "working_set.json", self._default_working_set())

    def _load_from_dict(self, state: dict[str, Any]) -> None:
        """Load state from a dictionary (for seeding)."""
        self._identity = state.get("identity", {})
        self._config = state.get("config", self._default_config())
        self._sensors = state.get("sensors", {"sensors": {}})
        self._mouth = state.get("mouth", self._default_mouth())
        self._embodiment = state.get("embodiment", {})
        self._working_set = state.get("working_set", self._default_working_set())

    def _read_json_file(self, path: Path, default: dict) -> dict:
        """Read a JSON file, returning default if not found or invalid."""
        if not path.exists():
            return default
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return default

    def _default_config(self) -> dict[str, Any]:
        """Default agent configuration."""
        return {
            "refresh_rate_ms": 100,
            "permission_mode": "acceptEdits",
            "model": None,
        }

    def _default_mouth(self) -> dict[str, Any]:
        """Default mouth configuration."""
        return {
            "mode": "response",
            "target": None,
            "encoding": "text",
            "enabled": True,
            "append": True,
            "can_change": True,
            "channel_name": None,
        }

    def _default_working_set(self) -> dict[str, Any]:
        """Default working set."""
        return {
            "goals": [],
            "active_tasks": [],
            "pending_decisions": [],
            "open_loops": [],
            "last_actions": [],
        }

    def append_output(self, text: str) -> None:
        """Append text to the step's output."""
        self._step_output_parts.append(text)

    def flush_step(self) -> Path:
        """
        Flush the current step to disk.

        Writes all in-memory state to the step directory,
        updates the `current` symlink.

        Returns the step directory path.
        """
        step_dir = self._current_step_dir()
        step_dir.mkdir(parents=True, exist_ok=True)

        # Write all body files to step directory
        self._write_json_file(step_dir / "identity.json", self._identity)
        self._write_json_file(step_dir / "config.json", self._config)
        self._write_json_file(step_dir / "sensors.json", self._sensors)
        self._write_json_file(step_dir / "mouth.json", self._mouth)
        self._write_json_file(step_dir / "embodiment.json", self._embodiment)
        self._write_json_file(step_dir / "working_set.json", self._working_set)

        # Write output
        if self._step_output_parts:
            (step_dir / "output.txt").write_text("\n".join(self._step_output_parts))

        # Update current symlink
        current_link = self.root / "current"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(f"steps/{self._current_step:04d}")

        # Also write to root for easy access (these are the "live" files)
        self._write_json_file(self.root / "config.json", self._config)
        self._write_json_file(self.root / "sensors.json", self._sensors)
        self._write_json_file(self.root / "mouth.json", self._mouth)
        self._write_json_file(self.root / "embodiment.json", self._embodiment)
        self._write_json_file(self.root / "working_set.json", self._working_set)

        return step_dir

    def _write_json_file(self, path: Path, data: dict) -> None:
        """Write a dictionary to a JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # =========================================================================
    # Read operations
    # =========================================================================

    def read(self, rel_path: str) -> bytes:
        """
        Read a body file.

        Returns the current in-memory state as JSON for body files,
        or actual file contents for other files.
        """
        # Route to appropriate state
        if rel_path == "identity.json":
            return json.dumps(self._identity, indent=2).encode()
        elif rel_path == "config.json":
            return json.dumps(self._config, indent=2).encode()
        elif rel_path == "sensors.json":
            return json.dumps(self._sensors, indent=2).encode()
        elif rel_path == "mouth.json":
            return json.dumps(self._mouth, indent=2).encode()
        elif rel_path == "embodiment.json":
            return json.dumps(self._embodiment, indent=2).encode()
        elif rel_path == "working_set.json":
            return json.dumps(self._working_set, indent=2).encode()

        # For other files, read from disk
        full_path = self.root / rel_path
        if full_path.exists() and full_path.is_file():
            return full_path.read_bytes()

        raise FileNotFoundError(f"Body file not found: {rel_path}")

    def read_text(self, rel_path: str) -> str:
        """Read a body file as text."""
        return self.read(rel_path).decode("utf-8")

    # =========================================================================
    # Write operations
    # =========================================================================

    def write(self, rel_path: str, content: bytes) -> WriteResult:
        """
        Write to a body file.

        For JSON body files:
        - Validates the content is valid JSON
        - Validates against schema (if any)
        - Updates in-memory state
        - Triggers callbacks for state changes

        Returns WriteResult with success/error info.
        """
        # Check if immutable
        if rel_path == "identity.json":
            return WriteResult(
                success=False,
                error="identity.json is immutable - cannot write to it"
            )

        # For JSON body files, validate and update in-memory
        if rel_path in ("config.json", "sensors.json", "mouth.json",
                        "embodiment.json", "working_set.json"):
            return self._write_json_body_file(rel_path, content)

        # For other files (like in steps/), write directly
        full_path = self.root / rel_path

        # Don't allow writing outside the agent directory
        try:
            full_path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return WriteResult(
                success=False,
                error=f"Cannot write outside agent directory: {rel_path}"
            )

        # Write the file
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(content)
            return WriteResult(success=True, bytes_written=len(content))
        except Exception as e:
            return WriteResult(success=False, error=str(e))

    def _write_json_body_file(self, rel_path: str, content: bytes) -> WriteResult:
        """Write a JSON body file with validation."""
        # Parse JSON
        try:
            data = json.loads(content.decode("utf-8"))
        except json.JSONDecodeError as e:
            return WriteResult(
                success=False,
                error=f"Invalid JSON in {rel_path}: {e}"
            )

        # Validate and update based on file type
        if rel_path == "config.json":
            valid, error = self._validate_config(data)
            if not valid:
                return WriteResult(success=False, error=error)
            self._config = data
            if self._on_config_change:
                self._on_config_change(data)

        elif rel_path == "sensors.json":
            valid, error = self._validate_sensors(data)
            if not valid:
                return WriteResult(success=False, error=error)
            self._sensors = data
            if self._on_sensors_change:
                self._on_sensors_change(data)

        elif rel_path == "mouth.json":
            valid, error = self._validate_mouth(data)
            if not valid:
                return WriteResult(success=False, error=error)
            self._mouth = data
            if self._on_mouth_change:
                self._on_mouth_change(data)

        elif rel_path == "embodiment.json":
            valid, error = self._validate_embodiment(data)
            if not valid:
                return WriteResult(success=False, error=error)
            self._embodiment = data
            if self._on_embodiment_change:
                self._on_embodiment_change(data)

        elif rel_path == "working_set.json":
            valid, error = self._validate_working_set(data)
            if not valid:
                return WriteResult(success=False, error=error)
            self._working_set = data
            if self._on_working_set_change:
                self._on_working_set_change(data)

        return WriteResult(success=True, bytes_written=len(content))

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_config(self, data: dict) -> tuple[bool, Optional[str]]:
        """Validate config.json structure."""
        # Required fields
        if "refresh_rate_ms" in data:
            if not isinstance(data["refresh_rate_ms"], int) or data["refresh_rate_ms"] < 10:
                return False, "refresh_rate_ms must be an integer >= 10"

        if "permission_mode" in data:
            valid_modes = ["default", "acceptEdits", "bypassPermissions", "plan"]
            if data["permission_mode"] not in valid_modes:
                return False, f"permission_mode must be one of: {valid_modes}"

        return True, None

    def _validate_sensors(self, data: dict) -> tuple[bool, Optional[str]]:
        """Validate sensors.json structure."""
        if "sensors" not in data:
            return False, "sensors.json must have a 'sensors' key"

        if not isinstance(data["sensors"], dict):
            return False, "'sensors' must be a dictionary"

        for name, sensor in data["sensors"].items():
            if not isinstance(sensor, dict):
                return False, f"Sensor '{name}' must be a dictionary"
            if "source" not in sensor:
                return False, f"Sensor '{name}' must have a 'source' field"

        return True, None

    def _validate_mouth(self, data: dict) -> tuple[bool, Optional[str]]:
        """Validate mouth.json structure."""
        if "mode" in data:
            valid_modes = ["response", "file", "stream", "channel", "tee"]
            if data["mode"] not in valid_modes:
                return False, f"mouth mode must be one of: {valid_modes}"

        # Check if mouth changes are allowed
        if "can_change" in self._mouth and not self._mouth.get("can_change", True):
            return False, "Mouth configuration is locked and cannot be changed"

        return True, None

    def _validate_embodiment(self, data: dict) -> tuple[bool, Optional[str]]:
        """Validate embodiment.json structure."""
        # Most embodiment changes are allowed
        # But some fields might be protected
        return True, None

    def _validate_working_set(self, data: dict) -> tuple[bool, Optional[str]]:
        """Validate working_set.json structure."""
        if "goals" in data and not isinstance(data["goals"], list):
            return False, "'goals' must be a list"
        if "open_loops" in data and not isinstance(data["open_loops"], list):
            return False, "'open_loops' must be a list"
        return True, None

    # =========================================================================
    # Callback registration
    # =========================================================================

    def on_config_change(self, callback: Callable[[dict], None]) -> None:
        """Register callback for config changes."""
        self._on_config_change = callback

    def on_sensors_change(self, callback: Callable[[dict], None]) -> None:
        """Register callback for sensor changes."""
        self._on_sensors_change = callback

    def on_mouth_change(self, callback: Callable[[dict], None]) -> None:
        """Register callback for mouth changes."""
        self._on_mouth_change = callback

    def on_embodiment_change(self, callback: Callable[[dict], None]) -> None:
        """Register callback for embodiment changes."""
        self._on_embodiment_change = callback

    def on_working_set_change(self, callback: Callable[[dict], None]) -> None:
        """Register callback for working set changes."""
        self._on_working_set_change = callback

    # =========================================================================
    # State access (for agent internals)
    # =========================================================================

    @property
    def identity(self) -> dict[str, Any]:
        return self._identity

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def sensors(self) -> dict[str, Any]:
        return self._sensors

    @property
    def mouth(self) -> dict[str, Any]:
        return self._mouth

    @property
    def embodiment(self) -> dict[str, Any]:
        return self._embodiment

    @property
    def working_set(self) -> dict[str, Any]:
        return self._working_set

    @property
    def current_step(self) -> int:
        return self._current_step

    # =========================================================================
    # Initialization
    # =========================================================================

    def create_identity(
        self,
        name: str = "me",
        parent_id: Optional[str] = None,
        generation: int = 0,
    ) -> dict[str, Any]:
        """
        Create the identity file (only on first creation).

        This should only be called once when the agent is first created.
        Returns the identity dict.
        """
        if self._identity:
            return self._identity  # Already exists

        identity = {
            "agent_id": self.agent_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "parent_id": parent_id,
            "generation": generation,
        }

        # Write to root (immutable)
        self._write_json_file(self.root / "identity.json", identity)
        self._identity = identity

        return identity

    # =========================================================================
    # Directory listing
    # =========================================================================

    def list_dir(self, rel_path: str = "") -> list[dict[str, Any]]:
        """List contents of a directory in the body VFS."""
        full_path = self.root / rel_path if rel_path else self.root

        if not full_path.exists():
            raise FileNotFoundError(f"Directory not found: {rel_path}")

        if not full_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {rel_path}")

        entries = []
        for item in full_path.iterdir():
            stat = item.stat()
            entries.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else 0,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return sorted(entries, key=lambda e: (e["type"] != "directory", e["name"]))

    # =========================================================================
    # Utility
    # =========================================================================

    def get_step_count(self) -> int:
        """Get the number of completed steps."""
        steps_dir = self.root / "steps"
        if not steps_dir.exists():
            return 0
        return len([
            d for d in steps_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])

    def get_previous_step_dir(self) -> Optional[Path]:
        """Get the previous step's directory."""
        if self._current_step <= 1:
            return None
        prev_dir = self.root / "steps" / f"{self._current_step - 1:04d}"
        return prev_dir if prev_dir.exists() else None

    def export(self, destination: Path) -> bool:
        """Export the entire agent directory."""
        try:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(self.root, destination, symlinks=True)
            return True
        except Exception:
            return False

    def to_dict(self) -> dict[str, Any]:
        """Get the complete current state as a dictionary."""
        return {
            "identity": self._identity,
            "config": self._config,
            "sensors": self._sensors,
            "mouth": self._mouth,
            "embodiment": self._embodiment,
            "working_set": self._working_set,
            "step": self._current_step,
        }
