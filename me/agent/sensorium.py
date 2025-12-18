"""
Sensorium & Mouth - Perception and expression channels for embodied agents.

The sensorium is the collection of always-on information streams that
constitute the agent's perception. Unlike tool calls (which are active
queries), sensors are passive receivers - they deliver information
whether the agent asks for it or not.

The mouth is the agent's output channel - where it "speaks". By default
this is the normal response, but it can be configured to write to files,
streams (for speech synthesis), or channels (chat files that other agents
watch).

Philosophy:
- Your eyes don't "query" light. They receive it constantly. (Sensorium)
- Your mouth is how you speak. You can whisper to one, shout to many. (Mouth)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from me.agent.core import AgentConfig


class RefreshMode(Enum):
    """When to refresh sensor readings."""
    EVERY_TURN = "every_turn"   # Read fresh each turn
    ON_CHANGE = "on_change"     # Only when file mtime changes
    MANUAL = "manual"           # Only when explicitly requested
    PROCESS = "process"         # Read from a running process buffer


# =============================================================================
# Mouth - The agent's output/speech channel
# =============================================================================

class MouthMode(Enum):
    """Where the agent's speech goes."""
    RESPONSE = "response"   # Normal LLM response (default)
    FILE = "file"           # Write to a file (append mode)
    STREAM = "stream"       # Write to a stream/fd (for speech synthesis)
    CHANNEL = "channel"     # Write to shared channel (other agents watch)
    TEE = "tee"             # Both response AND another target


@dataclass
class Mouth:
    """
    The agent's output channel - where it "speaks".

    By default, agents speak through normal responses. But the mouth
    can be configured to:
    - Write to a file (e.g., a log, a chat transcript)
    - Write to a stream (e.g., fd mapped to speech synthesis)
    - Write to a channel (a file other agents watch via sensorium)
    - Tee to both response and another target

    This enables:
    - Agent-to-agent communication via watched files
    - Speech synthesis via fd mapping
    - Logging/transcripts of agent output
    - Silent agents that only write to files
    """
    mode: MouthMode = MouthMode.RESPONSE
    target: str | None = None       # File path, fd, or channel name
    encoding: str = "text"          # text | json | speech
    enabled: bool = True
    append: bool = True             # For file mode: append vs overwrite
    can_change: bool = True         # Whether agent can reconfigure its mouth

    # For channel mode
    channel_name: str | None = None

    # Runtime state
    _fd: int | None = field(default=None, repr=False)
    _file_handle: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "mode": self.mode.value,
            "target": self.target,
            "encoding": self.encoding,
            "enabled": self.enabled,
            "append": self.append,
            "can_change": self.can_change,
            "channel_name": self.channel_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mouth":
        """Create from dictionary."""
        return cls(
            mode=MouthMode(data.get("mode", "response")),
            target=data.get("target"),
            encoding=data.get("encoding", "text"),
            enabled=data.get("enabled", True),
            append=data.get("append", True),
            can_change=data.get("can_change", True),
            channel_name=data.get("channel_name"),
        )

    def open(self):
        """Open the output channel if needed."""
        if self.mode == MouthMode.FILE and self.target:
            mode = "a" if self.append else "w"
            self._file_handle = open(self.target, mode)
        elif self.mode == MouthMode.STREAM and self.target:
            # Target is an fd number as string
            try:
                self._fd = int(self.target)
            except ValueError:
                pass

    def close(self):
        """Close the output channel."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        self._fd = None

    def speak(self, content: str) -> bool:
        """
        Output content through this mouth.

        Returns True if content was written to a non-response target.
        For RESPONSE mode, returns False (content goes through normal response).
        """
        if not self.enabled:
            return False

        if self.mode == MouthMode.RESPONSE:
            # Normal response - content handled by caller
            return False

        elif self.mode == MouthMode.FILE:
            if self._file_handle:
                self._file_handle.write(content)
                self._file_handle.flush()
                return True
            elif self.target:
                # Open on demand
                mode = "a" if self.append else "w"
                with open(self.target, mode) as f:
                    f.write(content)
                return True

        elif self.mode == MouthMode.STREAM:
            if self._fd is not None:
                os.write(self._fd, content.encode())
                return True

        elif self.mode == MouthMode.CHANNEL:
            # Channel is a file that gets watched
            if self.target:
                with open(self.target, "a") as f:
                    # Write with timestamp for channel protocol
                    timestamp = datetime.now().isoformat()
                    if self.encoding == "json":
                        import json
                        f.write(json.dumps({
                            "timestamp": timestamp,
                            "channel": self.channel_name,
                            "content": content,
                        }) + "\n")
                    else:
                        f.write(f"[{timestamp}] {content}\n")
                return True

        elif self.mode == MouthMode.TEE:
            # Write to target AND return False so caller also uses response
            if self.target:
                with open(self.target, "a") as f:
                    f.write(content)
            return False  # Still output as response too

        return False

    def format_for_prompt(self) -> str:
        """Format mouth configuration for prompt injection."""
        if self.mode == MouthMode.RESPONSE:
            return "response (normal)"
        elif self.mode == MouthMode.FILE:
            return f"file: {self.target}"
        elif self.mode == MouthMode.STREAM:
            return f"stream: fd={self.target}"
        elif self.mode == MouthMode.CHANNEL:
            return f"channel: {self.channel_name} ({self.target})"
        elif self.mode == MouthMode.TEE:
            return f"tee: response + {self.target}"
        return "unknown"


@dataclass
class Sensor:
    """
    A single perceptual channel.

    Sensors watch files or paths and inject their (tail) content
    into every prompt. The agent can't choose not to see them -
    they are part of its body.

    For process sensors (RefreshMode.PROCESS), the source is ignored
    and content is read from the linked process buffer.

    Salience: Not all sensors are equal. Salience determines how much
    a sensor presses on attention. High salience sensors appear more
    prominently; low salience may be truncated or collapsed.
    """
    name: str                       # Human-readable identifier
    source: str                     # Path (supports {cwd}, {agent_dir}), or process command for PROCESS mode
    tail_chars: int = 200           # How much to show
    always_on: bool = False         # If True, cannot be disabled
    refresh: RefreshMode = RefreshMode.EVERY_TURN
    enabled: bool = True

    # Salience - how much this sensor presses on attention (0-1)
    salience: float = 0.5
    salience_triggers: list[str] = field(default_factory=list)  # Patterns that spike salience
    salience_decay: float = 0.1     # How fast salience fades when nothing happens

    # For PROCESS mode sensors
    process_id: str | None = None   # Linked process ID
    auto_remove: bool = False       # Remove when process completes

    # Runtime state (not persisted)
    _last_mtime: float = field(default=0.0, repr=False)
    _last_content: str = field(default="", repr=False)
    _triggered_this_turn: bool = field(default=False, repr=False)

    def update_salience(self, content: str):
        """Update salience based on content and triggers."""
        self._triggered_this_turn = False

        # Check for trigger patterns
        for trigger in self.salience_triggers:
            if trigger.lower() in content.lower():
                self.salience = min(1.0, self.salience + 0.3)
                self._triggered_this_turn = True
                return

        # Decay salience if no triggers found
        self.salience = max(0.2, self.salience - self.salience_decay)

    def resolve_path(self, cwd: Path, agent_dir: Path) -> Path:
        """Resolve the source path with variable substitution."""
        resolved = self.source.format(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
        )
        return Path(resolved).expanduser()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        d = {
            "name": self.name,
            "source": self.source,
            "tail_chars": self.tail_chars,
            "always_on": self.always_on,
            "refresh": self.refresh.value,
            "enabled": self.enabled,
            "salience": self.salience,
            "salience_triggers": self.salience_triggers,
            "salience_decay": self.salience_decay,
        }
        if self.process_id:
            d["process_id"] = self.process_id
        if self.auto_remove:
            d["auto_remove"] = self.auto_remove
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sensor":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            source=data["source"],
            tail_chars=data.get("tail_chars", 200),
            always_on=data.get("always_on", False),
            refresh=RefreshMode(data.get("refresh", "every_turn")),
            enabled=data.get("enabled", True),
            salience=data.get("salience", 0.5),
            salience_triggers=data.get("salience_triggers", []),
            salience_decay=data.get("salience_decay", 0.1),
            process_id=data.get("process_id"),
            auto_remove=data.get("auto_remove", False),
        )


@dataclass
class SensorReading:
    """A single reading from a sensor."""
    sensor_name: str
    source_path: str
    content: str
    truncated: bool         # True if content was tailed
    timestamp: str
    salience: float = 0.5   # Current salience level
    error: str | None = None


@dataclass
class ProprioceptiveState:
    """
    Internal state - direct, unmediated self-knowledge.

    Proprioception is different from external sensing. The agent knows
    its own state directly, not through sensors that might be stale.
    This is the difference between knowing your arm position (direct)
    and seeing a log file (mediated).
    """
    # From embodiment (set externally)
    stress: float = 0.0
    fatigue: float = 0.0
    mood_confidence: float = 0.5
    mood_frustration: float = 0.0
    mood_curiosity: float = 0.5
    mood_momentum: float = 0.0

    # From horizon
    turn_count: int = 0
    in_final_stretch: bool = False

    # From working set
    goal_count: int = 0
    open_loop_count: int = 0
    abandoned_count: int = 0

    def update_from_embodiment(self, embodiment_dict: dict[str, Any]):
        """Update proprioception from embodiment state."""
        somatic = embodiment_dict.get("somatic", {})
        self.stress = somatic.get("stress", 0.0)
        self.fatigue = somatic.get("fatigue", 0.0)

        mood = somatic.get("mood", {})
        self.mood_confidence = mood.get("confidence", 0.5)
        self.mood_frustration = mood.get("frustration", 0.0)
        self.mood_curiosity = mood.get("curiosity", 0.5)
        self.mood_momentum = mood.get("momentum", 0.0)

        horizon = embodiment_dict.get("horizon", {})
        self.turn_count = horizon.get("turn_count", 0)
        self.in_final_stretch = horizon.get("in_final_stretch", False)

        working_set = embodiment_dict.get("working_set", {})
        self.goal_count = len(working_set.get("goals", []))
        self.open_loop_count = len(working_set.get("open_loops", []))
        self.abandoned_count = len(working_set.get("abandoned", []))

    def to_prompt_section(self) -> str:
        """Format for prompt injection - brief internal status."""
        status_parts = []

        if self.stress > 0.5:
            status_parts.append(f"stress={self.stress:.2f}")
        if self.fatigue > 0.5:
            status_parts.append(f"fatigue={self.fatigue:.2f}")
        if self.mood_frustration > 0.3:
            status_parts.append(f"frustrated={self.mood_frustration:.2f}")
        if self.in_final_stretch:
            status_parts.append("FINAL_STRETCH")

        if status_parts:
            return f"[{', '.join(status_parts)}]"
        return "[nominal]"


class Sensorium:
    """
    The agent's perceptual and expressive system.

    The sensorium manages:
    - Proprioception: direct, unmediated internal state (always accurate)
    - Sensors: always-on input channels (files, processes, channels)
    - Mouth: the agent's output channel (response, file, stream, channel)

    Proprioception is epistemically distinct from external sensing.
    The agent knows its own state directly; it knows the world through
    sensors that might be stale or incomplete.

    Sensors are ordered by salience - what's pressing on attention appears first.
    """

    # Default sensors that every agent gets
    DEFAULT_SENSORS = [
        Sensor(
            name="self",
            source="{agent_dir}/current/state.json",
            tail_chars=500,
            always_on=True,
            refresh=RefreshMode.EVERY_TURN,
            salience=1.0,  # Self is always maximally salient
        ),
    ]

    def __init__(
        self,
        agent_dir: Path,
        cwd: Path,
        process_reader: "ProcessReader | None" = None,
        mouth: Mouth | None = None,
    ):
        self.agent_dir = agent_dir
        self.cwd = cwd
        self._sensors: dict[str, Sensor] = {}
        self._readings: dict[str, SensorReading] = {}
        self._process_reader = process_reader

        # Proprioception - direct internal state (updated externally)
        self.proprioception = ProprioceptiveState()

        # The agent's mouth - where it speaks
        self.mouth = mouth or Mouth()

        # Initialize with defaults
        for sensor in self.DEFAULT_SENSORS:
            self._sensors[sensor.name] = sensor

        # Load persisted config if exists
        self._load_config()

    def update_proprioception(self, embodiment_dict: dict[str, Any]):
        """Update proprioceptive state from embodiment."""
        self.proprioception.update_from_embodiment(embodiment_dict)

    def set_process_reader(self, reader: "ProcessReader"):
        """Set the process reader for process-based sensors."""
        self._process_reader = reader

    @property
    def config_path(self) -> Path:
        """Path to sensor configuration file."""
        return self.agent_dir / "sensors" / "config.json"

    def _load_config(self):
        """Load sensor and mouth configuration from disk."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                # Load sensors
                for name, sensor_data in data.get("sensors", {}).items():
                    # Don't override always_on sensors
                    if name in self._sensors and self._sensors[name].always_on:
                        continue
                    self._sensors[name] = Sensor.from_dict(sensor_data)
                # Load mouth config
                if "mouth" in data:
                    self.mouth = Mouth.from_dict(data["mouth"])
            except (json.JSONDecodeError, KeyError) as e:
                pass  # Use defaults on error

    def _save_config(self):
        """Persist sensor and mouth configuration to disk."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "sensors": {
                name: sensor.to_dict()
                for name, sensor in self._sensors.items()
            },
            "mouth": self.mouth.to_dict(),
        }
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_sensor(
        self,
        name: str,
        source: str,
        tail_chars: int = 200,
        always_on: bool = False,
        refresh: RefreshMode = RefreshMode.EVERY_TURN,
    ) -> Sensor:
        """Add a new sensor."""
        sensor = Sensor(
            name=name,
            source=source,
            tail_chars=tail_chars,
            always_on=always_on,
            refresh=refresh,
            enabled=True,
        )
        self._sensors[name] = sensor
        self._save_config()
        return sensor

    def add_terminal_sensor(
        self,
        process_id: str,
        command: str,
        tail_chars: int = 500,
    ) -> Sensor:
        """
        Add a sensor for a running terminal process.

        This is called automatically when a command runs longer than
        the streaming threshold (5x refresh rate). The terminal's
        output buffer becomes part of the agent's perception.

        The sensor is auto-removed when the process completes.
        """
        name = f"terminal:{process_id}"
        sensor = Sensor(
            name=name,
            source=command,  # Store command for display
            tail_chars=tail_chars,
            always_on=False,
            refresh=RefreshMode.PROCESS,
            enabled=True,
            process_id=process_id,
            auto_remove=True,
        )
        self._sensors[name] = sensor
        # Don't persist process sensors - they're runtime only
        return sensor

    def remove_terminal_sensor(self, process_id: str) -> bool:
        """Remove a terminal sensor by process ID."""
        name = f"terminal:{process_id}"
        if name in self._sensors:
            del self._sensors[name]
            return True
        return False

    def cleanup_completed_processes(self) -> list[str]:
        """
        Remove sensors for completed processes.

        Called each turn to clean up auto_remove sensors
        for processes that have finished.

        Returns list of removed sensor names.
        """
        if not self._process_reader:
            return []

        from me.agent.process import ProcessStatus

        removed = []
        to_remove = []

        for name, sensor in self._sensors.items():
            if sensor.refresh == RefreshMode.PROCESS and sensor.auto_remove:
                if sensor.process_id:
                    status = self._process_reader.get_status(sensor.process_id)
                    if status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED):
                        to_remove.append(name)

        for name in to_remove:
            del self._sensors[name]
            removed.append(name)

        return removed

    def remove_sensor(self, name: str) -> bool:
        """Remove a sensor. Returns False if sensor is always_on."""
        if name not in self._sensors:
            return False
        if self._sensors[name].always_on:
            return False
        del self._sensors[name]
        self._save_config()
        return True

    def enable_sensor(self, name: str) -> bool:
        """Enable a sensor."""
        if name not in self._sensors:
            return False
        self._sensors[name].enabled = True
        self._save_config()
        return True

    def disable_sensor(self, name: str) -> bool:
        """Disable a sensor. Returns False if sensor is always_on."""
        if name not in self._sensors:
            return False
        if self._sensors[name].always_on:
            return False
        self._sensors[name].enabled = False
        self._save_config()
        return True

    # =========================================================================
    # Mouth management
    # =========================================================================

    def set_mouth(
        self,
        mode: MouthMode | str,
        target: str | None = None,
        channel_name: str | None = None,
        encoding: str = "text",
        append: bool = True,
    ) -> bool:
        """
        Configure the agent's mouth (output channel).

        Returns False if mouth changes are not allowed.

        Args:
            mode: Where output goes (response, file, stream, channel, tee)
            target: File path, fd number, or channel path depending on mode
            channel_name: For channel mode, the logical channel name
            encoding: text | json | speech
            append: For file mode, whether to append or overwrite
        """
        if not self.mouth.can_change:
            return False

        # Handle string mode
        if isinstance(mode, str):
            mode = MouthMode(mode)

        # Close existing mouth if open
        self.mouth.close()

        # Configure new mouth
        self.mouth.mode = mode
        self.mouth.target = target
        self.mouth.channel_name = channel_name
        self.mouth.encoding = encoding
        self.mouth.append = append

        # Persist
        self._save_config()
        return True

    def get_mouth_config(self) -> dict[str, Any]:
        """Get current mouth configuration."""
        return {
            **self.mouth.to_dict(),
            "formatted": self.mouth.format_for_prompt(),
        }

    def list_sensors(self) -> list[dict[str, Any]]:
        """List all sensors with their configuration."""
        return [
            {
                **sensor.to_dict(),
                "resolved_path": str(sensor.resolve_path(self.cwd, self.agent_dir)),
            }
            for sensor in self._sensors.values()
        ]

    def update_cwd(self, cwd: Path):
        """Update the current working directory (for {cwd} substitution)."""
        self.cwd = cwd

    def _read_file_tail(self, path: Path, chars: int) -> tuple[str, bool]:
        """
        Read the tail of a file.

        Returns (content, truncated) tuple.
        """
        if not path.exists():
            return f"(file not found: {path})", False

        if not path.is_file():
            return f"(not a file: {path})", False

        try:
            # For efficiency, seek to end and read backwards
            file_size = path.stat().st_size
            if file_size <= chars:
                return path.read_text(), False

            with open(path, "r") as f:
                f.seek(max(0, file_size - chars))
                # Read a bit more to find a clean line break
                content = f.read()
                # Skip partial first line
                newline_pos = content.find("\n")
                if newline_pos > 0 and newline_pos < 50:
                    content = content[newline_pos + 1:]
                return content, True

        except Exception as e:
            return f"(error reading: {e})", False

    def read_sensor(self, name: str, full: bool = False) -> SensorReading | None:
        """
        Read a single sensor's value.

        If full=True, read entire file instead of just tail.
        For process sensors, reads from the process output buffer.
        """
        if name not in self._sensors:
            return None

        sensor = self._sensors[name]
        if not sensor.enabled:
            return SensorReading(
                sensor_name=name,
                source_path=sensor.source,
                content="(disabled)",
                truncated=False,
                timestamp=datetime.now().isoformat(),
            )

        # Handle process sensors
        if sensor.refresh == RefreshMode.PROCESS:
            return self._read_process_sensor(sensor, full)

        path = sensor.resolve_path(self.cwd, self.agent_dir)

        # Check refresh mode
        if sensor.refresh == RefreshMode.ON_CHANGE and not full:
            try:
                mtime = path.stat().st_mtime
                if mtime == sensor._last_mtime and sensor._last_content:
                    return SensorReading(
                        sensor_name=name,
                        source_path=str(path),
                        content=sensor._last_content,
                        truncated=True,
                        timestamp=datetime.now().isoformat(),
                    )
                sensor._last_mtime = mtime
            except:
                pass

        # Read the file
        if full:
            try:
                content = path.read_text()
                truncated = False
            except Exception as e:
                content = f"(error: {e})"
                truncated = False
        else:
            content, truncated = self._read_file_tail(path, sensor.tail_chars)

        # Cache for on_change mode
        sensor._last_content = content

        # Update salience based on content
        sensor.update_salience(content)

        return SensorReading(
            sensor_name=name,
            source_path=str(path),
            content=content,
            truncated=truncated,
            timestamp=datetime.now().isoformat(),
            salience=sensor.salience,
        )

    def _read_process_sensor(self, sensor: Sensor, full: bool = False) -> SensorReading:
        """Read content from a process output buffer."""
        if not self._process_reader or not sensor.process_id:
            return SensorReading(
                sensor_name=sensor.name,
                source_path=sensor.source,
                content="(no process reader)",
                truncated=False,
                timestamp=datetime.now().isoformat(),
                error="Process reader not available",
            )

        # Get process info
        from me.agent.process import ProcessStatus
        status = self._process_reader.get_status(sensor.process_id)
        tracked = self._process_reader._processes.get(sensor.process_id)

        if not tracked:
            return SensorReading(
                sensor_name=sensor.name,
                source_path=sensor.source,
                content="(process not found)",
                truncated=False,
                timestamp=datetime.now().isoformat(),
                error="Process not found",
            )

        # Read the output buffer
        if full:
            content = tracked.output_buffer
            truncated = False
        else:
            # Tail the buffer
            buffer = tracked.output_buffer
            if len(buffer) > sensor.tail_chars:
                content = buffer[-sensor.tail_chars:]
                # Skip partial first line
                newline_pos = content.find("\n")
                if 0 < newline_pos < 50:
                    content = content[newline_pos + 1:]
                truncated = True
            else:
                content = buffer
                truncated = False

        # Add status indicator
        status_str = f"[{status.value}]"
        if status == ProcessStatus.RUNNING:
            elapsed = (datetime.now() - tracked.started_at).total_seconds()
            status_str = f"[running {elapsed:.1f}s]"

        return SensorReading(
            sensor_name=sensor.name,
            source_path=f"{sensor.source} {status_str}",
            content=content if content else "(no output yet)",
            truncated=truncated,
            timestamp=datetime.now().isoformat(),
        )

    def read_all(self) -> list[SensorReading]:
        """Read all enabled sensors."""
        readings = []
        for name, sensor in self._sensors.items():
            if sensor.enabled:
                reading = self.read_sensor(name)
                if reading:
                    readings.append(reading)
        return readings

    def to_prompt_section(self) -> str:
        """
        Format sensorium state for prompt injection.

        This appears in every prompt - the agent cannot choose
        not to see it. Includes proprioception, sensor readings (ordered by salience),
        and mouth status.
        """
        parts = []

        # Proprioception - direct internal state (first, always)
        proprio_status = f"**Proprioception:** {self.proprioception.to_prompt_section()}"
        parts.append(proprio_status)

        # Mouth status (how the agent speaks)
        mouth_status = f"**Mouth:** {self.mouth.format_for_prompt()}"
        if not self.mouth.enabled:
            mouth_status += " (muted)"
        if not self.mouth.can_change:
            mouth_status += " (locked)"
        parts.append(mouth_status)

        # Sensor readings (what the agent sees) - ordered by salience
        readings = self.read_all()
        if not readings:
            parts.append("**Sensors:** (none)")
        else:
            # Sort by salience (highest first)
            readings.sort(key=lambda r: r.salience, reverse=True)

            parts.append("**Sensors:**")
            for reading in readings:
                # Show source, salience indicator, and size
                header = f"### {reading.sensor_name}"

                # Salience indicator
                if reading.salience > 0.7:
                    header += " [!]"  # High salience
                elif reading.salience < 0.3:
                    header += " [~]"  # Low salience

                if reading.truncated:
                    header += " (tail)"

                # Format content with ellipsis if truncated
                content = reading.content
                if reading.truncated:
                    content = "..." + content

                # For low salience sensors, truncate more aggressively
                if reading.salience < 0.3 and len(content) > 100:
                    content = content[:100] + "... [low salience, truncated]"

                parts.append(f"{header}\n```\n{content}\n```")

        return "\n\n".join(parts)

    def save_readings_to_step(self, step_dir: Path):
        """
        Save current sensor readings to a step directory.

        This creates the historical record of what the agent
        perceived at each step.
        """
        sensors_dir = step_dir / "sensors"
        sensors_dir.mkdir(parents=True, exist_ok=True)

        readings = self.read_all()

        # Save index
        index = {
            "timestamp": datetime.now().isoformat(),
            "sensors": [
                {
                    "name": r.sensor_name,
                    "source": r.source_path,
                    "truncated": r.truncated,
                }
                for r in readings
            ]
        }
        with open(sensors_dir / ".index.json", "w") as f:
            json.dump(index, f, indent=2)

        # Save each reading
        for reading in readings:
            # Use .txt for readability
            filename = f"{reading.sensor_name}.txt"
            with open(sensors_dir / filename, "w") as f:
                f.write(reading.content)
