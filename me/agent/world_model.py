"""
World Model - Explicit representation of environment state and dynamics.

The agent doesn't just react to the world—it builds an internal model of:
- Filesystem structure and contents
- Process state and dependencies
- State transitions (how actions change the world)
- Causal relationships (what causes what)
- Predictions (what will happen next)

Philosophy: Intelligence requires modeling. You can't plan effectively without
understanding the environment's structure, what's possible, and what effects your
actions will have. The world model is the substrate for reasoning about the unseen.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any


# =============================================================================
# File System Model
# =============================================================================

class FileType(str, Enum):
    """Types of filesystem entries."""
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    UNKNOWN = "unknown"


@dataclass
class FileNode:
    """A node in the filesystem model."""
    path: str
    type: FileType
    size: int = 0
    modified: datetime | None = None
    permissions: str = ""
    content_hash: str | None = None  # SHA256 hash for change detection
    children: list[str] = field(default_factory=list)  # For directories
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "type": self.type.value,
            "size": self.size,
            "modified": self.modified.isoformat() if self.modified else None,
            "permissions": self.permissions,
            "content_hash": self.content_hash,
            "children": self.children,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileNode":
        return cls(
            path=data["path"],
            type=FileType(data["type"]),
            size=data.get("size", 0),
            modified=datetime.fromisoformat(data["modified"]) if data.get("modified") else None,
            permissions=data.get("permissions", ""),
            content_hash=data.get("content_hash"),
            children=data.get("children", []),
            metadata=data.get("metadata", {}),
        )


class FileSystemModel:
    """
    Model of the filesystem.

    Tracks directory structure, file metadata, and changes over time.
    Enables reasoning about:
    - What files exist where
    - What has changed since last observation
    - What patterns exist in the structure
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.fs_model_path = self.model_dir / "filesystem.json"

        # Load or initialize
        self._nodes: dict[str, FileNode] = {}
        self._load()

    def observe_path(self, path: Path) -> FileNode:
        """
        Observe a filesystem path and update the model.

        Returns the FileNode representing this path.
        """
        path_str = str(path.resolve())

        # Determine type
        if path.is_symlink():
            ftype = FileType.SYMLINK
        elif path.is_dir():
            ftype = FileType.DIRECTORY
        elif path.is_file():
            ftype = FileType.FILE
        else:
            ftype = FileType.UNKNOWN

        # Get metadata
        try:
            stat = path.stat()
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            permissions = oct(stat.st_mode)[-3:]
        except Exception:
            size = 0
            modified = None
            permissions = ""

        # Compute content hash for files (expensive, so only for small files)
        content_hash = None
        if ftype == FileType.FILE and size < 1024 * 100:  # < 100KB
            try:
                content = path.read_bytes()
                content_hash = hashlib.sha256(content).hexdigest()[:16]
            except Exception:
                pass

        # Get children for directories
        children = []
        if ftype == FileType.DIRECTORY:
            try:
                children = [str(p.resolve()) for p in path.iterdir()]
            except Exception:
                pass

        # Create/update node
        node = FileNode(
            path=path_str,
            type=ftype,
            size=size,
            modified=modified,
            permissions=permissions,
            content_hash=content_hash,
            children=children,
        )

        self._nodes[path_str] = node
        return node

    def observe_tree(self, root: Path, max_depth: int = 3) -> list[FileNode]:
        """
        Observe an entire directory tree up to max_depth.

        Returns all observed nodes.
        """
        nodes = []

        def _recurse(path: Path, depth: int):
            if depth > max_depth:
                return

            try:
                node = self.observe_path(path)
                nodes.append(node)

                if path.is_dir():
                    for child in path.iterdir():
                        _recurse(child, depth + 1)
            except Exception:
                pass

        _recurse(root, 0)
        return nodes

    def get_node(self, path: str | Path) -> FileNode | None:
        """Get node for a path."""
        return self._nodes.get(str(Path(path).resolve()))

    def get_changes_since(self, since: datetime) -> list[FileNode]:
        """Get all nodes modified since a given time."""
        changes = []
        for node in self._nodes.values():
            if node.modified and node.modified > since:
                changes.append(node)
        return changes

    def detect_patterns(self) -> dict[str, Any]:
        """
        Detect patterns in the filesystem.

        Returns insights about structure and organization.
        """
        patterns = {
            "total_files": 0,
            "total_dirs": 0,
            "total_size": 0,
            "file_types": defaultdict(int),
            "largest_files": [],
            "deepest_paths": [],
        }

        for node in self._nodes.values():
            if node.type == FileType.FILE:
                patterns["total_files"] += 1
                patterns["total_size"] += node.size

                # Track file extensions
                ext = Path(node.path).suffix or "no_ext"
                patterns["file_types"][ext] += 1

            elif node.type == FileType.DIRECTORY:
                patterns["total_dirs"] += 1

        # Find largest files
        files = [n for n in self._nodes.values() if n.type == FileType.FILE]
        files.sort(key=lambda n: n.size, reverse=True)
        patterns["largest_files"] = [
            {"path": n.path, "size": n.size}
            for n in files[:10]
        ]

        # Find deepest paths
        nodes_list = list(self._nodes.values())
        nodes_list.sort(key=lambda n: n.path.count('/'), reverse=True)
        patterns["deepest_paths"] = [n.path for n in nodes_list[:10]]

        return patterns

    def save(self):
        """Persist the model to disk."""
        data = {
            node.path: node.to_dict()
            for node in self._nodes.values()
        }
        with open(self.fs_model_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load the model from disk."""
        if not self.fs_model_path.exists():
            return

        try:
            with open(self.fs_model_path, 'r') as f:
                data = json.load(f)

            self._nodes = {
                path: FileNode.from_dict(node_data)
                for path, node_data in data.items()
            }
        except Exception:
            pass


# =============================================================================
# Process Model
# =============================================================================

@dataclass
class ProcessNode:
    """A running process in the system."""
    pid: int
    name: str
    cmdline: str
    status: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    parent_pid: int | None = None
    children: list[int] = field(default_factory=list)
    created: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "cmdline": self.cmdline,
            "status": self.status,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "parent_pid": self.parent_pid,
            "children": self.children,
            "created": self.created.isoformat() if self.created else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessNode":
        return cls(
            pid=data["pid"],
            name=data["name"],
            cmdline=data["cmdline"],
            status=data["status"],
            cpu_percent=data.get("cpu_percent", 0.0),
            memory_mb=data.get("memory_mb", 0.0),
            parent_pid=data.get("parent_pid"),
            children=data.get("children", []),
            created=datetime.fromisoformat(data["created"]) if data.get("created") else None,
            metadata=data.get("metadata", {}),
        )


class ProcessModel:
    """
    Model of running processes.

    Tracks what's running, resource usage, and process relationships.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.process_model_path = self.model_dir / "processes.json"

        self._processes: dict[int, ProcessNode] = {}
        self._load()

    def observe_process(self, pid: int) -> ProcessNode | None:
        """Observe a single process."""
        try:
            import psutil
            proc = psutil.Process(pid)

            node = ProcessNode(
                pid=pid,
                name=proc.name(),
                cmdline=' '.join(proc.cmdline()),
                status=proc.status(),
                cpu_percent=proc.cpu_percent(),
                memory_mb=proc.memory_info().rss / (1024 * 1024),
                parent_pid=proc.ppid() if proc.ppid() != 0 else None,
                created=datetime.fromtimestamp(proc.create_time(), tz=UTC),
            )

            self._processes[pid] = node
            return node

        except Exception:
            return None

    def observe_all(self, threshold_cpu: float = 1.0):
        """
        Observe all running processes.

        Only tracks processes using more than threshold_cpu percent CPU
        to avoid noise.
        """
        try:
            import psutil

            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    info = proc.info
                    if info['cpu_percent'] and info['cpu_percent'] > threshold_cpu:
                        self.observe_process(info['pid'])
                except Exception:
                    pass

        except Exception:
            pass

    def get_process_tree(self, root_pid: int) -> dict[int, ProcessNode]:
        """Get a process and all its descendants."""
        tree = {}

        def _recurse(pid: int):
            node = self._processes.get(pid)
            if not node:
                # Try to observe if not in model
                node = self.observe_process(pid)

            if node:
                tree[pid] = node
                for child_pid in node.children:
                    _recurse(child_pid)

        _recurse(root_pid)
        return tree

    def get_resource_summary(self) -> dict[str, Any]:
        """Get summary of resource usage."""
        summary = {
            "total_processes": len(self._processes),
            "total_cpu": sum(p.cpu_percent for p in self._processes.values()),
            "total_memory_mb": sum(p.memory_mb for p in self._processes.values()),
            "top_cpu": [],
            "top_memory": [],
        }

        # Top CPU
        by_cpu = sorted(self._processes.values(), key=lambda p: p.cpu_percent, reverse=True)
        summary["top_cpu"] = [
            {"pid": p.pid, "name": p.name, "cpu": p.cpu_percent}
            for p in by_cpu[:5]
        ]

        # Top memory
        by_mem = sorted(self._processes.values(), key=lambda p: p.memory_mb, reverse=True)
        summary["top_memory"] = [
            {"pid": p.pid, "name": p.name, "memory_mb": p.memory_mb}
            for p in by_mem[:5]
        ]

        return summary

    def save(self):
        """Persist model to disk."""
        data = {
            str(pid): node.to_dict()
            for pid, node in self._processes.items()
        }
        with open(self.process_model_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load model from disk."""
        if not self.process_model_path.exists():
            return

        try:
            with open(self.process_model_path, 'r') as f:
                data = json.load(f)

            self._processes = {
                int(pid): ProcessNode.from_dict(node_data)
                for pid, node_data in data.items()
            }
        except Exception:
            pass


# =============================================================================
# State Transition Model
# =============================================================================

@dataclass
class StateTransition:
    """A recorded state transition: (state, action) -> new_state."""
    id: str
    timestamp: datetime
    initial_state: dict[str, Any]
    action: str
    action_params: dict[str, Any]
    resulting_state: dict[str, Any]
    success: bool
    duration_ms: float
    side_effects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "initial_state": self.initial_state,
            "action": self.action,
            "action_params": self.action_params,
            "resulting_state": self.resulting_state,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "side_effects": self.side_effects,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateTransition":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            initial_state=data["initial_state"],
            action=data["action"],
            action_params=data["action_params"],
            resulting_state=data["resulting_state"],
            success=data["success"],
            duration_ms=data["duration_ms"],
            side_effects=data.get("side_effects", []),
        )


class TransitionModel:
    """
    Model of state transitions.

    Records how actions change the world state. Enables:
    - Learning action effects
    - Predicting outcomes
    - Detecting anomalies (unexpected transitions)
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.transitions_path = self.model_dir / "transitions.jsonl"

        self._transitions: list[StateTransition] = []
        self._load()

    def record_transition(
        self,
        initial_state: dict[str, Any],
        action: str,
        action_params: dict[str, Any],
        resulting_state: dict[str, Any],
        success: bool,
        duration_ms: float,
        side_effects: list[str] | None = None,
    ) -> StateTransition:
        """Record a state transition."""
        import uuid

        transition = StateTransition(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(UTC),
            initial_state=initial_state,
            action=action,
            action_params=action_params,
            resulting_state=resulting_state,
            success=success,
            duration_ms=duration_ms,
            side_effects=side_effects or [],
        )

        self._transitions.append(transition)

        # Append to file
        with open(self.transitions_path, 'a') as f:
            f.write(json.dumps(transition.to_dict(), default=str) + '\n')

        return transition

    def get_transitions_for_action(self, action: str) -> list[StateTransition]:
        """Get all recorded transitions for a specific action."""
        return [t for t in self._transitions if t.action == action]

    def predict_outcome(
        self,
        action: str,
        current_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Predict the outcome of an action based on past transitions.

        Returns predicted resulting state with confidence.
        """
        relevant = self.get_transitions_for_action(action)

        if not relevant:
            return {
                "predicted_state": None,
                "confidence": 0.0,
                "evidence_count": 0,
            }

        # Simple prediction: most common outcome
        outcomes = [t.resulting_state for t in relevant if t.success]

        if not outcomes:
            return {
                "predicted_state": None,
                "confidence": 0.0,
                "evidence_count": len(relevant),
            }

        # Return most recent outcome as prediction
        prediction = outcomes[-1]
        confidence = sum(1 for t in relevant if t.success) / len(relevant)

        return {
            "predicted_state": prediction,
            "confidence": confidence,
            "evidence_count": len(relevant),
        }

    def detect_anomaly(
        self,
        action: str,
        expected_duration_ms: float,
        actual_duration_ms: float,
        threshold: float = 2.0,
    ) -> bool:
        """
        Detect if a transition was anomalous.

        Returns True if duration exceeds threshold * expected.
        """
        return actual_duration_ms > threshold * expected_duration_ms

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about recorded transitions."""
        stats = {
            "total_transitions": len(self._transitions),
            "unique_actions": len(set(t.action for t in self._transitions)),
            "success_rate": 0.0,
            "action_counts": defaultdict(int),
            "avg_duration_by_action": {},
        }

        if self._transitions:
            stats["success_rate"] = (
                sum(1 for t in self._transitions if t.success) / len(self._transitions)
            )

        # Count actions
        for t in self._transitions:
            stats["action_counts"][t.action] += 1

        # Average duration by action
        action_durations = defaultdict(list)
        for t in self._transitions:
            action_durations[t.action].append(t.duration_ms)

        for action, durations in action_durations.items():
            stats["avg_duration_by_action"][action] = sum(durations) / len(durations)

        return stats

    def _load(self):
        """Load transitions from disk."""
        if not self.transitions_path.exists():
            return

        try:
            with open(self.transitions_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    transition = StateTransition.from_dict(data)
                    self._transitions.append(transition)

        except Exception:
            pass


# =============================================================================
# Unified World Model
# =============================================================================

class WorldModel:
    """
    Unified world model combining all subsystems.

    Provides a single interface for:
    - Observing the environment
    - Recording state changes
    - Making predictions
    - Detecting anomalies
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Subsystems
        self.filesystem = FileSystemModel(self.model_dir / "filesystem")
        self.processes = ProcessModel(self.model_dir / "processes")
        self.transitions = TransitionModel(self.model_dir / "transitions")

    def observe_environment(self, cwd: Path, observe_processes: bool = True):
        """
        Observe the current environment state.

        Updates filesystem and process models.
        """
        # Observe current directory tree
        self.filesystem.observe_tree(cwd, max_depth=2)

        # Observe processes
        if observe_processes:
            self.processes.observe_all()

    def record_action(
        self,
        action: str,
        action_params: dict[str, Any],
        success: bool,
        duration_ms: float,
    ):
        """
        Record an action and its effects.

        This builds the transition model over time.
        """
        # Snapshot initial state (simplified)
        initial_state = {
            "file_count": len(self.filesystem._nodes),
            "process_count": len(self.processes._processes),
        }

        # After action, snapshot resulting state
        resulting_state = {
            "file_count": len(self.filesystem._nodes),
            "process_count": len(self.processes._processes),
        }

        # Record transition
        self.transitions.record_transition(
            initial_state=initial_state,
            action=action,
            action_params=action_params,
            resulting_state=resulting_state,
            success=success,
            duration_ms=duration_ms,
        )

    def predict_action_outcome(
        self,
        action: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Predict the outcome of an action before executing.

        Returns prediction with confidence.
        """
        current_state = {
            "file_count": len(self.filesystem._nodes),
            "process_count": len(self.processes._processes),
        }

        return self.transitions.predict_outcome(action, current_state)

    def save_all(self):
        """Persist all models to disk."""
        self.filesystem.save()
        self.processes.save()
        # Transitions are appended incrementally

    def get_summary(self) -> dict[str, Any]:
        """Get summary of the world model."""
        fs_patterns = self.filesystem.detect_patterns()
        proc_summary = self.processes.get_resource_summary()
        trans_stats = self.transitions.get_statistics()

        return {
            "filesystem": fs_patterns,
            "processes": proc_summary,
            "transitions": trans_stats,
            "timestamp": datetime.now(UTC).isoformat(),
        }
