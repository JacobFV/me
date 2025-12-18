"""
Multi-Agent Communication Substrate.

Enables agents to coordinate through filesystem-backed channels, discover each other,
share world state, and achieve collective intelligence through message passing.

Philosophy: Communication IS embodiment. Agents don't just exist in isolation—
they exist in a social substrate where meaning is shared and collective understanding emerges.

Architecture:
    - Channels: Topic-based message queues (filesystem-backed)
    - AgentDirectory: Service discovery for active agents
    - SharedBlackboard: Collective world state
    - MessageProtocol: Structured message format with semantics
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Message Protocol
# =============================================================================

class MessageType(str, Enum):
    """Types of inter-agent messages."""
    INFO = "info"                    # Information sharing
    REQUEST = "request"              # Request for help/data
    RESPONSE = "response"            # Response to request
    PROPOSAL = "proposal"            # Propose action/decision
    AGREEMENT = "agreement"          # Agree to proposal
    DISAGREEMENT = "disagreement"    # Disagree with proposal
    OBSERVATION = "observation"      # Share environmental observation
    BELIEF = "belief"                # Share belief about world
    GOAL = "goal"                    # Announce goal
    COORDINATION = "coordination"    # Coordinate joint action
    ALERT = "alert"                  # Urgent notification


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Message:
    """
    A message between agents.

    Messages are the atoms of coordination. They carry semantic content,
    have types that determine how they should be processed, and include
    context for interpretation.
    """
    id: str
    sender_id: str
    sender_name: str
    channel: str
    type: MessageType
    priority: MessagePriority
    content: str
    timestamp: datetime
    reply_to: str | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "channel": self.channel,
            "type": self.type.value,
            "priority": self.priority.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "reply_to": self.reply_to,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            sender_name=data["sender_name"],
            channel=data["channel"],
            type=MessageType(data["type"]),
            priority=MessagePriority(data["priority"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reply_to=data.get("reply_to"),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )

    def to_jsonl(self) -> str:
        """Serialize to JSON line."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_jsonl(cls, line: str) -> "Message":
        """Parse from JSON line."""
        return cls.from_dict(json.loads(line))


# =============================================================================
# Channel (Topic-Based Message Queue)
# =============================================================================

class Channel:
    """
    A filesystem-backed message channel.

    Channels are append-only logs of messages. Agents can publish to channels
    and subscribe to read new messages. This is the basic primitive for coordination.
    """

    def __init__(self, channel_dir: Path, name: str, max_messages: int = 1000):
        self.name = name
        self.path = channel_dir / f"{name}.jsonl"
        self.max_messages = max_messages
        self._last_read_position: int = 0

        # Ensure channel file exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def publish(self, message: Message):
        """Publish a message to this channel."""
        with open(self.path, 'a') as f:
            f.write(message.to_jsonl() + '\n')

        # Rotate if too large
        self._maybe_rotate()

    def subscribe(self, since: datetime | None = None) -> list[Message]:
        """
        Read new messages since last read.

        If since is provided, reads from that timestamp.
        Otherwise, reads from last read position.
        """
        if not self.path.exists():
            return []

        messages = []
        with open(self.path, 'r') as f:
            lines = f.readlines()

            # Read from last position
            for i in range(self._last_read_position, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue

                try:
                    msg = Message.from_jsonl(line)

                    # Filter by timestamp if provided
                    if since and msg.timestamp < since:
                        continue

                    # Filter expired messages
                    if msg.expires_at and datetime.now(UTC) > msg.expires_at:
                        continue

                    messages.append(msg)
                except Exception:
                    pass

            self._last_read_position = len(lines)

        return messages

    def read_all(self, limit: int | None = None) -> list[Message]:
        """Read all messages in channel."""
        if not self.path.exists():
            return []

        messages = []
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = Message.from_jsonl(line)

                    # Filter expired
                    if msg.expires_at and datetime.now(UTC) > msg.expires_at:
                        continue

                    messages.append(msg)
                except Exception:
                    pass

        if limit:
            messages = messages[-limit:]

        return messages

    def _maybe_rotate(self):
        """Rotate channel if it exceeds max messages."""
        if not self.path.exists():
            return

        try:
            lines = self.path.read_text().strip().split('\n')
            if len(lines) > self.max_messages:
                # Keep last max_messages
                lines = lines[-self.max_messages:]
                self.path.write_text('\n'.join(lines) + '\n')
        except Exception:
            pass


# =============================================================================
# Agent Directory (Service Discovery)
# =============================================================================

class AgentInfo(BaseModel):
    """Information about a registered agent."""
    agent_id: str
    name: str
    generation: int
    parent_id: str | None
    registered_at: datetime
    last_heartbeat: datetime
    location: str  # cwd
    capabilities: list[str] = Field(default_factory=list)
    subscribed_channels: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentDirectory:
    """
    Registry of active agents for service discovery.

    Agents register themselves and send heartbeats. This enables:
    - Discovery: What agents are available?
    - Capabilities: What can each agent do?
    - Location: Where is each agent?
    - Lineage: Who spawned whom?
    """

    def __init__(self, bus_dir: Path):
        self.dir_path = bus_dir / "directory"
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self._heartbeat_timeout_seconds = 60

    def register(self, agent_info: AgentInfo):
        """Register an agent in the directory."""
        path = self.dir_path / f"{agent_info.agent_id}.json"
        with open(path, 'w') as f:
            json.dump(agent_info.model_dump(mode='json'), f, indent=2, default=str)

    def heartbeat(self, agent_id: str):
        """Update agent's last heartbeat timestamp."""
        path = self.dir_path / f"{agent_id}.json"
        if not path.exists():
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            data['last_heartbeat'] = datetime.now(UTC).isoformat()

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def unregister(self, agent_id: str):
        """Unregister an agent."""
        path = self.dir_path / f"{agent_id}.json"
        if path.exists():
            path.unlink()

    def get(self, agent_id: str) -> AgentInfo | None:
        """Get info for a specific agent."""
        path = self.dir_path / f"{agent_id}.json"
        if not path.exists():
            return None

        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return AgentInfo.model_validate(data)
        except Exception:
            return None

    def list_active(self) -> list[AgentInfo]:
        """List all active agents (recent heartbeat)."""
        active = []
        now = datetime.now(UTC)

        for path in self.dir_path.glob("*.json"):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                info = AgentInfo.model_validate(data)

                # Check if heartbeat is recent
                time_since = (now - info.last_heartbeat).total_seconds()
                if time_since < self._heartbeat_timeout_seconds:
                    active.append(info)
            except Exception:
                pass

        return active

    def find_by_capability(self, capability: str) -> list[AgentInfo]:
        """Find agents with a specific capability."""
        return [
            agent for agent in self.list_active()
            if capability in agent.capabilities
        ]

    def get_children(self, parent_id: str) -> list[AgentInfo]:
        """Get all children of a parent agent."""
        return [
            agent for agent in self.list_active()
            if agent.parent_id == parent_id
        ]


# =============================================================================
# Shared Blackboard (Collective World State)
# =============================================================================

class WorldFact(BaseModel):
    """A fact about the world known by agents."""
    id: str
    key: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)  # agent_ids
    timestamp: datetime
    expires_at: datetime | None = None


class SharedBlackboard:
    """
    Shared knowledge base for collective world state.

    Agents can write facts they observe and read facts written by others.
    Facts have confidence levels and sources, enabling reasoning about
    reliability and consensus.
    """

    def __init__(self, bus_dir: Path):
        self.blackboard_path = bus_dir / "blackboard.json"
        self.blackboard_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.blackboard_path.exists():
            self._write_blackboard({})

    def write_fact(self, fact: WorldFact):
        """Write a fact to the blackboard."""
        facts = self._read_blackboard()

        # Check if fact already exists
        if fact.key in facts:
            existing = WorldFact.model_validate(facts[fact.key])

            # Merge sources
            all_sources = list(set(existing.sources + fact.sources))

            # Update confidence (weighted average by source count)
            total_confidence = (existing.confidence * len(existing.sources) +
                              fact.confidence * len(fact.sources))
            new_confidence = total_confidence / len(all_sources)

            fact.sources = all_sources
            fact.confidence = new_confidence

        facts[fact.key] = fact.model_dump(mode='json')
        self._write_blackboard(facts)

    def read_fact(self, key: str) -> WorldFact | None:
        """Read a fact from the blackboard."""
        facts = self._read_blackboard()
        if key not in facts:
            return None

        try:
            fact = WorldFact.model_validate(facts[key])

            # Check expiration
            if fact.expires_at and datetime.now(UTC) > fact.expires_at:
                return None

            return fact
        except Exception:
            return None

    def query(self, pattern: str) -> list[WorldFact]:
        """Query facts matching a pattern."""
        facts = self._read_blackboard()
        results = []

        for key, data in facts.items():
            if pattern in key:
                try:
                    fact = WorldFact.model_validate(data)

                    # Filter expired
                    if fact.expires_at and datetime.now(UTC) > fact.expires_at:
                        continue

                    results.append(fact)
                except Exception:
                    pass

        return results

    def consensus(self, key: str, threshold: float = 0.7) -> Any | None:
        """
        Get consensus value for a key if confidence exceeds threshold.

        Returns None if no consensus.
        """
        fact = self.read_fact(key)
        if not fact:
            return None

        if fact.confidence >= threshold:
            return fact.value

        return None

    def _read_blackboard(self) -> dict[str, Any]:
        """Read blackboard from disk."""
        if not self.blackboard_path.exists():
            return {}

        try:
            with open(self.blackboard_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_blackboard(self, facts: dict[str, Any]):
        """Write blackboard to disk."""
        with open(self.blackboard_path, 'w') as f:
            json.dump(facts, f, indent=2, default=str)


# =============================================================================
# Agent Bus (Main Interface)
# =============================================================================

class AgentBus:
    """
    The main multi-agent communication substrate.

    Provides:
    - publish/subscribe messaging via channels
    - agent discovery via directory
    - shared world state via blackboard

    All file-backed for persistence and observability.
    """

    def __init__(self, bus_dir: Path | None = None):
        self.bus_dir = bus_dir or (Path.home() / ".me" / "bus")
        self.bus_dir.mkdir(parents=True, exist_ok=True)

        self.channels_dir = self.bus_dir / "channels"
        self.channels_dir.mkdir(parents=True, exist_ok=True)

        # Core subsystems
        self.directory = AgentDirectory(self.bus_dir)
        self.blackboard = SharedBlackboard(self.bus_dir)

        # Channel cache
        self._channels: dict[str, Channel] = {}

    def get_channel(self, name: str) -> Channel:
        """Get or create a channel."""
        if name not in self._channels:
            self._channels[name] = Channel(self.channels_dir, name)
        return self._channels[name]

    def publish(
        self,
        channel: str,
        content: str,
        sender_id: str,
        sender_name: str,
        type: MessageType = MessageType.INFO,
        priority: MessagePriority = MessagePriority.NORMAL,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Publish a message to a channel."""
        message = Message(
            id=str(uuid.uuid4())[:8],
            sender_id=sender_id,
            sender_name=sender_name,
            channel=channel,
            type=type,
            priority=priority,
            content=content,
            timestamp=datetime.now(UTC),
            reply_to=reply_to,
            metadata=metadata or {},
        )

        ch = self.get_channel(channel)
        ch.publish(message)

        return message

    def subscribe(self, channel: str, since: datetime | None = None) -> list[Message]:
        """Subscribe to a channel and read new messages."""
        ch = self.get_channel(channel)
        return ch.subscribe(since)

    def read_channel(self, channel: str, limit: int | None = None) -> list[Message]:
        """Read all messages from a channel."""
        ch = self.get_channel(channel)
        return ch.read_all(limit)

    def list_channels(self) -> list[str]:
        """List all available channels."""
        return [p.stem for p in self.channels_dir.glob("*.jsonl")]

    def register_agent(
        self,
        agent_id: str,
        name: str,
        generation: int = 0,
        parent_id: str | None = None,
        location: str = "~",
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentInfo:
        """Register an agent in the directory."""
        now = datetime.now(UTC)
        info = AgentInfo(
            agent_id=agent_id,
            name=name,
            generation=generation,
            parent_id=parent_id,
            registered_at=now,
            last_heartbeat=now,
            location=location,
            capabilities=capabilities or [],
            metadata=metadata or {},
        )
        self.directory.register(info)
        return info

    def heartbeat(self, agent_id: str):
        """Send heartbeat for an agent."""
        self.directory.heartbeat(agent_id)

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        self.directory.unregister(agent_id)

    def list_active_agents(self) -> list[AgentInfo]:
        """List all active agents."""
        return self.directory.list_active()

    def find_agents_by_capability(self, capability: str) -> list[AgentInfo]:
        """Find agents with a capability."""
        return self.directory.find_by_capability(capability)

    def write_fact(
        self,
        key: str,
        value: Any,
        agent_id: str,
        confidence: float = 1.0,
    ):
        """Write a fact to the shared blackboard."""
        fact = WorldFact(
            id=str(uuid.uuid4())[:8],
            key=key,
            value=value,
            confidence=confidence,
            sources=[agent_id],
            timestamp=datetime.now(UTC),
        )
        self.blackboard.write_fact(fact)

    def read_fact(self, key: str) -> WorldFact | None:
        """Read a fact from the blackboard."""
        return self.blackboard.read_fact(key)

    def query_facts(self, pattern: str) -> list[WorldFact]:
        """Query facts matching a pattern."""
        return self.blackboard.query(pattern)

    def get_consensus(self, key: str, threshold: float = 0.7) -> Any | None:
        """Get consensus value if confidence threshold is met."""
        return self.blackboard.consensus(key, threshold)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_default_bus() -> AgentBus:
    """Get the default agent bus (singleton)."""
    return AgentBus()
