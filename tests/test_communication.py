"""Tests for multi-agent communication substrate."""

import tempfile
from datetime import datetime, UTC, timedelta
from pathlib import Path

import pytest

from me.agent.communication import (
    AgentBus,
    AgentDirectory,
    AgentInfo,
    Channel,
    Message,
    MessageType,
    MessagePriority,
    SharedBlackboard,
    WorldFact,
)


# =============================================================================
# Message Tests
# =============================================================================

class TestMessage:
    def test_create_message(self):
        msg = Message(
            id="msg1",
            sender_id="agent1",
            sender_name="Alice",
            channel="test-channel",
            type=MessageType.INFO,
            priority=MessagePriority.NORMAL,
            content="Test message",
            timestamp=datetime.now(UTC),
        )
        assert msg.id == "msg1"
        assert msg.sender_id == "agent1"
        assert msg.type == MessageType.INFO

    def test_message_serialization(self):
        msg = Message(
            id="msg1",
            sender_id="agent1",
            sender_name="Alice",
            channel="test",
            type=MessageType.REQUEST,
            priority=MessagePriority.HIGH,
            content="Help needed",
            timestamp=datetime.now(UTC),
        )

        # Serialize to dict
        data = msg.to_dict()
        assert data["id"] == "msg1"
        assert data["type"] == "request"

        # Deserialize from dict
        msg2 = Message.from_dict(data)
        assert msg2.id == msg.id
        assert msg2.type == msg.type

    def test_message_jsonl(self):
        msg = Message(
            id="msg1",
            sender_id="agent1",
            sender_name="Alice",
            channel="test",
            type=MessageType.INFO,
            priority=MessagePriority.NORMAL,
            content="Test",
            timestamp=datetime.now(UTC),
        )

        jsonl = msg.to_jsonl()
        msg2 = Message.from_jsonl(jsonl)

        assert msg2.id == msg.id
        assert msg2.content == msg.content


# =============================================================================
# Channel Tests
# =============================================================================

class TestChannel:
    def test_publish_and_subscribe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            channel_dir = Path(tmpdir)
            channel = Channel(channel_dir, "test-channel")

            # Publish messages
            msg1 = Message(
                id="msg1",
                sender_id="agent1",
                sender_name="Alice",
                channel="test-channel",
                type=MessageType.INFO,
                priority=MessagePriority.NORMAL,
                content="First message",
                timestamp=datetime.now(UTC),
            )
            channel.publish(msg1)

            msg2 = Message(
                id="msg2",
                sender_id="agent2",
                sender_name="Bob",
                channel="test-channel",
                type=MessageType.INFO,
                priority=MessagePriority.NORMAL,
                content="Second message",
                timestamp=datetime.now(UTC),
            )
            channel.publish(msg2)

            # Subscribe (reads new messages)
            messages = channel.subscribe()
            assert len(messages) == 2
            assert messages[0].content == "First message"
            assert messages[1].content == "Second message"

    def test_subscribe_incremental(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            channel_dir = Path(tmpdir)
            channel = Channel(channel_dir, "test")

            # Publish first batch
            for i in range(3):
                msg = Message(
                    id=f"msg{i}",
                    sender_id="agent1",
                    sender_name="Alice",
                    channel="test",
                    type=MessageType.INFO,
                    priority=MessagePriority.NORMAL,
                    content=f"Message {i}",
                    timestamp=datetime.now(UTC),
                )
                channel.publish(msg)

            # First subscribe
            messages = channel.subscribe()
            assert len(messages) == 3

            # Publish more
            for i in range(3, 5):
                msg = Message(
                    id=f"msg{i}",
                    sender_id="agent1",
                    sender_name="Alice",
                    channel="test",
                    type=MessageType.INFO,
                    priority=MessagePriority.NORMAL,
                    content=f"Message {i}",
                    timestamp=datetime.now(UTC),
                )
                channel.publish(msg)

            # Second subscribe (should only get new messages)
            messages = channel.subscribe()
            assert len(messages) == 2
            assert messages[0].content == "Message 3"

    def test_read_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            channel_dir = Path(tmpdir)
            channel = Channel(channel_dir, "test")

            # Publish messages
            for i in range(5):
                msg = Message(
                    id=f"msg{i}",
                    sender_id="agent1",
                    sender_name="Alice",
                    channel="test",
                    type=MessageType.INFO,
                    priority=MessagePriority.NORMAL,
                    content=f"Message {i}",
                    timestamp=datetime.now(UTC),
                )
                channel.publish(msg)

            # Read all
            messages = channel.read_all()
            assert len(messages) == 5

            # Read with limit
            messages = channel.read_all(limit=3)
            assert len(messages) == 3

    def test_expired_messages_filtered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            channel_dir = Path(tmpdir)
            channel = Channel(channel_dir, "test")

            # Publish expired message
            expired = Message(
                id="expired",
                sender_id="agent1",
                sender_name="Alice",
                channel="test",
                type=MessageType.INFO,
                priority=MessagePriority.NORMAL,
                content="Expired",
                timestamp=datetime.now(UTC),
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
            channel.publish(expired)

            # Publish valid message
            valid = Message(
                id="valid",
                sender_id="agent1",
                sender_name="Alice",
                channel="test",
                type=MessageType.INFO,
                priority=MessagePriority.NORMAL,
                content="Valid",
                timestamp=datetime.now(UTC),
            )
            channel.publish(valid)

            # Subscribe should only return valid
            messages = channel.subscribe()
            assert len(messages) == 1
            assert messages[0].content == "Valid"


# =============================================================================
# Agent Directory Tests
# =============================================================================

class TestAgentDirectory:
    def test_register_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir)
            directory = AgentDirectory(bus_dir)

            # Register agent
            info = AgentInfo(
                agent_id="agent1",
                name="Alice",
                generation=0,
                parent_id=None,
                registered_at=datetime.now(UTC),
                last_heartbeat=datetime.now(UTC),
                location="/home/alice",
                capabilities=["coding", "analysis"],
            )
            directory.register(info)

            # Get agent
            retrieved = directory.get("agent1")
            assert retrieved is not None
            assert retrieved.name == "Alice"
            assert "coding" in retrieved.capabilities

    def test_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir)
            directory = AgentDirectory(bus_dir)

            # Register agent
            info = AgentInfo(
                agent_id="agent1",
                name="Alice",
                generation=0,
                parent_id=None,
                registered_at=datetime.now(UTC),
                last_heartbeat=datetime.now(UTC) - timedelta(seconds=30),
                location="/home/alice",
            )
            directory.register(info)

            # Send heartbeat
            directory.heartbeat("agent1")

            # Check updated
            updated = directory.get("agent1")
            assert updated is not None
            assert (datetime.now(UTC) - updated.last_heartbeat).total_seconds() < 5

    def test_list_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir)
            directory = AgentDirectory(bus_dir)

            # Register active agent
            active = AgentInfo(
                agent_id="active",
                name="Active",
                generation=0,
                parent_id=None,
                registered_at=datetime.now(UTC),
                last_heartbeat=datetime.now(UTC),
                location="/home",
            )
            directory.register(active)

            # Register stale agent (old heartbeat)
            stale = AgentInfo(
                agent_id="stale",
                name="Stale",
                generation=0,
                parent_id=None,
                registered_at=datetime.now(UTC) - timedelta(minutes=5),
                last_heartbeat=datetime.now(UTC) - timedelta(minutes=5),
                location="/home",
            )
            directory.register(stale)

            # List active should only return recent
            active_agents = directory.list_active()
            assert len(active_agents) == 1
            assert active_agents[0].agent_id == "active"

    def test_find_by_capability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir)
            directory = AgentDirectory(bus_dir)

            # Register agents with different capabilities
            for i, caps in enumerate([["coding"], ["analysis"], ["coding", "testing"]]):
                info = AgentInfo(
                    agent_id=f"agent{i}",
                    name=f"Agent{i}",
                    generation=0,
                    parent_id=None,
                    registered_at=datetime.now(UTC),
                    last_heartbeat=datetime.now(UTC),
                    location="/home",
                    capabilities=caps,
                )
                directory.register(info)

            # Find coders
            coders = directory.find_by_capability("coding")
            assert len(coders) == 2

            # Find analysts
            analysts = directory.find_by_capability("analysis")
            assert len(analysts) == 1

    def test_get_children(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir)
            directory = AgentDirectory(bus_dir)

            # Register parent
            parent = AgentInfo(
                agent_id="parent",
                name="Parent",
                generation=0,
                parent_id=None,
                registered_at=datetime.now(UTC),
                last_heartbeat=datetime.now(UTC),
                location="/home",
            )
            directory.register(parent)

            # Register children
            for i in range(3):
                child = AgentInfo(
                    agent_id=f"child{i}",
                    name=f"Child{i}",
                    generation=1,
                    parent_id="parent",
                    registered_at=datetime.now(UTC),
                    last_heartbeat=datetime.now(UTC),
                    location="/home",
                )
                directory.register(child)

            # Get children
            children = directory.get_children("parent")
            assert len(children) == 3


# =============================================================================
# Shared Blackboard Tests
# =============================================================================

class TestSharedBlackboard:
    def test_write_and_read_fact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir)
            blackboard = SharedBlackboard(bus_dir)

            # Write fact
            fact = WorldFact(
                id="fact1",
                key="weather",
                value="sunny",
                confidence=0.9,
                sources=["agent1"],
                timestamp=datetime.now(UTC),
            )
            blackboard.write_fact(fact)

            # Read fact
            retrieved = blackboard.read_fact("weather")
            assert retrieved is not None
            assert retrieved.value == "sunny"
            assert retrieved.confidence == 0.9

    def test_fact_merging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir)
            blackboard = SharedBlackboard(bus_dir)

            # First agent writes
            fact1 = WorldFact(
                id="fact1",
                key="temperature",
                value=72,
                confidence=0.8,
                sources=["agent1"],
                timestamp=datetime.now(UTC),
            )
            blackboard.write_fact(fact1)

            # Second agent writes same key
            fact2 = WorldFact(
                id="fact2",
                key="temperature",
                value=72,
                confidence=0.9,
                sources=["agent2"],
                timestamp=datetime.now(UTC),
            )
            blackboard.write_fact(fact2)

            # Read should show merged sources and updated confidence
            merged = blackboard.read_fact("temperature")
            assert merged is not None
            assert len(merged.sources) == 2
            assert "agent1" in merged.sources
            assert "agent2" in merged.sources
            # Confidence should be weighted average
            assert merged.confidence > 0.8

    def test_query_facts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir)
            blackboard = SharedBlackboard(bus_dir)

            # Write multiple facts
            for i in range(5):
                fact = WorldFact(
                    id=f"fact{i}",
                    key=f"sensor_{i}_value",
                    value=i * 10,
                    confidence=0.9,
                    sources=["agent1"],
                    timestamp=datetime.now(UTC),
                )
                blackboard.write_fact(fact)

            # Query by pattern
            sensor_facts = blackboard.query("sensor_")
            assert len(sensor_facts) == 5

    def test_consensus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir)
            blackboard = SharedBlackboard(bus_dir)

            # High confidence fact
            high_conf = WorldFact(
                id="high",
                key="consensus_test",
                value="agreed",
                confidence=0.9,
                sources=["agent1", "agent2", "agent3"],
                timestamp=datetime.now(UTC),
            )
            blackboard.write_fact(high_conf)

            # Should reach consensus
            consensus = blackboard.consensus("consensus_test", threshold=0.8)
            assert consensus == "agreed"

            # Low confidence fact
            low_conf = WorldFact(
                id="low",
                key="uncertain",
                value="maybe",
                confidence=0.5,
                sources=["agent1"],
                timestamp=datetime.now(UTC),
            )
            blackboard.write_fact(low_conf)

            # Should not reach consensus
            no_consensus = blackboard.consensus("uncertain", threshold=0.8)
            assert no_consensus is None


# =============================================================================
# Agent Bus Integration Tests
# =============================================================================

class TestAgentBus:
    def test_bus_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir) / "bus"
            bus = AgentBus(bus_dir)

            assert bus.bus_dir.exists()
            assert bus.channels_dir.exists()
            assert bus.directory is not None
            assert bus.blackboard is not None

    def test_publish_subscribe_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir) / "bus"
            bus = AgentBus(bus_dir)

            # Publish message
            msg = bus.publish(
                channel="test-channel",
                content="Hello agents!",
                sender_id="agent1",
                sender_name="Alice",
                type=MessageType.INFO,
            )

            assert msg.id is not None
            assert msg.content == "Hello agents!"

            # Subscribe
            messages = bus.subscribe("test-channel")
            assert len(messages) == 1
            assert messages[0].content == "Hello agents!"

    def test_agent_registration_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir) / "bus"
            bus = AgentBus(bus_dir)

            # Register agent
            info = bus.register_agent(
                agent_id="agent1",
                name="Alice",
                capabilities=["coding", "testing"],
            )

            assert info.agent_id == "agent1"

            # List active
            active = bus.list_active_agents()
            assert len(active) == 1

            # Find by capability
            coders = bus.find_agents_by_capability("coding")
            assert len(coders) == 1

    def test_blackboard_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir) / "bus"
            bus = AgentBus(bus_dir)

            # Write fact
            bus.write_fact(
                key="system_status",
                value="operational",
                agent_id="agent1",
                confidence=0.95,
            )

            # Read fact
            fact = bus.read_fact("system_status")
            assert fact is not None
            assert fact.value == "operational"

            # Query facts
            facts = bus.query_facts("system_")
            assert len(facts) == 1

    def test_multi_agent_coordination(self):
        """Test multiple agents coordinating via bus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir) / "bus"
            bus = AgentBus(bus_dir)

            # Register multiple agents
            for i in range(3):
                bus.register_agent(
                    agent_id=f"agent{i}",
                    name=f"Agent{i}",
                    capabilities=["coordination"],
                )

            # Agent 0 proposes action
            bus.publish(
                channel="coordination",
                content="Propose: Deploy to production",
                sender_id="agent0",
                sender_name="Agent0",
                type=MessageType.PROPOSAL,
                priority=MessagePriority.HIGH,
            )

            # Agents 1 and 2 agree
            for i in [1, 2]:
                bus.publish(
                    channel="coordination",
                    content="I agree with the proposal",
                    sender_id=f"agent{i}",
                    sender_name=f"Agent{i}",
                    type=MessageType.AGREEMENT,
                )

            # Read coordination messages
            messages = bus.read_channel("coordination")
            assert len(messages) == 3

            proposal = [m for m in messages if m.type == MessageType.PROPOSAL]
            agreements = [m for m in messages if m.type == MessageType.AGREEMENT]

            assert len(proposal) == 1
            assert len(agreements) == 2

    def test_list_channels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = Path(tmpdir) / "bus"
            bus = AgentBus(bus_dir)

            # Create multiple channels
            for channel_name in ["alerts", "coordination", "observations"]:
                bus.publish(
                    channel=channel_name,
                    content="Test",
                    sender_id="agent1",
                    sender_name="Alice",
                )

            # List channels
            channels = bus.list_channels()
            assert len(channels) == 3
            assert "alerts" in channels
            assert "coordination" in channels
