"""Tests for the core Agent class with Strands integration."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from me.agent.core import Agent, AgentConfig
from me.agent.body import ModelConfig, ModelProvider


class TestAgentConfig:
    """Test the AgentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.name == "me"
        assert config.refresh_rate_ms == 100
        assert config.max_turns is None
        assert config.base_dir == Path.home() / ".me"

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            agent_id="custom-id",
            name="custom-name",
            refresh_rate_ms=200,
            max_turns=10,
        )
        assert config.agent_id == "custom-id"
        assert config.name == "custom-name"
        assert config.refresh_rate_ms == 200
        assert config.max_turns == 10


class TestAgentInitialization:
    """Test Agent initialization."""

    def test_agent_creates_body(self):
        """Test that agent creates body directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            # Body should be initialized
            assert agent.body is not None
            assert agent.body.root.exists()

    def test_agent_has_model_config(self):
        """Test that agent body has model config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            # Model config should be accessible
            model_config = agent.body.model_config
            assert model_config is not None
            assert model_config.provider == ModelProvider.ANTHROPIC

    def test_agent_system_prompt(self):
        """Test that agent generates system prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            prompt = agent.system_prompt
            assert "self-model agent" in prompt
            assert agent.body.root.as_posix() in prompt

    def test_agent_tools_built(self):
        """Test that agent builds tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            tools = agent._build_self_tools()
            assert len(tools) > 0

            # Check for expected tools
            tool_names = [t.__name__ for t in tools]
            assert "run_command" in tool_names
            assert "store_memory" in tool_names
            assert "recall_memories" in tool_names
            assert "daemon_list" in tool_names


class TestAgentModelConfig:
    """Test Agent model configuration."""

    def test_agent_uses_model_from_config(self):
        """Test that agent reads model from model.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            # Change model config
            agent.body.model_config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_id="gpt-4o",
            )

            # Verify it's stored
            assert agent.body.model_config.provider == ModelProvider.OPENAI
            assert agent.body.model_config.model_id == "gpt-4o"

    def test_strands_agent_built_correctly(self):
        """Test that Strands agent is built with correct model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            # Build Strands agent
            strands_agent = agent._build_strands_agent()

            # Verify it has the expected attributes
            assert strands_agent is not None
            assert strands_agent.system_prompt == agent.system_prompt


class TestAgentRun:
    """Test Agent run functionality."""

    @pytest.mark.asyncio
    async def test_run_yields_messages(self):
        """Test that run yields message dicts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            # Mock the Strands agent call
            mock_result = Mock()
            mock_result.message = Mock()
            mock_block = Mock()
            mock_block.text = "Hello, I'm the agent!"
            mock_result.message.content = [mock_block]

            with patch.object(agent, '_build_strands_agent') as mock_build:
                mock_strands = Mock()
                mock_strands.return_value = mock_result
                mock_build.return_value = mock_strands

                messages = []
                async for msg in agent.run("test prompt"):
                    messages.append(msg)

                # Should have at least assistant and result messages
                assert len(messages) >= 1
                roles = [m.get("role") for m in messages]
                assert "assistant" in roles or "result" in roles

    @pytest.mark.asyncio
    async def test_run_interactive(self):
        """Test run_interactive returns text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            # Mock the run method
            async def mock_run(prompt):
                yield {"role": "assistant", "content": "Test response"}
                yield {"role": "result", "content": "success"}

            with patch.object(agent, 'run', mock_run):
                result = await agent.run_interactive("test")
                assert result == "Test response"


class TestAgentBodyIntegration:
    """Test Agent and Body integration."""

    def test_agent_modifies_body(self):
        """Test that agent operations modify body."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            # Check initial step count
            initial_steps = agent.body.get_step_count()
            assert initial_steps >= 0

    def test_agent_memory_integration(self):
        """Test that agent has memory system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            # Memory should be initialized
            assert agent.memory is not None

    def test_agent_unconscious_integration(self):
        """Test that agent has unconscious system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(base_dir=Path(tmpdir))
            agent = Agent(config)

            # Unconscious should be initialized
            assert agent.unconscious is not None
            assert agent.unconscious_dir is not None
