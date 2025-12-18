"""Tests for the model provider abstraction layer."""
import pytest
import tempfile
from pathlib import Path

from me.agent.body import ModelConfig, ModelProvider, BodyDirectory
from me.agent.models import build_model, get_provider_info


class TestModelConfig:
    """Test the ModelConfig Pydantic model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.provider == ModelProvider.ANTHROPIC
        assert config.model_id == "claude-sonnet-4-20250514"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_anthropic_config(self):
        """Test Anthropic configuration."""
        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-opus-4-20250514",
            temperature=0.5,
        )
        assert config.provider == ModelProvider.ANTHROPIC
        assert config.model_id == "claude-opus-4-20250514"
        assert config.temperature == 0.5

    def test_openai_config(self):
        """Test OpenAI configuration."""
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o",
            temperature=0.9,
            max_tokens=8192,
        )
        assert config.provider == ModelProvider.OPENAI
        assert config.model_id == "gpt-4o"
        assert config.temperature == 0.9
        assert config.max_tokens == 8192

    def test_ollama_config(self):
        """Test Ollama configuration with custom host."""
        config = ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_id="llama3",
            host="http://localhost:11434",
        )
        assert config.provider == ModelProvider.OLLAMA
        assert config.model_id == "llama3"
        assert config.host == "http://localhost:11434"

    def test_litellm_config(self):
        """Test LiteLLM configuration."""
        config = ModelConfig(
            provider=ModelProvider.LITELLM,
            model_id="ollama/llama3",
            base_url="http://localhost:4000",
        )
        assert config.provider == ModelProvider.LITELLM
        assert config.model_id == "ollama/llama3"
        assert config.base_url == "http://localhost:4000"

    def test_extra_params(self):
        """Test extra parameters passthrough."""
        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-sonnet-4-20250514",
            extra_params={"custom_key": "custom_value"},
        )
        assert config.extra_params == {"custom_key": "custom_value"}

    def test_stop_sequences(self):
        """Test stop sequences configuration."""
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o",
            stop_sequences=["<END>", "###"],
        )
        assert config.stop_sequences == ["<END>", "###"]

    def test_config_serialization(self):
        """Test JSON serialization."""
        config = ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_id="mistral",
            temperature=0.8,
        )
        json_data = config.model_dump()
        assert json_data["provider"] == "ollama"
        assert json_data["model_id"] == "mistral"

        # Deserialize
        restored = ModelConfig.model_validate(json_data)
        assert restored.provider == ModelProvider.OLLAMA
        assert restored.model_id == "mistral"


class TestModelProvider:
    """Test the ModelProvider enum."""

    def test_all_providers(self):
        """Test all provider values."""
        providers = list(ModelProvider)
        assert ModelProvider.ANTHROPIC in providers
        assert ModelProvider.OPENAI in providers
        assert ModelProvider.OLLAMA in providers
        assert ModelProvider.BEDROCK in providers
        assert ModelProvider.LITELLM in providers

    def test_provider_values(self):
        """Test provider string values."""
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.OLLAMA.value == "ollama"
        assert ModelProvider.BEDROCK.value == "bedrock"
        assert ModelProvider.LITELLM.value == "litellm"


class TestBuildModel:
    """Test the build_model factory function."""

    def test_build_anthropic_model(self):
        """Test building Anthropic model."""
        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-sonnet-4-20250514",
        )
        model = build_model(config)
        # Check it's the right type
        assert model is not None
        assert "AnthropicModel" in type(model).__name__

    def test_build_openai_model(self):
        """Test building OpenAI model."""
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o",
        )
        model = build_model(config)
        assert model is not None
        assert "OpenAIModel" in type(model).__name__

    def test_build_ollama_model(self):
        """Test building Ollama model."""
        pytest.importorskip("ollama")
        config = ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_id="llama3",
            host="http://localhost:11434",
        )
        model = build_model(config)
        assert model is not None
        assert "OllamaModel" in type(model).__name__

    def test_build_litellm_model(self):
        """Test building LiteLLM model."""
        pytest.importorskip("litellm")
        config = ModelConfig(
            provider=ModelProvider.LITELLM,
            model_id="gpt-4o",
        )
        model = build_model(config)
        assert model is not None
        assert "LiteLLMModel" in type(model).__name__


class TestGetProviderInfo:
    """Test the provider info function."""

    def test_anthropic_info(self):
        """Test Anthropic provider info."""
        info = get_provider_info(ModelProvider.ANTHROPIC)
        assert info["name"] == "Anthropic"
        assert info["env_var"] == "ANTHROPIC_API_KEY"
        assert "claude-sonnet-4-20250514" in info["common_models"]

    def test_openai_info(self):
        """Test OpenAI provider info."""
        info = get_provider_info(ModelProvider.OPENAI)
        assert info["name"] == "OpenAI"
        assert info["env_var"] == "OPENAI_API_KEY"
        assert "gpt-4o" in info["common_models"]

    def test_ollama_info(self):
        """Test Ollama provider info."""
        info = get_provider_info(ModelProvider.OLLAMA)
        assert info["name"] == "Ollama (Local)"
        assert info["env_var"] is None
        assert "llama3" in info["common_models"]


class TestBodyDirectoryModelConfig:
    """Test model config integration with BodyDirectory."""

    def test_model_config_in_body(self):
        """Test that model config is part of body directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            agent_id = "test-agent"

            body = BodyDirectory(base_dir, agent_id)
            body.initialize(name="test")

            # Check default model config
            model_config = body.model_config
            assert model_config is not None
            assert model_config.provider == ModelProvider.ANTHROPIC

    def test_model_config_persistence(self):
        """Test that model config persists to model.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            agent_id = "test-agent"

            body = BodyDirectory(base_dir, agent_id)
            body.initialize(name="test")

            # Modify and save
            new_config = ModelConfig(
                provider=ModelProvider.OLLAMA,
                model_id="llama3",
                host="http://localhost:11434",
            )
            body.model_config = new_config

            # Check file exists
            model_file = base_dir / "agents" / agent_id / "model.json"
            assert model_file.exists()

            # Reload and verify
            body2 = BodyDirectory(base_dir, agent_id)
            body2.initialize()
            assert body2.model_config.provider == ModelProvider.OLLAMA
            assert body2.model_config.model_id == "llama3"

    def test_model_config_update(self):
        """Test updating model config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            agent_id = "test-agent"

            body = BodyDirectory(base_dir, agent_id)
            body.initialize(name="test")

            # Update
            body.model_config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_id="gpt-4o-mini",
            )

            # Verify
            assert body.model_config.provider == ModelProvider.OPENAI
            assert body.model_config.model_id == "gpt-4o-mini"
