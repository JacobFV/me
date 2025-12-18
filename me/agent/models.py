"""
Model provider abstraction for Strands Agents.

This module creates Strands model instances from the declarative ModelConfig.
The agent's model.json file determines which LLM provider and model to use.

Supported providers:
- anthropic: Claude models (default)
- openai: GPT models
- ollama: Local models via Ollama
- bedrock: AWS Bedrock models
- litellm: Any model via LiteLLM proxy
"""

from __future__ import annotations

import os
from typing import Any

from me.agent.body import ModelConfig, ModelProvider


def build_model(config: ModelConfig) -> Any:
    """
    Build a Strands model from ModelConfig.

    The model.json file in the agent's body directory controls this:

    ```json
    {
      "provider": "anthropic",
      "model_id": "claude-sonnet-4-20250514",
      "temperature": 0.7
    }
    ```

    Returns a Strands model instance ready to use with Agent().
    """
    if config.provider == ModelProvider.ANTHROPIC:
        return _build_anthropic(config)
    elif config.provider == ModelProvider.OPENAI:
        return _build_openai(config)
    elif config.provider == ModelProvider.OLLAMA:
        return _build_ollama(config)
    elif config.provider == ModelProvider.BEDROCK:
        return _build_bedrock(config)
    elif config.provider == ModelProvider.LITELLM:
        return _build_litellm(config)
    else:
        raise ValueError(f"Unknown model provider: {config.provider}")


def _build_anthropic(config: ModelConfig) -> Any:
    """Build Anthropic/Claude model."""
    from strands.models.anthropic import AnthropicModel

    client_args: dict[str, Any] = {}
    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        client_args["api_key"] = api_key

    params: dict[str, Any] = {"temperature": config.temperature}
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop_sequences"] = config.stop_sequences
    params.update(config.extra_params)

    return AnthropicModel(
        client_args=client_args if client_args else None,
        model_id=config.model_id,
        max_tokens=config.max_tokens,
        params=params,
    )


def _build_openai(config: ModelConfig) -> Any:
    """Build OpenAI model."""
    from strands.models.openai import OpenAIModel

    client_args: dict[str, Any] = {}
    api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
    if api_key:
        client_args["api_key"] = api_key
    if config.base_url:
        client_args["base_url"] = config.base_url

    params: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    params.update(config.extra_params)

    return OpenAIModel(
        client_args=client_args if client_args else None,
        model_id=config.model_id,
        params=params,
    )


def _build_ollama(config: ModelConfig) -> Any:
    """Build Ollama model for local inference."""
    from strands.models.ollama import OllamaModel

    host = config.host or "http://localhost:11434"

    kwargs: dict[str, Any] = {
        "host": host,
        "model_id": config.model_id,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.stop_sequences:
        kwargs["stop_sequences"] = config.stop_sequences
    if config.extra_params:
        kwargs["options"] = config.extra_params

    return OllamaModel(**kwargs)


def _build_bedrock(config: ModelConfig) -> Any:
    """Build AWS Bedrock model."""
    from strands.models import BedrockModel

    kwargs: dict[str, Any] = {
        "model_id": config.model_id,
        "temperature": config.temperature,
        "streaming": True,
    }
    kwargs.update(config.extra_params)

    return BedrockModel(**kwargs)


def _build_litellm(config: ModelConfig) -> Any:
    """Build LiteLLM model (supports 100+ providers)."""
    from strands.models.litellm import LiteLLMModel

    model_config: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.top_p is not None:
        model_config["top_p"] = config.top_p
    if config.stop_sequences:
        model_config["stop"] = config.stop_sequences
    if config.api_key:
        model_config["api_key"] = config.api_key
    if config.base_url:
        model_config["api_base"] = config.base_url
    model_config.update(config.extra_params)

    return LiteLLMModel(
        model_id=config.model_id,
        model_config=model_config,
    )


def get_provider_info(provider: ModelProvider) -> dict[str, Any]:
    """Get information about a provider and common models."""
    info = {
        ModelProvider.ANTHROPIC: {
            "name": "Anthropic",
            "env_var": "ANTHROPIC_API_KEY",
            "common_models": [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-3-5-haiku-20241022",
            ],
            "docs": "https://docs.anthropic.com",
        },
        ModelProvider.OPENAI: {
            "name": "OpenAI",
            "env_var": "OPENAI_API_KEY",
            "common_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini"],
            "docs": "https://platform.openai.com/docs",
        },
        ModelProvider.OLLAMA: {
            "name": "Ollama (Local)",
            "env_var": None,
            "common_models": ["llama3.1", "llama3", "mistral", "codellama", "deepseek-r1"],
            "docs": "https://ollama.ai",
        },
        ModelProvider.BEDROCK: {
            "name": "AWS Bedrock",
            "env_var": "AWS_ACCESS_KEY_ID",
            "common_models": [
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "anthropic.claude-3-opus-20240229-v1:0",
                "amazon.nova-pro-v1:0",
            ],
            "docs": "https://aws.amazon.com/bedrock/",
        },
        ModelProvider.LITELLM: {
            "name": "LiteLLM (100+ providers)",
            "env_var": None,
            "common_models": ["gpt-4o", "claude-3-opus", "ollama/llama3"],
            "docs": "https://docs.litellm.ai",
        },
    }
    return info.get(provider, {})
