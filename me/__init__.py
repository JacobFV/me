"""
me - Self-model agent architecture

Goal: Upload "me" into a computer through a self-aware agent system.
"""

__version__ = "0.1.0"

from me.agent.core import Agent, AgentConfig
from me.agent.body import BodyDirectory
from me.agent.memory import Memory
from me.agent.mcp import MCPRegistry

__all__ = ["Agent", "AgentConfig", "BodyDirectory", "Memory", "MCPRegistry"]
