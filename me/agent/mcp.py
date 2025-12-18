"""
MCP (Model Context Protocol) server registration system.

Allows the agent to dynamically register and unregister MCP servers
to extend its capabilities at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def to_sdk_config(self) -> dict[str, Any]:
        """Convert to Claude SDK MCP config format."""
        config: dict[str, Any] = {
            "type": "stdio",
            "command": self.command,
        }
        if self.args:
            config["args"] = self.args
        if self.env:
            config["env"] = self.env
        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            command=data["command"],
            args=data.get("args", []),
            env=data.get("env", {}),
            enabled=data.get("enabled", True),
        )


class MCPRegistry:
    """
    Registry for dynamically managing MCP servers.

    The agent can:
    - Register new MCP servers at runtime
    - Unregister servers when no longer needed
    - Persist registry state across sessions
    """

    def __init__(self, persist_path: Path | None = None):
        self._servers: dict[str, MCPServerConfig] = {}
        self._persist_path = persist_path or Path.home() / ".me" / "mcp_registry.json"
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self._persist_path.exists():
            try:
                with open(self._persist_path, "r") as f:
                    data = json.load(f)
                    for item in data.get("servers", []):
                        config = MCPServerConfig.from_dict(item)
                        self._servers[config.name] = config
            except Exception:
                pass  # Start with empty registry on error

    def _save(self) -> None:
        """Save registry to disk."""
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "servers": [s.to_dict() for s in self._servers.values()]
        }
        with open(self._persist_path, "w") as f:
            json.dump(data, f, indent=2)

    async def register(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> bool:
        """
        Register a new MCP server.

        Returns True if successful, False if name already exists.
        """
        if name in self._servers:
            # Update existing
            self._servers[name].command = command
            self._servers[name].args = args or []
            self._servers[name].env = env or {}
            self._servers[name].enabled = True
        else:
            # Create new
            self._servers[name] = MCPServerConfig(
                name=name,
                command=command,
                args=args or [],
                env=env or {},
            )

        self._save()
        return True

    async def unregister(self, name: str) -> bool:
        """
        Unregister an MCP server.

        Returns True if found and removed, False if not found.
        """
        if name not in self._servers:
            return False

        del self._servers[name]
        self._save()
        return True

    async def enable(self, name: str) -> bool:
        """Enable a registered MCP server."""
        if name not in self._servers:
            return False
        self._servers[name].enabled = True
        self._save()
        return True

    async def disable(self, name: str) -> bool:
        """Disable a registered MCP server without removing it."""
        if name not in self._servers:
            return False
        self._servers[name].enabled = False
        self._save()
        return True

    def get(self, name: str) -> MCPServerConfig | None:
        """Get a specific MCP server config."""
        return self._servers.get(name)

    def list(self) -> list[dict[str, Any]]:
        """List all registered MCP servers."""
        return [
            {
                "name": s.name,
                "command": s.command,
                "args": s.args,
                "enabled": s.enabled,
            }
            for s in self._servers.values()
        ]

    def get_configs(self) -> dict[str, Any]:
        """
        Get SDK-compatible configs for all enabled servers.

        Returns a dict mapping server names to their configs,
        suitable for passing to ClaudeAgentOptions.mcp_servers.
        """
        return {
            name: config.to_sdk_config()
            for name, config in self._servers.items()
            if config.enabled
        }

    def get_tools_list(self) -> list[str]:
        """
        Get list of tool patterns for all enabled servers.

        Returns patterns like "mcp__servername__*" for allowed_tools.
        """
        return [
            f"mcp__{name}__*"
            for name, config in self._servers.items()
            if config.enabled
        ]


# Common MCP servers that can be easily registered
COMMON_MCPS = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/"],
        "description": "File system access",
    },
    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "description": "GitHub API access",
    },
    "memory": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "description": "Persistent memory storage",
    },
    "puppeteer": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "description": "Browser automation",
    },
    "brave-search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "description": "Brave search API",
    },
    "sqlite": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite"],
        "description": "SQLite database access",
    },
}
