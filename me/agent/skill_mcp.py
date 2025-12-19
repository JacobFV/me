"""
MCP Skill Server - Expose agent skills as MCP tools.

This module provides:
1. MCP tool definitions for skill operations
2. A standalone MCP server for skill management
3. Integration with the agent's skill system

The MCP server can be registered with the agent to allow external
orchestration of skill operations.
"""

from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from me.agent.skills import SkillManager, Skill, SkillMetadata


# =============================================================================
# MCP Tool Definitions
# =============================================================================

SKILL_TOOLS = [
    {
        "name": "list_skills",
        "description": "List all skills with their proficiency and state (learned/developing/atrophied)",
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["learned", "developing", "atrophied", "all"],
                    "description": "Filter by skill state",
                    "default": "all",
                },
                "active_only": {
                    "type": "boolean",
                    "description": "Only show currently active skills",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "get_skill",
        "description": "Get detailed information about a specific skill including instructions",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to retrieve",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "activate_skill",
        "description": "Activate a skill to bring it into conscious attention",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to activate",
                },
                "with_dependencies": {
                    "type": "boolean",
                    "description": "Also activate skill dependencies",
                    "default": False,
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "deactivate_skill",
        "description": "Deactivate a skill to free cognitive resources",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to deactivate",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "use_skill",
        "description": "Record skill usage with success/failure for proficiency tracking",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill used",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the skill application was successful",
                },
                "context": {
                    "type": "string",
                    "description": "Context in which skill was used",
                },
                "outcome": {
                    "type": "string",
                    "description": "Description of the outcome",
                },
            },
            "required": ["name", "success"],
        },
    },
    {
        "name": "search_skills",
        "description": "Search skills by keyword or semantic similarity",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "semantic": {
                    "type": "boolean",
                    "description": "Use semantic similarity search",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "recommend_skills",
        "description": "Get skill recommendations based on current context or goal",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Current goal or objective",
                },
                "task": {
                    "type": "string",
                    "description": "Current task description",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum recommendations",
                    "default": 5,
                },
            },
        },
    },
    {
        "name": "codify_skill",
        "description": "Create a new skill from instructions",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the new skill (hyphen-separated)",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the skill",
                },
                "instructions": {
                    "type": "string",
                    "description": "Full instructions in markdown format",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Applicable domains",
                },
            },
            "required": ["name", "description", "instructions"],
        },
    },
    {
        "name": "skill_health",
        "description": "Check skill health and atrophy warnings",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "export_skill",
        "description": "Export a skill for transfer to another agent",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of skill to export",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to write export file (optional)",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "import_skill",
        "description": "Import a skill from a package file",
        "input_schema": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to skill package file",
                },
            },
            "required": ["input_path"],
        },
    },
]


# =============================================================================
# MCP Tool Handler
# =============================================================================

class SkillMCPHandler:
    """
    Handles MCP tool calls for skill operations.

    This can be used:
    1. Directly by the agent's tool system
    2. As part of a standalone MCP server
    3. By external orchestrators
    """

    def __init__(self, manager: "SkillManager", agent_id: str | None = None):
        self.manager = manager
        self.agent_id = agent_id
        self._integration = None

    @property
    def integration(self):
        """Lazy-load integration module."""
        if self._integration is None:
            from me.agent.skill_integration import SkillIntegration
            self._integration = SkillIntegration(
                self.manager,
                self.manager.skills_dir.parent,
                self.agent_id
            )
        return self._integration

    def get_tools(self) -> list[dict[str, Any]]:
        """Get all skill MCP tool definitions."""
        return SKILL_TOOLS

    async def handle_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle a skill MCP tool call."""
        handlers = {
            "list_skills": self._list_skills,
            "get_skill": self._get_skill,
            "activate_skill": self._activate_skill,
            "deactivate_skill": self._deactivate_skill,
            "use_skill": self._use_skill,
            "search_skills": self._search_skills,
            "recommend_skills": self._recommend_skills,
            "codify_skill": self._codify_skill,
            "skill_health": self._skill_health,
            "export_skill": self._export_skill,
            "import_skill": self._import_skill,
        }

        handler = handlers.get(name)
        if not handler:
            return {"error": f"Unknown tool: {name}"}

        try:
            return await handler(arguments)
        except Exception as e:
            return {"error": str(e)}

    async def _list_skills(self, args: dict[str, Any]) -> dict[str, Any]:
        """List skills."""
        from me.agent.skills import SkillState

        state_filter = args.get("state", "all")
        active_only = args.get("active_only", False)

        if active_only:
            skills = self.manager.get_active_skills()
            return {
                "skills": [
                    {
                        "name": s.metadata.name,
                        "description": s.metadata.description,
                        "proficiency": s.metadata.proficiency,
                        "state": s.metadata.state.value,
                        "is_active": True,
                    }
                    for s in skills if s
                ]
            }

        if state_filter == "all":
            skills = self.manager.list_skills()
        else:
            state = SkillState(state_filter)
            skills = self.manager.list_skills(state)

        return {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "proficiency": s.proficiency,
                    "state": s.state.value,
                    "is_active": s.is_active,
                    "use_count": s.use_count,
                }
                for s in skills
            ]
        }

    async def _get_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get skill details."""
        name = args["name"]
        skill = self.manager.get_skill(name)

        if not skill:
            return {"error": f"Skill not found: {name}"}

        return {
            "name": skill.metadata.name,
            "description": skill.metadata.description,
            "version": skill.metadata.version,
            "proficiency": skill.metadata.proficiency,
            "state": skill.metadata.state.value,
            "is_active": skill.metadata.is_active,
            "use_count": skill.metadata.use_count,
            "success_count": skill.metadata.success_count,
            "tags": skill.metadata.tags,
            "domains": skill.metadata.domains,
            "instructions": skill.instructions,
            "steps": skill.steps,
            "scripts": skill.scripts,
        }

    async def _activate_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        """Activate a skill."""
        name = args["name"]
        with_deps = args.get("with_dependencies", False)

        if with_deps:
            activated = self.integration.activate_skill_with_deps(name)
            return {
                "activated": [s.metadata.name for s in activated],
                "count": len(activated),
            }
        else:
            skill = self.manager.activate_skill(name)
            if not skill:
                return {"error": f"Skill not found: {name}"}
            return {
                "activated": skill.metadata.name,
                "proficiency": skill.metadata.proficiency,
            }

    async def _deactivate_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        """Deactivate a skill."""
        name = args["name"]
        success = self.manager.deactivate_skill(name)
        return {"deactivated": name, "success": success}

    async def _use_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        """Record skill usage."""
        name = args["name"]
        success = args["success"]
        context = args.get("context", "")
        outcome = args.get("outcome", "")

        meta = self.manager.record_usage(name, success, context=context, outcome=outcome)
        if not meta:
            return {"error": f"Skill not found: {name}"}

        return {
            "name": name,
            "recorded": True,
            "new_proficiency": meta.proficiency,
            "state": meta.state.value,
            "total_uses": meta.use_count,
        }

    async def _search_skills(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search skills."""
        query = args["query"]
        semantic = args.get("semantic", False)
        limit = args.get("limit", 5)

        if semantic:
            from me.agent.skill_integration import semantic_skill_search
            results = semantic_skill_search(self.manager, query, limit)
            return {
                "results": [
                    {
                        "name": meta.name,
                        "description": meta.description,
                        "proficiency": meta.proficiency,
                        "similarity": score,
                    }
                    for meta, score in results
                ]
            }
        else:
            results = self.manager.search_skills(query, limit)
            return {
                "results": [
                    {
                        "name": meta.name,
                        "description": meta.description,
                        "proficiency": meta.proficiency,
                    }
                    for meta in results
                ]
            }

    async def _recommend_skills(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get skill recommendations."""
        goal = args.get("goal")
        task = args.get("task")
        limit = args.get("limit", 5)

        recs = self.integration.recommender.recommend(
            goal=goal,
            current_task=task,
            top_k=limit,
        )

        return {
            "recommendations": [
                {
                    "name": r.skill_name,
                    "score": r.score,
                    "reasons": r.reasons,
                }
                for r in recs
            ]
        }

    async def _codify_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        """Create a new skill."""
        name = args["name"]
        description = args["description"]
        instructions = args["instructions"]
        tags = args.get("tags", [])
        domains = args.get("domains", [])

        skill = self.manager.codify_skill(
            name=name,
            description=description,
            instructions=instructions,
            tags=tags,
            domains=domains,
        )

        return {
            "created": skill.metadata.name,
            "state": skill.metadata.state.value,
            "path": str(self.manager.get_skill_path(name)),
        }

    async def _skill_health(self, args: dict[str, Any]) -> dict[str, Any]:
        """Check skill health."""
        warnings = self.integration.atrophy_detector.check_atrophy()

        return {
            "warnings": [
                {
                    "skill": w.skill_name,
                    "days_since_use": w.days_since_use,
                    "proficiency": w.current_proficiency,
                    "level": w.warning_level,
                }
                for w in warnings
            ],
            "summary": self.integration.atrophy_detector.get_atrophy_summary(),
        }

    async def _export_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        """Export a skill."""
        name = args["name"]
        output_path = args.get("output_path")

        package = self.integration.transfer.export_skill(name)
        if not package:
            return {"error": f"Skill not found: {name}"}

        result = {
            "name": package.name,
            "version": package.version,
            "exported": True,
        }

        if output_path:
            path = Path(output_path)
            success = self.integration.transfer.export_to_file(name, path)
            result["file"] = str(path) if success else None

        return result

    async def _import_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        """Import a skill."""
        input_path = Path(args["input_path"])

        skill = self.integration.transfer.import_from_file(input_path)
        if not skill:
            return {"error": f"Failed to import from: {input_path}"}

        return {
            "imported": skill.metadata.name,
            "state": skill.metadata.state.value,
            "path": str(self.manager.get_skill_path(skill.metadata.name)),
        }


# =============================================================================
# Standalone MCP Server (for external use)
# =============================================================================

def create_skill_mcp_server(manager: "SkillManager", agent_id: str | None = None):
    """
    Create a standalone MCP server for skill management.

    This can be run as a separate process and registered with the agent
    or used by external orchestrators.

    Usage:
        # In agent config or external orchestrator
        mcp_servers:
          skills:
            command: python
            args: ["-m", "me.agent.skill_mcp", "--body-dir", "/path/to/agent"]
    """
    try:
        from mcp.server import Server
        from mcp.server.models import InitializationOptions
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        raise ImportError("MCP server requires 'mcp' package. Install with: pip install mcp")

    handler = SkillMCPHandler(manager, agent_id)
    server = Server("skill-server")

    @server.list_tools()
    async def list_tools():
        """List all skill tools."""
        return [
            Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["input_schema"],
            )
            for tool in handler.get_tools()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        """Handle tool calls."""
        result = await handler.handle_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return server, handler


async def run_skill_mcp_server(body_dir: Path):
    """Run the skill MCP server."""
    from me.agent.skills import SkillManager

    skills_dir = body_dir / "skills"
    manager = SkillManager(skills_dir)
    manager.initialize()

    server, handler = create_skill_mcp_server(manager)

    try:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    except ImportError:
        print("MCP server requires 'mcp' package. Install with: pip install mcp")
        raise


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Skill MCP Server")
    parser.add_argument("--body-dir", type=Path, required=True, help="Path to agent body directory")
    args = parser.parse_args()

    asyncio.run(run_skill_mcp_server(args.body_dir))
