"""
Embodied Skills System.

Skills are learned capabilities that become part of the agent's body.
They are not external plugins - they are organic extensions that the
agent acquires, develops, and can let atrophy.

Key principles:
1. Skills are part of the body directory (like limbs)
2. Progressive disclosure: metadata loads at startup, full content on activation
3. Proficiency emerges from usage, not declaration
4. Skill acquisition changes character
5. Skills can define perception daemons that run when active

Directory structure:
    ~/.me/agents/<id>/skills/
    ├── index.json          # Skill metadata cache
    ├── learned/            # Mastered skills
    │   └── <skill-name>/
    │       ├── SKILL.md    # Frontmatter + instructions
    │       ├── scripts/    # Optional executables
    │       └── references/ # Optional reference docs
    ├── developing/         # Skills being learned
    └── atrophied/          # Unused skills (can be relearned)

SKILL.md format (AgentSkills spec):
    ---
    name: git-workflow
    description: Git branching and merging workflow
    version: 1.0.0
    tags: [git, version-control]
    daemons:
      - name: git-status-monitor
        trigger: on_change
        sources: ["{cwd}/.git/index"]
        prompt: "Summarize git status changes"
    ---

    ## Overview
    Instructions for this skill...

    ## Steps
    1. First step
    2. Second step
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, UTC
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# =============================================================================
# Skill State and Types
# =============================================================================

class SkillState(str, Enum):
    """Lifecycle state of a skill - where it lives in the body."""
    DEVELOPING = "developing"  # Being learned, low proficiency
    LEARNED = "learned"        # Mastered, part of identity
    ATROPHIED = "atrophied"    # Unused, proficiency decaying


class SkillType(str, Enum):
    """Types of skills."""
    ATOMIC = "atomic"          # Single focused capability
    COMPOSITE = "composite"    # Composed of other skills
    ABSTRACT = "abstract"      # Pattern that can be instantiated
    REFLEX = "reflex"          # Automatic response to stimulus


# =============================================================================
# Skill Metadata (Always Loaded - Unconscious Awareness)
# =============================================================================

class SkillMetadata(BaseModel):
    """
    Minimal skill metadata - always loaded (like sensor config).

    This is the "unconscious" awareness of the skill - the agent
    knows it exists without needing to load full instructions.
    """
    name: str
    description: str = ""
    version: str = "1.0.0"
    tags: list[str] = Field(default_factory=list)
    skill_type: SkillType = SkillType.ATOMIC

    # State and proficiency
    state: SkillState = SkillState.DEVELOPING
    proficiency: float = 0.0  # 0-1, computed from usage

    # Usage tracking
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Activation state
    is_active: bool = False
    activated_at: datetime | None = None

    # Daemon definitions (from frontmatter)
    daemons: list[dict[str, Any]] = Field(default_factory=list)

    # Learning metadata
    learned_from: str | None = None  # "external", "codified", "inherited", "extracted"
    parent_skill: str | None = None  # For skill evolution
    extracted_from: list[str] = Field(default_factory=list)  # Episode IDs

    # Domains where skill applies
    domains: list[str] = Field(default_factory=list)

    # Preconditions and postconditions
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)

    # Embedding for semantic matching
    embedding: list[float] = Field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Compute success rate from usage history."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    @property
    def is_mastered(self) -> bool:
        """Skill is mastered when proficiency exceeds threshold."""
        return self.proficiency >= 0.8

    @property
    def is_atrophying(self) -> bool:
        """Skill atrophies when unused for extended period."""
        if self.last_used is None:
            return False
        days_since_use = (datetime.now(UTC) - self.last_used).days
        return days_since_use > 30

    def compute_proficiency(self) -> float:
        """
        Compute proficiency from usage patterns.

        Proficiency emerges from:
        - Success rate (primary factor)
        - Use count (experience)
        - Recency (skills decay without use)
        """
        if self.use_count == 0:
            return 0.0

        # Base proficiency from success rate
        base = self.success_rate

        # Experience bonus (diminishing returns)
        experience_bonus = min(0.2, self.use_count * 0.02)

        # Recency decay
        recency_factor = 1.0
        if self.last_used:
            days_since = (datetime.now(UTC) - self.last_used).days
            if days_since > 7:
                recency_factor = max(0.5, 1.0 - (days_since - 7) * 0.01)

        proficiency = (base + experience_bonus) * recency_factor
        return min(1.0, max(0.0, proficiency))

    def is_applicable(self, context: dict[str, Any]) -> bool:
        """Check if skill is applicable in given context."""
        # Check domains
        if self.domains:
            context_domain = context.get("domain", "")
            if context_domain and context_domain not in self.domains:
                return False

        # Check preconditions (simplified - just check for key presence)
        for precond in self.preconditions:
            if precond.endswith("_available"):
                key = precond.replace("_available", "")
                if key not in context:
                    return False

        return True


# =============================================================================
# Full Skill (Loaded on Activation - Conscious Attention)
# =============================================================================

class Skill(BaseModel):
    """
    Full skill - loaded on activation (conscious attention).

    Contains the complete instructions and resources needed
    to exercise the skill.
    """
    metadata: SkillMetadata
    instructions: str = ""  # Full SKILL.md content (after frontmatter)

    # File paths (relative to skill directory)
    scripts: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    assets: list[str] = Field(default_factory=list)

    # Parsed sections from instructions
    sections: dict[str, str] = Field(default_factory=dict)

    # Execution steps (parsed from instructions or explicit)
    steps: list[str] = Field(default_factory=list)

    # Sub-skills for composite skills
    sub_skills: list[str] = Field(default_factory=list)

    # Parameters for skill execution
    parameters: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_skill_md(cls, skill_path: Path) -> "Skill":
        """Parse a SKILL.md file into a Skill object."""
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            raise ValueError(f"No SKILL.md found at {skill_path}")

        content = skill_md.read_text()

        # Parse YAML frontmatter
        frontmatter, body = parse_frontmatter(content)

        # Build metadata
        metadata = SkillMetadata(
            name=frontmatter.get("name", skill_path.name),
            description=frontmatter.get("description", ""),
            version=frontmatter.get("version", "1.0.0"),
            tags=frontmatter.get("tags", []),
            daemons=frontmatter.get("daemons", []),
            domains=frontmatter.get("domains", []),
            preconditions=frontmatter.get("preconditions", []),
            postconditions=frontmatter.get("postconditions", []),
            skill_type=SkillType(frontmatter.get("type", "atomic")),
        )

        # Find scripts, references, assets
        scripts = []
        references = []
        assets = []

        scripts_dir = skill_path / "scripts"
        if scripts_dir.exists():
            scripts = [str(f.relative_to(skill_path)) for f in scripts_dir.iterdir() if f.is_file()]

        refs_dir = skill_path / "references"
        if refs_dir.exists():
            references = [str(f.relative_to(skill_path)) for f in refs_dir.iterdir() if f.is_file()]

        assets_dir = skill_path / "assets"
        if assets_dir.exists():
            assets = [str(f.relative_to(skill_path)) for f in assets_dir.iterdir() if f.is_file()]

        # Parse sections from body
        sections = parse_sections(body)

        # Extract steps from sections or frontmatter
        steps = frontmatter.get("steps", [])
        if not steps and "steps" in sections:
            steps = parse_numbered_list(sections["steps"])

        # Get sub-skills for composite
        sub_skills = frontmatter.get("sub_skills", [])

        # Get parameters
        parameters = frontmatter.get("parameters", {})

        return cls(
            metadata=metadata,
            instructions=body,
            scripts=scripts,
            references=references,
            assets=assets,
            sections=sections,
            steps=steps,
            sub_skills=sub_skills,
            parameters=parameters,
        )

    def to_prompt_context(self) -> str:
        """Format skill for injection into agent prompt."""
        lines = [
            f"## Skill: {self.metadata.name}",
            f"*Proficiency: {self.metadata.proficiency:.0%}*",
            "",
        ]

        if self.metadata.description:
            lines.extend([self.metadata.description, ""])

        if self.instructions:
            lines.append(self.instructions)

        if self.scripts:
            lines.extend([
                "",
                "### Available Scripts",
                *[f"- `{s}`" for s in self.scripts],
            ])

        return "\n".join(lines)

    def get_execution_instructions(self) -> str:
        """Get execution-focused instructions."""
        if self.steps:
            return "\n".join(f"{i+1}. {step}" for i, step in enumerate(self.steps))
        if "steps" in self.sections:
            return self.sections["steps"]
        return self.instructions


# =============================================================================
# Skill Index (Always Loaded)
# =============================================================================

class SkillIndex(BaseModel):
    """
    Index of all skills - always loaded (like sensors.json).

    This provides the agent with awareness of all its skills
    without loading full instructions.
    """
    skills: dict[str, SkillMetadata] = Field(default_factory=dict)
    active_skills: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def get_by_state(self, state: SkillState) -> list[SkillMetadata]:
        """Get all skills in a particular state."""
        return [s for s in self.skills.values() if s.state == state]

    def get_active(self) -> list[SkillMetadata]:
        """Get currently active skills."""
        return [self.skills[name] for name in self.active_skills if name in self.skills]

    def search(self, query: str) -> list[SkillMetadata]:
        """Search skills by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for skill in self.skills.values():
            score = 0
            if query_lower in skill.name.lower():
                score += 3
            if query_lower in skill.description.lower():
                score += 2
            if any(query_lower in tag.lower() for tag in skill.tags):
                score += 1
            if any(query_lower in domain.lower() for domain in skill.domains):
                score += 1
            if score > 0:
                results.append((skill, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in results]


# =============================================================================
# Skill Usage Record
# =============================================================================

class SkillUsageRecord(BaseModel):
    """Record of a skill being used."""
    skill_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    success: bool
    duration_seconds: float | None = None
    context: str = ""  # What was the agent doing?
    outcome: str = ""  # What happened?
    reward: float = 0.0


# =============================================================================
# Skill Manager
# =============================================================================

class SkillManager:
    """
    Manages the agent's skills directory.

    The SkillManager treats skills as part of the agent's body:
    - Skills are files in the body directory
    - Metadata is cached for quick access
    - Full skills load on activation
    - Usage updates proficiency and can trigger state transitions
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.index_path = skills_dir / "index.json"
        self._index: SkillIndex | None = None
        self._loaded_skills: dict[str, Skill] = {}

    def initialize(self) -> None:
        """Initialize skills directory structure."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        (self.skills_dir / "learned").mkdir(exist_ok=True)
        (self.skills_dir / "developing").mkdir(exist_ok=True)
        (self.skills_dir / "atrophied").mkdir(exist_ok=True)

        if not self.index_path.exists():
            self._save_index(SkillIndex())

    @property
    def index(self) -> SkillIndex:
        """Load or return cached index."""
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _load_index(self) -> SkillIndex:
        """Load skill index from file."""
        if not self.index_path.exists():
            return SkillIndex()
        try:
            data = json.loads(self.index_path.read_text())
            return SkillIndex.model_validate(data)
        except Exception:
            return SkillIndex()

    def _save_index(self, index: SkillIndex) -> None:
        """Save skill index to file."""
        index.updated_at = datetime.now(UTC)
        self.index_path.write_text(
            json.dumps(index.model_dump(mode="json"), indent=2, default=str)
        )
        self._index = index

    def refresh_index(self) -> SkillIndex:
        """
        Scan skills directories and rebuild index.

        This discovers new skills and updates metadata from SKILL.md files.
        """
        index = self.index

        for state in SkillState:
            state_dir = self.skills_dir / state.value
            if not state_dir.exists():
                continue

            for skill_dir in state_dir.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue

                try:
                    skill = Skill.from_skill_md(skill_dir)

                    # Preserve usage stats if skill already known
                    if skill.metadata.name in index.skills:
                        existing = index.skills[skill.metadata.name]
                        skill.metadata.use_count = existing.use_count
                        skill.metadata.success_count = existing.success_count
                        skill.metadata.failure_count = existing.failure_count
                        skill.metadata.last_used = existing.last_used
                        skill.metadata.created_at = existing.created_at
                        skill.metadata.is_active = existing.is_active
                        skill.metadata.activated_at = existing.activated_at
                        skill.metadata.learned_from = existing.learned_from
                        skill.metadata.extracted_from = existing.extracted_from
                        skill.metadata.embedding = existing.embedding

                    skill.metadata.state = state
                    skill.metadata.proficiency = skill.metadata.compute_proficiency()
                    index.skills[skill.metadata.name] = skill.metadata

                except Exception as e:
                    # Log but don't fail on bad skill files
                    print(f"Warning: Could not parse skill at {skill_dir}: {e}")

        self._save_index(index)
        return index

    def get_skill(self, name: str) -> Skill | None:
        """
        Load full skill by name.

        This is "conscious attention" - loading the full instructions.
        """
        if name in self._loaded_skills:
            return self._loaded_skills[name]

        # Find skill directory
        for state in SkillState:
            skill_dir = self.skills_dir / state.value / name
            if skill_dir.exists() and (skill_dir / "SKILL.md").exists():
                skill = Skill.from_skill_md(skill_dir)

                # Merge with index metadata (preserves usage stats and runtime state)
                if name in self.index.skills:
                    index_meta = self.index.skills[name]
                    skill.metadata.use_count = index_meta.use_count
                    skill.metadata.success_count = index_meta.success_count
                    skill.metadata.failure_count = index_meta.failure_count
                    skill.metadata.last_used = index_meta.last_used
                    skill.metadata.proficiency = index_meta.proficiency
                    skill.metadata.state = index_meta.state
                    skill.metadata.is_active = index_meta.is_active
                    skill.metadata.activated_at = index_meta.activated_at
                    skill.metadata.embedding = index_meta.embedding
                    skill.metadata.learned_from = index_meta.learned_from
                    skill.metadata.extracted_from = index_meta.extracted_from
                    skill.metadata.created_at = index_meta.created_at

                self._loaded_skills[name] = skill
                return skill

        return None

    def get_skill_path(self, name: str) -> Path | None:
        """Get the filesystem path to a skill directory."""
        for state in SkillState:
            skill_dir = self.skills_dir / state.value / name
            if skill_dir.exists():
                return skill_dir
        return None

    def activate_skill(self, name: str) -> Skill | None:
        """
        Activate a skill - bring it into conscious attention.

        Returns the full skill with instructions loaded.
        """
        skill = self.get_skill(name)
        if not skill:
            return None

        # Update index
        index = self.index
        if name in index.skills:
            index.skills[name].is_active = True
            index.skills[name].activated_at = datetime.now(UTC)
        if name not in index.active_skills:
            index.active_skills.append(name)
        self._save_index(index)

        # Update the skill's metadata to reflect activation
        skill.metadata.is_active = True
        skill.metadata.activated_at = index.skills[name].activated_at if name in index.skills else datetime.now(UTC)

        return skill

    def deactivate_skill(self, name: str) -> bool:
        """Deactivate a skill - remove from conscious attention."""
        index = self.index
        if name in index.skills:
            index.skills[name].is_active = False
            index.skills[name].activated_at = None
        if name in index.active_skills:
            index.active_skills.remove(name)
        self._save_index(index)

        # Remove from loaded cache
        if name in self._loaded_skills:
            del self._loaded_skills[name]

        return True

    def record_usage(
        self,
        name: str,
        success: bool,
        duration_seconds: float | None = None,
        context: str = "",
        outcome: str = "",
        reward: float = 0.0,
    ) -> SkillMetadata | None:
        """
        Record skill usage and update proficiency.

        This is how proficiency emerges - from actual usage patterns.
        """
        index = self.index
        if name not in index.skills:
            return None

        meta = index.skills[name]
        meta.use_count += 1
        if success:
            meta.success_count += 1
        else:
            meta.failure_count += 1
        meta.last_used = datetime.now(UTC)

        # Recompute proficiency
        meta.proficiency = meta.compute_proficiency()

        # Check for state transitions
        self._check_state_transition(meta)

        self._save_index(index)

        # Save usage record to skill's history
        self._save_usage_record(name, SkillUsageRecord(
            skill_name=name,
            success=success,
            duration_seconds=duration_seconds,
            context=context,
            outcome=outcome,
            reward=reward,
        ))

        # Update loaded skill if cached
        if name in self._loaded_skills:
            self._loaded_skills[name].metadata = meta

        return meta

    def _check_state_transition(self, meta: SkillMetadata) -> None:
        """Check if skill should transition states."""
        old_state = meta.state
        new_state = old_state

        if meta.state == SkillState.DEVELOPING:
            if meta.is_mastered:
                new_state = SkillState.LEARNED
        elif meta.state == SkillState.LEARNED:
            if meta.is_atrophying:
                new_state = SkillState.ATROPHIED
        elif meta.state == SkillState.ATROPHIED:
            # Can be relearned faster
            if meta.proficiency > 0.5:
                new_state = SkillState.LEARNED
            elif meta.use_count > 0 and meta.last_used:
                # Recent use brings it back to developing
                days_since = (datetime.now(UTC) - meta.last_used).days
                if days_since < 7:
                    new_state = SkillState.DEVELOPING

        if new_state != old_state:
            self._move_skill(meta.name, old_state, new_state)
            meta.state = new_state

    def _move_skill(self, name: str, from_state: SkillState, to_state: SkillState) -> None:
        """Move skill directory between state directories."""
        from_dir = self.skills_dir / from_state.value / name
        to_dir = self.skills_dir / to_state.value / name

        if from_dir.exists():
            to_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(from_dir), str(to_dir))

            # Clear from loaded cache
            if name in self._loaded_skills:
                del self._loaded_skills[name]

    def _save_usage_record(self, name: str, record: SkillUsageRecord) -> None:
        """Save usage record to skill's history file."""
        for state in SkillState:
            skill_dir = self.skills_dir / state.value / name
            if skill_dir.exists():
                history_file = skill_dir / "usage_history.jsonl"
                with open(history_file, "a") as f:
                    f.write(json.dumps(record.model_dump(mode="json"), default=str) + "\n")
                break

    def codify_skill(
        self,
        name: str,
        description: str,
        instructions: str,
        tags: list[str] | None = None,
        domains: list[str] | None = None,
        steps: list[str] | None = None,
        scripts: dict[str, str] | None = None,
        skill_type: SkillType = SkillType.ATOMIC,
    ) -> Skill:
        """
        Codify a pattern as a new skill.

        This is how the agent creates its own skills from experience,
        similar to codifying procedures in memory.
        """
        # Create skill directory in developing/
        skill_dir = self.skills_dir / "developing" / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Build SKILL.md content
        frontmatter: dict[str, Any] = {
            "name": name,
            "description": description,
            "version": "1.0.0",
            "type": skill_type.value,
        }
        if tags:
            frontmatter["tags"] = tags
        if domains:
            frontmatter["domains"] = domains
        if steps:
            frontmatter["steps"] = steps

        content = "---\n"
        content += yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        content += "---\n\n"
        content += instructions

        (skill_dir / "SKILL.md").write_text(content)

        # Create scripts if provided
        if scripts:
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            for script_name, script_content in scripts.items():
                script_path = scripts_dir / script_name
                script_path.write_text(script_content)
                script_path.chmod(0o755)  # Make executable

        # Refresh index to pick up new skill
        self.refresh_index()

        # Update metadata
        index = self.index
        if name in index.skills:
            index.skills[name].learned_from = "codified"
            index.skills[name].created_at = datetime.now(UTC)
            self._save_index(index)

        return self.get_skill(name)

    def extract_skill(
        self,
        name: str,
        description: str,
        instructions: str,
        episode_ids: list[str],
        tags: list[str] | None = None,
        domains: list[str] | None = None,
    ) -> Skill:
        """
        Extract a skill from successful episodes.

        Similar to codify but marks the skill as extracted and
        tracks which episodes it came from.
        """
        skill = self.codify_skill(
            name=name,
            description=description,
            instructions=instructions,
            tags=tags,
            domains=domains,
        )

        # Update metadata to mark as extracted
        index = self.index
        if name in index.skills:
            index.skills[name].learned_from = "extracted"
            index.skills[name].extracted_from = episode_ids
            self._save_index(index)

        return skill

    def install_skill(
        self,
        skill_dir: Path,
        state: SkillState = SkillState.DEVELOPING
    ) -> Skill | None:
        """
        Install an external skill into the agent's body.

        This copies a skill from an external source into the agent's
        skills directory.
        """
        if not skill_dir.exists() or not (skill_dir / "SKILL.md").exists():
            return None

        skill = Skill.from_skill_md(skill_dir)
        name = skill.metadata.name

        # Copy to appropriate state directory
        dest_dir = self.skills_dir / state.value / name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(skill_dir, dest_dir)

        # Refresh index
        self.refresh_index()

        # Mark as external
        index = self.index
        if name in index.skills:
            index.skills[name].learned_from = "external"
            self._save_index(index)

        return self.get_skill(name)

    def delete_skill(self, name: str) -> bool:
        """Delete a skill completely."""
        for state in SkillState:
            skill_dir = self.skills_dir / state.value / name
            if skill_dir.exists():
                shutil.rmtree(skill_dir)

                # Remove from index
                index = self.index
                if name in index.skills:
                    del index.skills[name]
                if name in index.active_skills:
                    index.active_skills.remove(name)
                self._save_index(index)

                # Remove from cache
                if name in self._loaded_skills:
                    del self._loaded_skills[name]

                return True
        return False

    def list_skills(self, state: SkillState | None = None) -> list[SkillMetadata]:
        """List skills, optionally filtered by state."""
        if state:
            return self.index.get_by_state(state)
        return list(self.index.skills.values())

    def get_active_skills(self) -> list[Skill]:
        """Get all currently active skills with full content."""
        return [
            self.get_skill(name)
            for name in self.index.active_skills
            if self.get_skill(name) is not None
        ]

    def search_skills(self, query: str, top_k: int = 10) -> list[SkillMetadata]:
        """Search skills by query."""
        results = self.index.search(query)
        return results[:top_k]

    def recommend_for_context(
        self,
        context: dict[str, Any],
        top_k: int = 5
    ) -> list[SkillMetadata]:
        """Recommend skills applicable to context."""
        applicable = []

        for skill in self.index.skills.values():
            if skill.state == SkillState.ATROPHIED:
                continue

            if skill.is_applicable(context):
                # Score by proficiency and recency
                score = skill.proficiency
                if skill.last_used:
                    days_ago = (datetime.now(UTC) - skill.last_used).total_seconds() / 86400
                    recency_bonus = max(0, 1 - days_ago / 30) * 0.3
                    score += recency_bonus

                applicable.append((skill, score))

        applicable.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in applicable[:top_k]]

    def get_skill_context(self) -> str:
        """
        Get skill context for injection into agent prompt.

        This shows active skills with their instructions.
        """
        active = self.get_active_skills()
        if not active:
            return ""

        lines = ["## Active Skills", ""]
        for skill in active:
            lines.append(skill.to_prompt_context())
            lines.append("")

        return "\n".join(lines)

    def get_skill_summary(self) -> str:
        """
        Get summary of all skills for prompt awareness.

        This is the "unconscious" awareness - names and proficiency only.
        """
        index = self.index
        if not index.skills:
            return "No skills acquired."

        lines = ["### Skills"]

        # Group by state
        for state in SkillState:
            skills = index.get_by_state(state)
            if skills:
                lines.append(f"\n**{state.value.title()}:**")
                for skill in sorted(skills, key=lambda s: -s.proficiency):
                    active = " (active)" if skill.is_active else ""
                    lines.append(f"- {skill.name}: {skill.proficiency:.0%}{active}")

        return "\n".join(lines)

    def get_statistics(self) -> dict[str, Any]:
        """Get skill statistics."""
        skills = list(self.index.skills.values())
        active = [s for s in skills if s.state != SkillState.ATROPHIED]

        return {
            "total_skills": len(skills),
            "learned": len([s for s in skills if s.state == SkillState.LEARNED]),
            "developing": len([s for s in skills if s.state == SkillState.DEVELOPING]),
            "atrophied": len([s for s in skills if s.state == SkillState.ATROPHIED]),
            "active_now": len(self.index.active_skills),
            "total_uses": sum(s.use_count for s in skills),
            "avg_proficiency": sum(s.proficiency for s in active) / len(active) if active else 0,
            "by_type": {
                t.value: len([s for s in skills if s.skill_type == t])
                for t in SkillType
            },
            "domains": list(set(d for s in skills for d in s.domains)),
        }


# =============================================================================
# Skill Composer
# =============================================================================

class SkillComposer:
    """
    Compose skills into higher-level strategies.

    Creates composite skills from sequences of atomic skills,
    handles skill dependencies, and generates execution plans.
    """

    def __init__(self, manager: SkillManager):
        self.manager = manager

    def compose(
        self,
        skill_names: list[str],
        name: str,
        description: str,
    ) -> Skill | None:
        """Compose multiple skills into a composite skill."""
        skills = [self.manager.get_skill(n) for n in skill_names]
        skills = [s for s in skills if s is not None]

        if len(skills) < 2:
            return None

        # Build instructions from sub-skills
        instructions = f"This skill composes: {', '.join(skill_names)}\n\n"
        instructions += "## Execution Order\n"
        for i, skill in enumerate(skills, 1):
            instructions += f"\n### {i}. {skill.metadata.name}\n"
            instructions += skill.get_execution_instructions()

        # Merge domains
        all_domains = list(set(d for s in skills for d in s.metadata.domains))

        return self.manager.codify_skill(
            name=name,
            description=description,
            instructions=instructions,
            domains=all_domains,
            skill_type=SkillType.COMPOSITE,
        )

    def get_execution_plan(self, skill_name: str) -> list[Skill]:
        """Get flattened execution plan for a skill."""
        skill = self.manager.get_skill(skill_name)
        if not skill:
            return []

        if skill.metadata.skill_type != SkillType.COMPOSITE:
            return [skill]

        plan = []
        for sub_name in skill.sub_skills:
            sub_skill = self.manager.get_skill(sub_name)
            if sub_skill:
                if sub_skill.metadata.skill_type == SkillType.COMPOSITE:
                    plan.extend(self.get_execution_plan(sub_name))
                else:
                    plan.append(sub_skill)

        return plan

    def suggest_compositions(self, goal: str) -> list[list[str]]:
        """Suggest skill compositions that might achieve a goal."""
        relevant = self.manager.search_skills(goal, top_k=10)

        if len(relevant) < 2:
            return []

        suggestions = []
        for i in range(len(relevant)):
            for j in range(len(relevant)):
                if i != j:
                    pair = [relevant[i].name, relevant[j].name]
                    suggestions.append(pair)

        return suggestions[:5]


# =============================================================================
# Unified Skill System
# =============================================================================

class SkillSystem:
    """
    Unified interface for the embodied skills system.

    Provides high-level operations on the skill body.
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.manager = SkillManager(skills_dir)
        self.composer = SkillComposer(self.manager)

    def initialize(self) -> None:
        """Initialize the skills system."""
        self.manager.initialize()

    def activate(self, name: str) -> Skill | None:
        """Activate a skill."""
        return self.manager.activate_skill(name)

    def deactivate(self, name: str) -> bool:
        """Deactivate a skill."""
        return self.manager.deactivate_skill(name)

    def use(
        self,
        name: str,
        success: bool,
        duration_seconds: float | None = None,
        context: str = "",
        outcome: str = "",
        reward: float = 0.0,
    ) -> SkillMetadata | None:
        """Record skill usage."""
        return self.manager.record_usage(
            name, success, duration_seconds, context, outcome, reward
        )

    def codify(
        self,
        name: str,
        description: str,
        instructions: str,
        **kwargs,
    ) -> Skill:
        """Codify a new skill."""
        return self.manager.codify_skill(name, description, instructions, **kwargs)

    def compose(
        self,
        skill_names: list[str],
        name: str,
        description: str,
    ) -> Skill | None:
        """Compose skills."""
        return self.composer.compose(skill_names, name, description)

    def recommend(
        self,
        context: dict[str, Any],
        goal: str | None = None,
    ) -> list[SkillMetadata]:
        """Recommend skills for context."""
        recommendations = self.manager.recommend_for_context(context)

        if goal:
            goal_relevant = self.manager.search_skills(goal)
            seen = {s.name for s in recommendations}
            for skill in goal_relevant:
                if skill.name not in seen:
                    recommendations.append(skill)

        return recommendations[:10]

    def get_context(self) -> str:
        """Get skill context for prompts."""
        return self.manager.get_skill_context()

    def get_summary(self) -> str:
        """Get skill summary for prompts."""
        return self.manager.get_skill_summary()

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics."""
        return self.manager.get_statistics()


# =============================================================================
# Helper Functions
# =============================================================================

def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content

    # Find end of frontmatter
    end_match = re.search(r'\n---\s*\n', content[3:])
    if not end_match:
        return {}, content

    frontmatter_str = content[3:end_match.start() + 3]
    body = content[end_match.end() + 3:]

    try:
        frontmatter = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, body.strip()


def parse_sections(content: str) -> dict[str, str]:
    """Parse markdown sections from content."""
    sections = {}
    current_section = None
    current_content = []

    for line in content.split("\n"):
        if line.startswith("## "):
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = line[3:].strip().lower().replace(" ", "_")
            current_content = []
        elif current_section:
            current_content.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def parse_numbered_list(content: str) -> list[str]:
    """Parse a numbered list from content."""
    items = []
    for line in content.split("\n"):
        line = line.strip()
        # Match "1. " or "1) " patterns
        match = re.match(r'^\d+[.)]\s+(.+)$', line)
        if match:
            items.append(match.group(1))
    return items


# =============================================================================
# Skill-Daemon Integration
# =============================================================================

def skill_daemon_to_pipeline(
    skill_name: str,
    daemon_def: dict[str, Any],
    skills_dir: Path,
) -> dict[str, Any]:
    """
    Convert a skill daemon definition to a Pipeline-compatible dict.

    Skill daemon definitions in SKILL.md frontmatter:
    ```yaml
    daemons:
      - name: git-status-monitor
        trigger: on_change
        sources: ["{cwd}/.git/index"]
        prompt: "Summarize git status changes"
    ```

    This converts to a Pipeline dict that can be registered with
    the unconscious daemon system.
    """
    daemon_name = daemon_def.get("name", f"{skill_name}-daemon")
    full_name = f"skill-{skill_name}-{daemon_name}"

    # Map trigger to Pipeline trigger config
    trigger_mode = daemon_def.get("trigger", "on_change")
    trigger_config = {
        "mode": trigger_mode,
        "n_steps": daemon_def.get("n_steps", 5),
        "debounce_ms": daemon_def.get("debounce_ms", 1000),
    }

    # Map sources
    sources = daemon_def.get("sources", [])
    source_configs = []
    for src in sources:
        if isinstance(src, str):
            source_configs.append({"path": src, "mode": "full"})
        elif isinstance(src, dict):
            source_configs.append(src)

    # Build output path
    output_path = daemon_def.get("output", f"skills/{skill_name}/{daemon_name}.md")

    return {
        "name": full_name,
        "description": f"Skill daemon from {skill_name}: {daemon_def.get('description', '')}",
        "sources": source_configs,
        "trigger": trigger_config,
        "prompt": daemon_def.get("prompt", "Analyze the source data."),
        "output": output_path,
        "model": daemon_def.get("model", "haiku"),
        "max_tokens": daemon_def.get("max_tokens", 500),
        "temperature": daemon_def.get("temperature", 0.0),
        "enabled": True,
        "priority": daemon_def.get("priority", 5),
        "group": f"skill-{skill_name}",
        "tags": ["skill-daemon", skill_name],
    }


def get_active_skill_daemons(manager: "SkillManager") -> list[dict[str, Any]]:
    """
    Get Pipeline-compatible dicts for all daemons from active skills.

    Returns a list of pipeline configurations that can be registered
    with the UnconsciousRunner.
    """
    pipelines = []

    for skill in manager.get_active_skills():
        if not skill:
            continue

        for daemon_def in skill.metadata.daemons:
            pipeline = skill_daemon_to_pipeline(
                skill.metadata.name,
                daemon_def,
                manager.skills_dir,
            )
            pipelines.append(pipeline)

    return pipelines


def get_skill_daemon_names(skill: Skill) -> list[str]:
    """Get the full daemon names for a skill's daemons."""
    names = []
    for daemon_def in skill.metadata.daemons:
        daemon_name = daemon_def.get("name", f"{skill.metadata.name}-daemon")
        full_name = f"skill-{skill.metadata.name}-{daemon_name}"
        names.append(full_name)
    return names
