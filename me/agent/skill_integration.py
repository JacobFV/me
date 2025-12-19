"""
Skill Integration - Advanced skill features for the embodied agent.

This module bridges skills with other agent subsystems:
1. Auto-skill extraction from episodes
2. Skill recommendations based on context/goals
3. Live daemon integration (skill daemons ↔ UnconsciousRunner)
4. Semantic skill search via embeddings
5. Character evolution from skill mastery
6. Skill dependency resolution
7. Atrophy detection and warnings
8. Inter-agent skill transfer

Philosophy: Skills are not just stored capabilities—they actively shape
the agent's perception (via daemons), character (via mastery), and
decision-making (via recommendations).
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from me.agent.skills import (
    Skill,
    SkillManager,
    SkillMetadata,
    SkillState,
    SkillType,
    skill_daemon_to_pipeline,
    get_active_skill_daemons,
)
from me.agent.body import Episode, Character, Focus

if TYPE_CHECKING:
    from me.agent.unconscious import UnconsciousRunner, Pipeline
    from me.agent.learning import Pattern, Experience


# =============================================================================
# Semantic Skill Search (Embedding-based)
# =============================================================================

_embedding_model = None


def _get_embedding_model():
    """Lazy-load sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def compute_skill_embedding(skill: Skill) -> list[float]:
    """
    Compute embedding vector for a skill.

    Uses skill name, description, tags, and instructions to create
    a semantic representation for similarity matching.
    """
    model = _get_embedding_model()

    # Combine skill metadata into embeddable text
    text_parts = [
        skill.metadata.name,
        skill.metadata.description,
        " ".join(skill.metadata.tags),
        " ".join(skill.metadata.domains),
    ]

    # Add first 500 chars of instructions
    if skill.instructions:
        text_parts.append(skill.instructions[:500])

    text = " ".join(text_parts)
    embedding = model.encode(text, normalize_embeddings=True)

    return embedding.tolist()


def update_skill_embeddings(manager: SkillManager) -> int:
    """
    Update embeddings for all skills that need them.

    Returns number of skills updated.
    """
    updated = 0
    index = manager.index

    for name, meta in index.skills.items():
        if not meta.embedding:
            skill = manager.get_skill(name)
            if skill:
                meta.embedding = compute_skill_embedding(skill)
                updated += 1

    if updated > 0:
        manager._save_index(index)

    return updated


def semantic_skill_search(
    manager: SkillManager,
    query: str,
    top_k: int = 5,
) -> list[tuple[SkillMetadata, float]]:
    """
    Search skills using semantic similarity.

    Returns skills ranked by cosine similarity to query embedding.
    """
    model = _get_embedding_model()
    query_embedding = model.encode(query, normalize_embeddings=True)

    results = []
    for meta in manager.index.skills.values():
        if not meta.embedding:
            continue

        # Compute cosine similarity
        import numpy as np
        similarity = float(np.dot(query_embedding, meta.embedding))
        results.append((meta, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =============================================================================
# Skill Recommendations
# =============================================================================

class SkillRecommendation(BaseModel):
    """A skill recommendation with reasoning."""
    skill_name: str
    score: float
    reasons: list[str] = Field(default_factory=list)
    source: str = "unknown"  # context, goal, semantic, pattern


class SkillRecommender:
    """
    Recommends skills based on current context and goals.

    Uses multiple signals:
    - Semantic similarity to current task/goal
    - Context matching (domain, preconditions)
    - Historical success patterns
    - Proficiency (prefer mastered skills)
    """

    def __init__(self, manager: SkillManager):
        self.manager = manager

    def recommend(
        self,
        context: dict[str, Any] | None = None,
        goal: str | None = None,
        current_task: str | None = None,
        top_k: int = 5,
    ) -> list[SkillRecommendation]:
        """Get skill recommendations for current situation."""
        recommendations: dict[str, SkillRecommendation] = {}

        # 1. Context-based recommendations
        if context:
            for meta in self.manager.index.skills.values():
                if meta.state == SkillState.ATROPHIED:
                    continue
                if meta.is_applicable(context):
                    name = meta.name
                    if name not in recommendations:
                        recommendations[name] = SkillRecommendation(
                            skill_name=name,
                            score=0.0,
                            source="context",
                        )
                    rec = recommendations[name]
                    rec.score += 0.3 * meta.proficiency
                    rec.reasons.append(f"Applicable to context domain")

        # 2. Goal-based semantic recommendations
        if goal:
            semantic_results = semantic_skill_search(self.manager, goal, top_k=10)
            for meta, similarity in semantic_results:
                if meta.state == SkillState.ATROPHIED:
                    continue
                name = meta.name
                if name not in recommendations:
                    recommendations[name] = SkillRecommendation(
                        skill_name=name,
                        score=0.0,
                        source="goal",
                    )
                rec = recommendations[name]
                rec.score += similarity * 0.5
                rec.reasons.append(f"Semantic match to goal ({similarity:.2f})")

        # 3. Task-based semantic recommendations
        if current_task:
            semantic_results = semantic_skill_search(self.manager, current_task, top_k=10)
            for meta, similarity in semantic_results:
                if meta.state == SkillState.ATROPHIED:
                    continue
                name = meta.name
                if name not in recommendations:
                    recommendations[name] = SkillRecommendation(
                        skill_name=name,
                        score=0.0,
                        source="task",
                    )
                rec = recommendations[name]
                rec.score += similarity * 0.4
                rec.reasons.append(f"Semantic match to task ({similarity:.2f})")

        # 4. Proficiency bonus for mastered skills
        for name, rec in recommendations.items():
            meta = self.manager.index.skills.get(name)
            if meta:
                rec.score += meta.proficiency * 0.2
                if meta.is_mastered:
                    rec.reasons.append("Mastered skill")

        # Sort by score
        sorted_recs = sorted(
            recommendations.values(),
            key=lambda r: r.score,
            reverse=True
        )

        return sorted_recs[:top_k]

    def get_recommendation_prompt(
        self,
        context: dict[str, Any] | None = None,
        goal: str | None = None,
        current_task: str | None = None,
    ) -> str:
        """Generate prompt text for skill recommendations."""
        recs = self.recommend(context, goal, current_task)

        if not recs:
            return ""

        lines = ["### Recommended Skills"]
        for rec in recs:
            meta = self.manager.index.skills.get(rec.skill_name)
            if meta:
                reasons = "; ".join(rec.reasons[:2])
                lines.append(
                    f"- **{rec.skill_name}** ({meta.proficiency:.0%}): {reasons}"
                )

        return "\n".join(lines)


# =============================================================================
# Auto-Skill Extraction
# =============================================================================

class SkillCandidate(BaseModel):
    """A potential skill to be extracted from episodes."""
    name: str
    description: str
    source_episodes: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    suggested_steps: list[str] = Field(default_factory=list)
    suggested_tags: list[str] = Field(default_factory=list)


class SkillExtractor:
    """
    Extracts skill candidates from successful episodes.

    Analyzes patterns in completed episodes to identify
    reusable capabilities that could be codified as skills.
    """

    def __init__(self, manager: SkillManager, body_dir: Path):
        self.manager = manager
        self.body_dir = body_dir
        self.episodes_dir = body_dir / "memory" / "episodes"
        self._candidates_file = body_dir / "skills" / "candidates.json"

    def analyze_episodes(
        self,
        min_success_count: int = 2,
        lookback_days: int = 30,
    ) -> list[SkillCandidate]:
        """
        Analyze recent episodes for skill extraction candidates.

        Looks for patterns in successful episodes that could become skills.
        """
        from me.agent.body import markdown_to_model, Episode

        candidates: dict[str, SkillCandidate] = {}
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        # Load recent successful episodes
        episodes = []
        for ep_path in self.episodes_dir.glob("*.md"):
            try:
                content = ep_path.read_text()
                episode, body = markdown_to_model(content, Episode)
                if episode.outcome == "completed" and episode.started_at >= cutoff:
                    episodes.append((episode, body))
            except Exception:
                continue

        if len(episodes) < min_success_count:
            return []

        # Group by common patterns
        # Pattern 1: Common tags
        tag_groups: dict[str, list[Episode]] = defaultdict(list)
        for episode, _ in episodes:
            for tag in episode.tags:
                tag_groups[tag].append(episode)

        for tag, eps in tag_groups.items():
            if len(eps) >= min_success_count:
                # Check if skill already exists
                if tag in self.manager.index.skills:
                    continue

                candidate = SkillCandidate(
                    name=f"{tag}-workflow",
                    description=f"Workflow pattern for {tag} tasks",
                    source_episodes=[e.id for e in eps],
                    confidence=min(1.0, len(eps) / 5),
                    suggested_tags=[tag],
                )

                # Extract common lessons as steps
                all_lessons = []
                for ep in eps:
                    all_lessons.extend(ep.lessons)

                if all_lessons:
                    candidate.suggested_steps = list(set(all_lessons))[:5]

                candidates[candidate.name] = candidate

        # Pattern 2: Common tools used
        tool_groups: dict[str, list[Episode]] = defaultdict(list)
        for episode, _ in episodes:
            tool_key = "-".join(sorted(episode.tools_used)[:3])
            if tool_key:
                tool_groups[tool_key].append(episode)

        for tool_key, eps in tool_groups.items():
            if len(eps) >= min_success_count:
                tools = tool_key.split("-")
                name = f"{'_'.join(tools[:2])}-pattern"

                if name in self.manager.index.skills:
                    continue

                candidate = SkillCandidate(
                    name=name,
                    description=f"Pattern using {', '.join(tools)}",
                    source_episodes=[e.id for e in eps],
                    confidence=min(1.0, len(eps) / 5),
                    suggested_tags=tools,
                )

                candidates[name] = candidate

        # Save candidates
        result = list(candidates.values())
        self._save_candidates(result)

        return result

    def create_skill_from_candidate(
        self,
        candidate: SkillCandidate,
        instructions: str | None = None,
    ) -> Skill | None:
        """Create a skill from a candidate."""
        if not instructions:
            # Generate basic instructions from candidate
            lines = [f"## Overview\n\n{candidate.description}"]

            if candidate.suggested_steps:
                lines.append("\n## Steps\n")
                for i, step in enumerate(candidate.suggested_steps, 1):
                    lines.append(f"{i}. {step}")

            if candidate.patterns:
                lines.append("\n## Patterns\n")
                for pattern in candidate.patterns:
                    lines.append(f"- {pattern}")

            instructions = "\n".join(lines)

        return self.manager.extract_skill(
            name=candidate.name,
            description=candidate.description,
            instructions=instructions,
            episode_ids=candidate.source_episodes,
            tags=candidate.suggested_tags,
        )

    def _save_candidates(self, candidates: list[SkillCandidate]) -> None:
        """Save candidates to file."""
        self._candidates_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._candidates_file, 'w') as f:
            json.dump(
                [c.model_dump(mode='json') for c in candidates],
                f, indent=2, default=str
            )

    def get_pending_candidates(self) -> list[SkillCandidate]:
        """Get saved candidates that haven't been processed."""
        if not self._candidates_file.exists():
            return []

        try:
            with open(self._candidates_file, 'r') as f:
                data = json.load(f)
            return [SkillCandidate.model_validate(c) for c in data]
        except Exception:
            return []


# =============================================================================
# Live Daemon Integration
# =============================================================================

class SkillDaemonManager:
    """
    Manages the lifecycle of skill-defined daemons.

    When a skill is activated, its daemons are registered with
    the UnconsciousRunner. When deactivated, they're unregistered.
    """

    def __init__(self, skill_manager: SkillManager):
        self.skill_manager = skill_manager
        self._active_daemons: dict[str, list[str]] = {}  # skill_name -> daemon names

    def sync_daemons(self, runner: "UnconsciousRunner") -> dict[str, Any]:
        """
        Synchronize skill daemons with UnconsciousRunner.

        - Register daemons for newly activated skills
        - Unregister daemons for deactivated skills

        Returns summary of changes.
        """
        from me.agent.skills import get_active_skill_daemons, get_skill_daemon_names

        changes = {"registered": [], "unregistered": []}

        current_active = set(self.skill_manager.index.active_skills)
        previously_active = set(self._active_daemons.keys())

        # Skills that were deactivated
        for skill_name in previously_active - current_active:
            daemon_names = self._active_daemons.pop(skill_name, [])
            for daemon_name in daemon_names:
                runner.remove_pipeline(daemon_name)
                changes["unregistered"].append(daemon_name)

        # Skills that were activated
        for skill_name in current_active - previously_active:
            skill = self.skill_manager.get_skill(skill_name)
            if not skill or not skill.metadata.daemons:
                continue

            daemon_names = []
            for daemon_def in skill.metadata.daemons:
                pipeline_dict = skill_daemon_to_pipeline(
                    skill_name,
                    daemon_def,
                    self.skill_manager.skills_dir,
                )

                # Register with runner
                from me.agent.unconscious import Pipeline
                pipeline = Pipeline.model_validate(pipeline_dict)
                runner.install_pipeline(pipeline)

                daemon_names.append(pipeline.name)
                changes["registered"].append(pipeline.name)

            self._active_daemons[skill_name] = daemon_names

        return changes

    def get_active_daemon_count(self) -> int:
        """Get count of active skill daemons."""
        return sum(len(daemons) for daemons in self._active_daemons.values())

    def get_daemons_for_skill(self, skill_name: str) -> list[str]:
        """Get daemon names for a skill."""
        return self._active_daemons.get(skill_name, [])


# =============================================================================
# Character Evolution from Skills
# =============================================================================

class SkillCharacterMapping(BaseModel):
    """Mapping from skill tags to character trait influences."""
    tag: str
    trait: str
    direction: float = 0.1  # Positive = increases, negative = decreases


# Default mappings - skills influence character
DEFAULT_SKILL_CHARACTER_MAPPINGS = [
    # Analytical skills increase thoroughness
    SkillCharacterMapping(tag="debugging", trait="thoroughness", direction=0.05),
    SkillCharacterMapping(tag="analysis", trait="thoroughness", direction=0.05),
    SkillCharacterMapping(tag="testing", trait="thoroughness", direction=0.03),

    # Creative skills increase exploration
    SkillCharacterMapping(tag="design", trait="exploration_tendency", direction=0.05),
    SkillCharacterMapping(tag="creative", trait="exploration_tendency", direction=0.05),
    SkillCharacterMapping(tag="refactoring", trait="exploration_tendency", direction=0.03),

    # Risk-taking skills increase risk tolerance
    SkillCharacterMapping(tag="deployment", trait="risk_tolerance", direction=0.03),
    SkillCharacterMapping(tag="production", trait="risk_tolerance", direction=0.02),

    # Persistent skills increase persistence
    SkillCharacterMapping(tag="debugging", trait="persistence", direction=0.05),
    SkillCharacterMapping(tag="troubleshooting", trait="persistence", direction=0.05),
]


def compute_character_influence(
    manager: SkillManager,
    mappings: list[SkillCharacterMapping] | None = None,
) -> dict[str, float]:
    """
    Compute character trait influences from mastered skills.

    Returns delta values to apply to character traits.
    """
    mappings = mappings or DEFAULT_SKILL_CHARACTER_MAPPINGS

    influences: dict[str, float] = defaultdict(float)

    # Only mastered skills influence character
    learned_skills = manager.index.get_by_state(SkillState.LEARNED)

    for skill in learned_skills:
        for mapping in mappings:
            if mapping.tag in skill.tags:
                # Scale by proficiency
                delta = mapping.direction * skill.proficiency
                influences[mapping.trait] += delta

    # Clamp influences
    return {
        trait: max(-0.3, min(0.3, value))
        for trait, value in influences.items()
    }


def apply_skill_character_evolution(
    manager: SkillManager,
    character: Character,
    mappings: list[SkillCharacterMapping] | None = None,
) -> Character:
    """
    Apply skill-based character evolution.

    Returns updated character with trait modifications from mastered skills.
    """
    influences = compute_character_influence(manager, mappings)

    # Apply influences
    if "thoroughness" in influences:
        character.thoroughness = max(0.0, min(1.0,
            character.thoroughness + influences["thoroughness"]
        ))

    if "exploration_tendency" in influences:
        character.exploration_tendency = max(0.0, min(1.0,
            character.exploration_tendency + influences["exploration_tendency"]
        ))

    if "risk_tolerance" in influences:
        character.risk_tolerance = max(0.0, min(1.0,
            character.risk_tolerance + influences["risk_tolerance"]
        ))

    if "persistence" in influences:
        character.persistence = max(0.0, min(1.0,
            character.persistence + influences["persistence"]
        ))

    return character


# =============================================================================
# Skill Dependencies
# =============================================================================

class SkillDependencyResolver:
    """
    Resolves skill dependencies for activation.

    Skills can declare prerequisites that must be activated first.
    """

    def __init__(self, manager: SkillManager):
        self.manager = manager

    def get_dependencies(self, skill_name: str) -> list[str]:
        """Get direct dependencies for a skill."""
        skill = self.manager.get_skill(skill_name)
        if not skill:
            return []

        # Check sub_skills for composite skills
        if skill.metadata.skill_type == SkillType.COMPOSITE:
            return skill.sub_skills

        # Check preconditions that reference other skills
        deps = []
        for precond in skill.metadata.preconditions:
            if precond.startswith("skill:"):
                deps.append(precond[6:])  # Remove "skill:" prefix

        return deps

    def resolve_activation_order(self, skill_name: str) -> list[str]:
        """
        Get ordered list of skills to activate.

        Returns skills in dependency order (dependencies first).
        """
        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            for dep in self.get_dependencies(name):
                if dep in self.manager.index.skills:
                    visit(dep)

            order.append(name)

        visit(skill_name)
        return order

    def activate_with_dependencies(self, skill_name: str) -> list[Skill]:
        """
        Activate a skill and all its dependencies.

        Returns list of activated skills.
        """
        activation_order = self.resolve_activation_order(skill_name)
        activated = []

        for name in activation_order:
            skill = self.manager.activate_skill(name)
            if skill:
                activated.append(skill)

        return activated

    def check_can_activate(self, skill_name: str) -> tuple[bool, list[str]]:
        """
        Check if a skill can be activated.

        Returns (can_activate, missing_dependencies).
        """
        deps = self.get_dependencies(skill_name)
        missing = [d for d in deps if d not in self.manager.index.skills]
        return len(missing) == 0, missing


# =============================================================================
# Atrophy Detection
# =============================================================================

class AtrophyWarning(BaseModel):
    """Warning about a skill approaching or in atrophy."""
    skill_name: str
    days_since_use: int
    current_proficiency: float
    state: SkillState
    warning_level: str  # "approaching", "atrophied", "critical"


class AtrophyDetector:
    """
    Detects skills that are atrophying from disuse.

    Provides warnings so the agent can choose to use skills
    before they decay significantly.
    """

    # Days thresholds
    APPROACHING_THRESHOLD = 21  # Warn at 3 weeks
    ATROPHIED_THRESHOLD = 30    # Atrophied at 1 month
    CRITICAL_THRESHOLD = 60     # Critical at 2 months

    def __init__(self, manager: SkillManager):
        self.manager = manager

    def check_atrophy(self) -> list[AtrophyWarning]:
        """Check all skills for atrophy warnings."""
        warnings = []
        now = datetime.now(UTC)

        for meta in self.manager.index.skills.values():
            if not meta.last_used:
                continue

            days_since = (now - meta.last_used).days

            if days_since >= self.CRITICAL_THRESHOLD:
                level = "critical"
            elif days_since >= self.ATROPHIED_THRESHOLD:
                level = "atrophied"
            elif days_since >= self.APPROACHING_THRESHOLD:
                level = "approaching"
            else:
                continue

            warnings.append(AtrophyWarning(
                skill_name=meta.name,
                days_since_use=days_since,
                current_proficiency=meta.proficiency,
                state=meta.state,
                warning_level=level,
            ))

        return sorted(warnings, key=lambda w: w.days_since_use, reverse=True)

    def get_atrophy_summary(self) -> str:
        """Get human-readable atrophy summary."""
        warnings = self.check_atrophy()

        if not warnings:
            return "All skills are healthy."

        lines = ["### Skill Atrophy Warnings"]

        critical = [w for w in warnings if w.warning_level == "critical"]
        if critical:
            lines.append("\n**Critical (60+ days unused):**")
            for w in critical:
                lines.append(f"- {w.skill_name}: {w.days_since_use} days")

        atrophied = [w for w in warnings if w.warning_level == "atrophied"]
        if atrophied:
            lines.append("\n**Atrophied (30+ days unused):**")
            for w in atrophied:
                lines.append(f"- {w.skill_name}: {w.days_since_use} days")

        approaching = [w for w in warnings if w.warning_level == "approaching"]
        if approaching:
            lines.append("\n**Approaching Atrophy (21+ days unused):**")
            for w in approaching:
                lines.append(f"- {w.skill_name}: {w.days_since_use} days")

        return "\n".join(lines)


# =============================================================================
# Inter-Agent Skill Transfer
# =============================================================================

class SkillPackage(BaseModel):
    """A packaged skill for transfer between agents."""
    name: str
    version: str
    description: str
    skill_md_content: str
    scripts: dict[str, str] = Field(default_factory=dict)  # filename -> content
    references: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    exported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_agent: str | None = None


class SkillTransfer:
    """
    Export and import skills between agents.

    Allows agents to share learned capabilities.
    """

    def __init__(self, manager: SkillManager, agent_id: str | None = None):
        self.manager = manager
        self.agent_id = agent_id

    def export_skill(self, skill_name: str) -> SkillPackage | None:
        """Export a skill as a transferable package."""
        skill = self.manager.get_skill(skill_name)
        if not skill:
            return None

        skill_dir = self.manager.get_skill_path(skill_name)
        if not skill_dir:
            return None

        # Read SKILL.md
        skill_md = skill_dir / "SKILL.md"
        skill_md_content = skill_md.read_text() if skill_md.exists() else ""

        # Read scripts
        scripts = {}
        scripts_dir = skill_dir / "scripts"
        if scripts_dir.exists():
            for script_path in scripts_dir.iterdir():
                if script_path.is_file():
                    scripts[script_path.name] = script_path.read_text()

        # Read references
        references = {}
        refs_dir = skill_dir / "references"
        if refs_dir.exists():
            for ref_path in refs_dir.iterdir():
                if ref_path.is_file():
                    references[ref_path.name] = ref_path.read_text()

        return SkillPackage(
            name=skill.metadata.name,
            version=skill.metadata.version,
            description=skill.metadata.description,
            skill_md_content=skill_md_content,
            scripts=scripts,
            references=references,
            metadata={
                "tags": skill.metadata.tags,
                "domains": skill.metadata.domains,
                "skill_type": skill.metadata.skill_type.value,
                "original_proficiency": skill.metadata.proficiency,
            },
            source_agent=self.agent_id,
        )

    def import_skill(
        self,
        package: SkillPackage,
        state: SkillState = SkillState.DEVELOPING,
    ) -> Skill | None:
        """Import a skill from a package."""
        # Create skill directory
        skill_dir = self.manager.skills_dir / state.value / package.name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md
        (skill_dir / "SKILL.md").write_text(package.skill_md_content)

        # Write scripts
        if package.scripts:
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            for filename, content in package.scripts.items():
                script_path = scripts_dir / filename
                script_path.write_text(content)
                script_path.chmod(0o755)

        # Write references
        if package.references:
            refs_dir = skill_dir / "references"
            refs_dir.mkdir(exist_ok=True)
            for filename, content in package.references.items():
                (refs_dir / filename).write_text(content)

        # Refresh index
        self.manager.refresh_index()

        # Update metadata
        index = self.manager.index
        if package.name in index.skills:
            index.skills[package.name].learned_from = f"transfer:{package.source_agent or 'unknown'}"
            self.manager._save_index(index)

        return self.manager.get_skill(package.name)

    def export_to_file(self, skill_name: str, output_path: Path) -> bool:
        """Export a skill to a JSON file."""
        package = self.export_skill(skill_name)
        if not package:
            return False

        output_path.write_text(
            json.dumps(package.model_dump(mode='json'), indent=2, default=str)
        )
        return True

    def import_from_file(
        self,
        input_path: Path,
        state: SkillState = SkillState.DEVELOPING,
    ) -> Skill | None:
        """Import a skill from a JSON file."""
        if not input_path.exists():
            return None

        try:
            data = json.loads(input_path.read_text())
            package = SkillPackage.model_validate(data)
            return self.import_skill(package, state)
        except Exception:
            return None


# =============================================================================
# Unified Skill Integration System
# =============================================================================

class SkillIntegration:
    """
    Unified interface for all skill integration features.

    Coordinates:
    - Recommendations based on context/goals
    - Auto-extraction from episodes
    - Daemon synchronization
    - Character evolution
    - Dependency resolution
    - Atrophy detection
    - Skill transfer
    """

    def __init__(self, manager: SkillManager, body_dir: Path, agent_id: str | None = None):
        self.manager = manager
        self.body_dir = body_dir
        self.agent_id = agent_id

        # Initialize subsystems
        self.recommender = SkillRecommender(manager)
        self.extractor = SkillExtractor(manager, body_dir)
        self.daemon_manager = SkillDaemonManager(manager)
        self.dependency_resolver = SkillDependencyResolver(manager)
        self.atrophy_detector = AtrophyDetector(manager)
        self.transfer = SkillTransfer(manager, agent_id)

    def initialize(self) -> None:
        """Initialize the integration system."""
        # Update embeddings for semantic search
        update_skill_embeddings(self.manager)

    def on_agent_step(
        self,
        context: dict[str, Any] | None = None,
        goal: str | None = None,
        current_task: str | None = None,
        runner: "UnconsciousRunner | None" = None,
    ) -> dict[str, Any]:
        """
        Called each agent step to provide skill-related updates.

        Returns dict with recommendations, warnings, and daemon changes.
        """
        result = {
            "recommendations": [],
            "atrophy_warnings": [],
            "daemon_changes": {},
        }

        # Get recommendations
        if goal or current_task:
            recs = self.recommender.recommend(context, goal, current_task)
            result["recommendations"] = [r.model_dump() for r in recs]

        # Check atrophy
        warnings = self.atrophy_detector.check_atrophy()
        result["atrophy_warnings"] = [w.model_dump() for w in warnings[:3]]

        # Sync daemons
        if runner:
            changes = self.daemon_manager.sync_daemons(runner)
            result["daemon_changes"] = changes

        return result

    def get_step_prompt_additions(
        self,
        context: dict[str, Any] | None = None,
        goal: str | None = None,
        current_task: str | None = None,
    ) -> str:
        """Get prompt additions for the current step."""
        sections = []

        # Recommendations
        rec_prompt = self.recommender.get_recommendation_prompt(context, goal, current_task)
        if rec_prompt:
            sections.append(rec_prompt)

        # Atrophy warnings (only critical)
        warnings = self.atrophy_detector.check_atrophy()
        critical = [w for w in warnings if w.warning_level in ("critical", "atrophied")]
        if critical:
            lines = ["### Skill Health Warnings"]
            for w in critical[:2]:
                lines.append(f"- {w.skill_name}: unused for {w.days_since_use} days")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def extract_skills_from_episodes(self) -> list[SkillCandidate]:
        """Run skill extraction analysis."""
        return self.extractor.analyze_episodes()

    def evolve_character(self, character: Character) -> Character:
        """Apply skill-based character evolution."""
        return apply_skill_character_evolution(self.manager, character)

    def activate_skill_with_deps(self, skill_name: str) -> list[Skill]:
        """Activate a skill with all dependencies."""
        return self.dependency_resolver.activate_with_dependencies(skill_name)

    def export_skill(self, skill_name: str) -> SkillPackage | None:
        """Export a skill for transfer."""
        return self.transfer.export_skill(skill_name)

    def import_skill(self, package: SkillPackage) -> Skill | None:
        """Import a skill from package."""
        return self.transfer.import_skill(package)


# =============================================================================
# Daemon Definitions for Background Processing
# =============================================================================

def get_skill_extraction_daemon() -> dict[str, Any]:
    """
    Get pipeline definition for skill extraction daemon.

    This daemon periodically analyzes episodes and suggests new skills.
    """
    return {
        "name": "skill-extractor",
        "description": "Analyzes episodes and suggests skills to codify",
        "sources": [
            {"path": "{body}/memory/episodes/", "mode": "full"},
        ],
        "trigger": {
            "mode": "every_n_steps",
            "n_steps": 50,
            "debounce_ms": 60000,
        },
        "prompt": """Analyze recent successful episodes and identify patterns that could become reusable skills.

Look for:
1. Repeated approaches to similar problems
2. Common tool combinations
3. Successful strategies with multiple uses

For each potential skill, note:
- Name (hyphen-separated, descriptive)
- What it accomplishes
- Key steps or patterns
- Tags/domains

Output as JSON array of skill candidates.""",
        "output": "skill-candidates.md",
        "model": "haiku",
        "max_tokens": 1000,
        "temperature": 0.3,
        "enabled": True,
        "priority": 8,
        "group": "learning",
        "tags": ["skill-extraction", "learning"],
    }


def get_atrophy_warning_daemon() -> dict[str, Any]:
    """
    Get pipeline definition for atrophy warning daemon.

    This daemon checks for skills approaching atrophy.
    """
    return {
        "name": "skill-atrophy-monitor",
        "description": "Monitors skill usage and warns about atrophy",
        "sources": [
            {"path": "{body}/skills/index.json", "mode": "full"},
        ],
        "trigger": {
            "mode": "every_n_steps",
            "n_steps": 20,
            "debounce_ms": 30000,
        },
        "prompt": """Review skill usage patterns and identify skills at risk of atrophy.

A skill is at risk if:
- Not used in 21+ days (approaching)
- Not used in 30+ days (atrophied)
- Not used in 60+ days (critical)

For each at-risk skill, suggest:
1. A simple task to practice the skill
2. Whether it should be allowed to atrophy (if rarely needed)

Be concise - only report if there are at-risk skills.""",
        "output": "skill-health.md",
        "model": "haiku",
        "max_tokens": 500,
        "temperature": 0.0,
        "enabled": True,
        "priority": 7,
        "group": "self-management",
        "tags": ["skill-health", "atrophy"],
    }
