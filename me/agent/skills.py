"""
Skill Extraction & Composition System.

Enables automatic extraction of reusable skills from successful episodes,
skill chaining into higher-level strategies, and transfer across domains.

Philosophy: Skills are the procedural atoms of intelligence. They encode
"how to do things" in a composable, transferable form. Building a skill
library is building a capability surface.

Architecture:
    - Skill: A reusable capability with applicability conditions
    - SkillLibrary: Storage and retrieval of skills
    - SkillExtractor: Automatic skill extraction from episodes
    - SkillComposer: Chain skills into higher-level strategies
    - SkillTransfer: Apply skills to new domains
"""

from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field


# =============================================================================
# Skill Model
# =============================================================================

class SkillType(str, Enum):
    """Types of skills."""
    ATOMIC = "atomic"          # Single action skill
    COMPOSITE = "composite"    # Composed of other skills
    ABSTRACT = "abstract"      # Pattern that can be instantiated
    REFLEX = "reflex"          # Automatic response to stimulus


class SkillStatus(str, Enum):
    """Skill status."""
    DRAFT = "draft"            # Being developed
    ACTIVE = "active"          # Ready for use
    DEPRECATED = "deprecated"  # No longer recommended
    ARCHIVED = "archived"      # Kept for history


class Skill(BaseModel):
    """
    A reusable skill that can be composed and transferred.

    Skills encode procedural knowledge - how to accomplish something.
    They have applicability conditions, execution steps, and success criteria.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    type: SkillType = SkillType.ATOMIC
    status: SkillStatus = SkillStatus.ACTIVE

    # Applicability
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)  # Where this skill applies
    context_requirements: dict[str, Any] = Field(default_factory=dict)

    # Execution
    steps: list[str] = Field(default_factory=list)
    sub_skills: list[str] = Field(default_factory=list)  # For composite skills
    parameters: dict[str, Any] = Field(default_factory=dict)
    estimated_duration_ms: float = 0.0

    # Performance tracking
    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    avg_reward: float = 0.0
    last_used: datetime | None = None

    # Learning
    extracted_from: list[str] = Field(default_factory=list)  # Episode IDs
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Embedding for semantic matching
    embedding: list[float] = Field(default_factory=list)

    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_uses == 0:
            return 0.0
        return self.success_count / self.total_uses

    def is_applicable(self, context: dict[str, Any]) -> bool:
        """Check if skill is applicable in given context."""
        # Check context requirements
        for key, value in self.context_requirements.items():
            if key not in context:
                return False
            if context[key] != value:
                return False

        return True


# =============================================================================
# Skill Library
# =============================================================================

class SkillLibrary:
    """
    Storage and retrieval of skills.

    Provides semantic search, filtering by domain, and skill recommendations.
    """

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._skills_file = library_dir / "skills.json"
        self._index_file = library_dir / "skill_index.json"
        self._load()

    def _load(self):
        """Load skills from disk."""
        self._skills: dict[str, Skill] = {}

        if self._skills_file.exists():
            try:
                with open(self._skills_file, 'r') as f:
                    data = json.load(f)
                for skill_data in data:
                    skill = Skill.model_validate(skill_data)
                    self._skills[skill.id] = skill
            except Exception:
                pass

    def _save(self):
        """Save skills to disk."""
        with open(self._skills_file, 'w') as f:
            json.dump(
                [s.model_dump(mode='json') for s in self._skills.values()],
                f, indent=2, default=str
            )

    def add(self, skill: Skill):
        """Add a skill to the library."""
        self._skills[skill.id] = skill
        self._save()

    def get(self, skill_id: str) -> Skill | None:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def get_by_name(self, name: str) -> Skill | None:
        """Get a skill by name."""
        for skill in self._skills.values():
            if skill.name == name:
                return skill
        return None

    def update(self, skill: Skill):
        """Update an existing skill."""
        skill.updated_at = datetime.now(UTC)
        self._skills[skill.id] = skill
        self._save()

    def delete(self, skill_id: str):
        """Delete a skill."""
        if skill_id in self._skills:
            del self._skills[skill_id]
            self._save()

    def list_all(self, include_archived: bool = False) -> list[Skill]:
        """List all skills."""
        skills = list(self._skills.values())
        if not include_archived:
            skills = [s for s in skills if s.status != SkillStatus.ARCHIVED]
        return skills

    def list_by_domain(self, domain: str) -> list[Skill]:
        """List skills for a specific domain."""
        return [
            s for s in self._skills.values()
            if domain in s.domains and s.status == SkillStatus.ACTIVE
        ]

    def list_by_type(self, skill_type: SkillType) -> list[Skill]:
        """List skills of a specific type."""
        return [
            s for s in self._skills.values()
            if s.type == skill_type and s.status == SkillStatus.ACTIVE
        ]

    def search(self, query: str, top_k: int = 10) -> list[Skill]:
        """
        Search for skills matching query.

        Simple text matching for now, could use embeddings for semantic search.
        """
        query_lower = query.lower()
        scored = []

        for skill in self._skills.values():
            if skill.status != SkillStatus.ACTIVE:
                continue

            score = 0
            # Name match
            if query_lower in skill.name.lower():
                score += 3
            # Description match
            if query_lower in skill.description.lower():
                score += 2
            # Step match
            for step in skill.steps:
                if query_lower in step.lower():
                    score += 1
                    break
            # Domain match
            for domain in skill.domains:
                if query_lower in domain.lower():
                    score += 1
                    break

            if score > 0:
                scored.append((skill, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:top_k]]

    def recommend_for_context(self, context: dict[str, Any], top_k: int = 5) -> list[Skill]:
        """Recommend skills applicable to context."""
        applicable = []

        for skill in self._skills.values():
            if skill.status != SkillStatus.ACTIVE:
                continue

            if skill.is_applicable(context):
                # Score by success rate and recency
                score = skill.success_rate()
                if skill.last_used:
                    days_ago = (datetime.now(UTC) - skill.last_used).total_seconds() / 86400
                    recency_bonus = max(0, 1 - days_ago / 30) * 0.3
                    score += recency_bonus

                applicable.append((skill, score))

        applicable.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in applicable[:top_k]]

    def get_statistics(self) -> dict[str, Any]:
        """Get library statistics."""
        skills = list(self._skills.values())
        active = [s for s in skills if s.status == SkillStatus.ACTIVE]

        return {
            "total_skills": len(skills),
            "active_skills": len(active),
            "by_type": {
                t.value: len([s for s in active if s.type == t])
                for t in SkillType
            },
            "by_status": {
                s.value: len([sk for sk in skills if sk.status == s])
                for s in SkillStatus
            },
            "total_uses": sum(s.total_uses for s in skills),
            "avg_success_rate": sum(s.success_rate() for s in active) / len(active) if active else 0,
            "domains": list(set(d for s in active for d in s.domains)),
        }


# =============================================================================
# Skill Extractor
# =============================================================================

class SkillExtractor:
    """
    Automatically extract skills from successful episodes.

    Analyzes episode transcripts to identify repeatable patterns
    that can become reusable skills.
    """

    def __init__(self, library: SkillLibrary):
        self.library = library

    def extract_from_episode(
        self,
        episode_id: str,
        actions: list[dict[str, Any]],
        outcome: str,
        success: bool,
        reward: float,
        context: dict[str, Any],
    ) -> list[Skill]:
        """
        Extract skills from an episode.

        Looks for:
        - Successful action sequences
        - Patterns that might generalize
        - Compositions of existing skills
        """
        if not success:
            return []

        extracted = []

        # Extract atomic skills from individual actions
        for i, action in enumerate(actions):
            if self._is_significant_action(action):
                skill = self._create_atomic_skill(
                    action, context, episode_id, reward / max(1, len(actions))
                )
                if skill and not self._skill_exists(skill):
                    extracted.append(skill)

        # Extract sequence skills from action sequences
        if len(actions) >= 2:
            sequence_skills = self._extract_sequences(
                actions, context, episode_id, reward
            )
            extracted.extend(sequence_skills)

        # Try to compose with existing skills
        composite = self._try_compose(actions, context, episode_id)
        if composite:
            extracted.append(composite)

        # Add to library
        for skill in extracted:
            self.library.add(skill)

        return extracted

    def _is_significant_action(self, action: dict[str, Any]) -> bool:
        """Check if an action is significant enough to extract."""
        # Filter out trivial actions
        trivial_patterns = ['print', 'log', 'debug', 'comment']
        action_type = action.get('type', '').lower()
        action_name = action.get('name', '').lower()

        for pattern in trivial_patterns:
            if pattern in action_type or pattern in action_name:
                return False

        return True

    def _create_atomic_skill(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
        episode_id: str,
        reward: float,
    ) -> Skill | None:
        """Create an atomic skill from a single action."""
        action_type = action.get('type', 'unknown')
        action_name = action.get('name', action_type)
        params = action.get('params', {})

        # Generate skill name
        name = f"{action_type}_{action_name}".replace(' ', '_').lower()

        # Generate description
        description = f"Skill extracted from action: {action_name}"

        # Identify preconditions from context
        preconditions = self._infer_preconditions(action, context)

        # Identify postconditions from action effects
        postconditions = self._infer_postconditions(action)

        return Skill(
            name=name,
            description=description,
            type=SkillType.ATOMIC,
            preconditions=preconditions,
            postconditions=postconditions,
            domains=self._infer_domains(action, context),
            steps=[json.dumps(action, default=str)],
            parameters=params,
            avg_reward=reward,
            success_count=1,
            total_uses=1,
            extracted_from=[episode_id],
        )

    def _extract_sequences(
        self,
        actions: list[dict[str, Any]],
        context: dict[str, Any],
        episode_id: str,
        total_reward: float,
    ) -> list[Skill]:
        """Extract sequence skills from consecutive actions."""
        skills = []

        # Try different sequence lengths
        for seq_len in range(2, min(5, len(actions) + 1)):
            for i in range(len(actions) - seq_len + 1):
                sequence = actions[i:i + seq_len]

                # Check if this is a coherent sequence
                if self._is_coherent_sequence(sequence):
                    skill = self._create_sequence_skill(
                        sequence, context, episode_id, total_reward / len(actions) * seq_len
                    )
                    if skill and not self._skill_exists(skill):
                        skills.append(skill)

        return skills

    def _is_coherent_sequence(self, sequence: list[dict[str, Any]]) -> bool:
        """Check if a sequence of actions is coherent (likely intentional)."""
        if len(sequence) < 2:
            return False

        # Check for common patterns
        # Pattern 1: Same type of actions
        types = [a.get('type', '') for a in sequence]
        if len(set(types)) == 1:
            return True

        # Pattern 2: Clear dependency chain
        # (output of one feeds into next)
        # This would require more sophisticated analysis

        return False

    def _create_sequence_skill(
        self,
        sequence: list[dict[str, Any]],
        context: dict[str, Any],
        episode_id: str,
        reward: float,
    ) -> Skill:
        """Create a composite skill from a sequence."""
        action_names = [a.get('name', a.get('type', 'unknown')) for a in sequence]
        name = '_then_'.join(action_names[:3]).replace(' ', '_').lower()[:50]

        return Skill(
            name=name,
            description=f"Sequence of {len(sequence)} actions",
            type=SkillType.COMPOSITE,
            preconditions=self._infer_preconditions(sequence[0], context),
            postconditions=self._infer_postconditions(sequence[-1]),
            domains=self._infer_domains(sequence[0], context),
            steps=[json.dumps(a, default=str) for a in sequence],
            avg_reward=reward,
            success_count=1,
            total_uses=1,
            extracted_from=[episode_id],
        )

    def _try_compose(
        self,
        actions: list[dict[str, Any]],
        context: dict[str, Any],
        episode_id: str,
    ) -> Skill | None:
        """Try to compose new skill from existing skills."""
        # Find existing skills that match action subsequences
        existing = self.library.list_all()
        if not existing:
            return None

        matched_skills = []
        for action in actions:
            action_str = json.dumps(action, default=str)
            for skill in existing:
                if skill.steps and action_str in skill.steps[0]:
                    matched_skills.append(skill.id)
                    break

        # If we matched multiple skills, create a composite
        if len(matched_skills) >= 2:
            return Skill(
                name=f"composed_{len(matched_skills)}_skills",
                description=f"Composition of {len(matched_skills)} existing skills",
                type=SkillType.COMPOSITE,
                sub_skills=matched_skills,
                domains=self._infer_domains(actions[0], context),
                success_count=1,
                total_uses=1,
                extracted_from=[episode_id],
            )

        return None

    def _skill_exists(self, skill: Skill) -> bool:
        """Check if a similar skill already exists."""
        existing = self.library.get_by_name(skill.name)
        return existing is not None

    def _infer_preconditions(self, action: dict[str, Any], context: dict[str, Any]) -> list[str]:
        """Infer preconditions for a skill."""
        preconditions = []

        # From context keys
        for key in list(context.keys())[:5]:
            preconditions.append(f"{key}_available")

        # From action requirements
        if 'requires' in action:
            preconditions.extend(action['requires'])

        return preconditions

    def _infer_postconditions(self, action: dict[str, Any]) -> list[str]:
        """Infer postconditions from action."""
        postconditions = []

        if 'produces' in action:
            postconditions.extend(action['produces'])

        if 'effect' in action:
            postconditions.append(action['effect'])

        return postconditions

    def _infer_domains(self, action: dict[str, Any], context: dict[str, Any]) -> list[str]:
        """Infer domains where this skill applies."""
        domains = []

        # From action type
        action_type = action.get('type', '').lower()
        if 'file' in action_type:
            domains.append('filesystem')
        if 'git' in action_type:
            domains.append('version_control')
        if 'test' in action_type:
            domains.append('testing')
        if 'build' in action_type:
            domains.append('build')

        # From context
        if 'cwd' in context:
            domains.append('local')

        return domains if domains else ['general']


# =============================================================================
# Skill Composer
# =============================================================================

class SkillComposer:
    """
    Compose skills into higher-level strategies.

    Creates composite skills from sequences of atomic skills,
    handles skill dependencies, and generates execution plans.
    """

    def __init__(self, library: SkillLibrary):
        self.library = library

    def compose(
        self,
        skill_ids: list[str],
        name: str,
        description: str,
    ) -> Skill | None:
        """Compose multiple skills into a composite skill."""
        skills = [self.library.get(sid) for sid in skill_ids]
        skills = [s for s in skills if s is not None]

        if len(skills) < 2:
            return None

        # Verify composability
        if not self._can_compose(skills):
            return None

        # Create composite
        composite = Skill(
            name=name,
            description=description,
            type=SkillType.COMPOSITE,
            sub_skills=skill_ids,
            preconditions=skills[0].preconditions.copy(),
            postconditions=skills[-1].postconditions.copy(),
            domains=list(set(d for s in skills for d in s.domains)),
            estimated_duration_ms=sum(s.estimated_duration_ms for s in skills),
        )

        self.library.add(composite)
        return composite

    def _can_compose(self, skills: list[Skill]) -> bool:
        """Check if skills can be composed in sequence."""
        for i in range(len(skills) - 1):
            current = skills[i]
            next_skill = skills[i + 1]

            # Check if current's postconditions satisfy next's preconditions
            # This is a simplified check
            current_effects = set(current.postconditions)
            next_requirements = set(next_skill.preconditions)

            # At least some overlap should exist (relaxed constraint)
            # In practice, would need more sophisticated dependency analysis

        return True

    def get_execution_plan(self, composite_id: str) -> list[Skill]:
        """Get flattened execution plan for a composite skill."""
        composite = self.library.get(composite_id)
        if not composite:
            return []

        if composite.type != SkillType.COMPOSITE:
            return [composite]

        plan = []
        for sub_id in composite.sub_skills:
            sub_skill = self.library.get(sub_id)
            if sub_skill:
                if sub_skill.type == SkillType.COMPOSITE:
                    # Recursive flattening
                    plan.extend(self.get_execution_plan(sub_id))
                else:
                    plan.append(sub_skill)

        return plan

    def suggest_compositions(self, goal: str) -> list[list[str]]:
        """Suggest skill compositions that might achieve a goal."""
        # Search for skills related to the goal
        relevant = self.library.search(goal, top_k=10)

        if len(relevant) < 2:
            return []

        suggestions = []

        # Try different orderings
        for i in range(len(relevant)):
            for j in range(len(relevant)):
                if i != j:
                    pair = [relevant[i].id, relevant[j].id]
                    if self._can_compose([relevant[i], relevant[j]]):
                        suggestions.append(pair)

        return suggestions[:5]


# =============================================================================
# Skill Transfer
# =============================================================================

class SkillTransfer:
    """
    Transfer skills to new domains.

    Maps skill abstractions to new contexts, adapts parameters,
    and tracks transfer success.
    """

    def __init__(self, library: SkillLibrary):
        self.library = library
        self._transfer_history: list[dict[str, Any]] = []

    def transfer_skill(
        self,
        skill_id: str,
        target_domain: str,
        parameter_mapping: dict[str, str] | None = None,
    ) -> Skill | None:
        """
        Transfer a skill to a new domain.

        Creates an adapted version of the skill for the target domain.
        """
        source = self.library.get(skill_id)
        if not source:
            return None

        # Create adapted skill
        adapted = Skill(
            name=f"{source.name}_for_{target_domain}",
            description=f"[Transferred from {source.name}] {source.description}",
            type=source.type,
            status=SkillStatus.DRAFT,  # Start as draft until validated
            preconditions=self._adapt_conditions(source.preconditions, parameter_mapping),
            postconditions=self._adapt_conditions(source.postconditions, parameter_mapping),
            domains=[target_domain],
            steps=self._adapt_steps(source.steps, parameter_mapping),
            sub_skills=source.sub_skills.copy(),
            parameters=self._adapt_parameters(source.parameters, parameter_mapping),
            extracted_from=[f"transferred_from_{skill_id}"],
        )

        self.library.add(adapted)

        # Record transfer
        self._transfer_history.append({
            "source_skill": skill_id,
            "adapted_skill": adapted.id,
            "source_domain": source.domains[0] if source.domains else "unknown",
            "target_domain": target_domain,
            "timestamp": datetime.now(UTC).isoformat(),
            "validated": False,
        })

        return adapted

    def _adapt_conditions(
        self,
        conditions: list[str],
        mapping: dict[str, str] | None,
    ) -> list[str]:
        """Adapt conditions using parameter mapping."""
        if not mapping:
            return conditions.copy()

        adapted = []
        for condition in conditions:
            new_condition = condition
            for old, new in mapping.items():
                new_condition = new_condition.replace(old, new)
            adapted.append(new_condition)

        return adapted

    def _adapt_steps(
        self,
        steps: list[str],
        mapping: dict[str, str] | None,
    ) -> list[str]:
        """Adapt steps using parameter mapping."""
        if not mapping:
            return steps.copy()

        adapted = []
        for step in steps:
            new_step = step
            for old, new in mapping.items():
                new_step = new_step.replace(old, new)
            adapted.append(new_step)

        return adapted

    def _adapt_parameters(
        self,
        params: dict[str, Any],
        mapping: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Adapt parameters using mapping."""
        if not mapping:
            return params.copy()

        adapted = {}
        for key, value in params.items():
            new_key = mapping.get(key, key)
            adapted[new_key] = value

        return adapted

    def validate_transfer(self, adapted_skill_id: str, success: bool):
        """Validate a transferred skill based on use."""
        skill = self.library.get(adapted_skill_id)
        if not skill:
            return

        if success:
            skill.status = SkillStatus.ACTIVE
            skill.success_count += 1
        else:
            skill.failure_count += 1

        skill.total_uses += 1
        self.library.update(skill)

        # Update history
        for record in self._transfer_history:
            if record["adapted_skill"] == adapted_skill_id:
                record["validated"] = success

    def find_transferable_skills(self, target_domain: str) -> list[Skill]:
        """Find skills that might transfer to a target domain."""
        all_skills = self.library.list_all()

        candidates = []
        for skill in all_skills:
            if target_domain not in skill.domains:
                # Score based on type and success rate
                if skill.type == SkillType.ABSTRACT:
                    score = 3  # Abstract skills transfer best
                elif skill.type == SkillType.COMPOSITE:
                    score = 2  # Composites might transfer
                else:
                    score = 1  # Atomic skills are domain-specific

                score *= skill.success_rate()
                candidates.append((skill, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in candidates[:10]]

    def get_transfer_statistics(self) -> dict[str, Any]:
        """Get statistics about skill transfers."""
        if not self._transfer_history:
            return {"total_transfers": 0}

        successful = sum(1 for t in self._transfer_history if t.get("validated", False))

        return {
            "total_transfers": len(self._transfer_history),
            "successful_transfers": successful,
            "success_rate": successful / len(self._transfer_history),
            "by_domain": self._count_by_domain(),
        }

    def _count_by_domain(self) -> dict[str, int]:
        """Count transfers by target domain."""
        counts: dict[str, int] = defaultdict(int)
        for record in self._transfer_history:
            counts[record["target_domain"]] += 1
        return dict(counts)


# =============================================================================
# Unified Skill System
# =============================================================================

class SkillSystem:
    """
    Unified interface for skill extraction, composition, and transfer.
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        self.library = SkillLibrary(skills_dir / "library")
        self.extractor = SkillExtractor(self.library)
        self.composer = SkillComposer(self.library)
        self.transfer = SkillTransfer(self.library)

    def extract_from_episode(
        self,
        episode_id: str,
        actions: list[dict[str, Any]],
        outcome: str,
        success: bool,
        reward: float,
        context: dict[str, Any],
    ) -> list[Skill]:
        """Extract skills from an episode."""
        return self.extractor.extract_from_episode(
            episode_id, actions, outcome, success, reward, context
        )

    def compose_skills(
        self,
        skill_ids: list[str],
        name: str,
        description: str,
    ) -> Skill | None:
        """Compose skills into a higher-level skill."""
        return self.composer.compose(skill_ids, name, description)

    def transfer_skill(
        self,
        skill_id: str,
        target_domain: str,
        parameter_mapping: dict[str, str] | None = None,
    ) -> Skill | None:
        """Transfer a skill to a new domain."""
        return self.transfer.transfer_skill(skill_id, target_domain, parameter_mapping)

    def record_skill_use(
        self,
        skill_id: str,
        success: bool,
        reward: float,
        duration_ms: float,
    ):
        """Record that a skill was used."""
        skill = self.library.get(skill_id)
        if not skill:
            return

        skill.total_uses += 1
        if success:
            skill.success_count += 1
        else:
            skill.failure_count += 1

        # Update running average reward
        n = skill.total_uses
        skill.avg_reward = (skill.avg_reward * (n - 1) + reward) / n
        skill.estimated_duration_ms = (skill.estimated_duration_ms * (n - 1) + duration_ms) / n
        skill.last_used = datetime.now(UTC)

        self.library.update(skill)

    def recommend(self, context: dict[str, Any], goal: str | None = None) -> list[Skill]:
        """Recommend skills for a context and optional goal."""
        recommendations = self.library.recommend_for_context(context)

        if goal:
            goal_relevant = self.library.search(goal)
            # Merge recommendations
            seen = {s.id for s in recommendations}
            for skill in goal_relevant:
                if skill.id not in seen:
                    recommendations.append(skill)

        return recommendations[:10]

    def get_statistics(self) -> dict[str, Any]:
        """Get overall skill system statistics."""
        return {
            "library": self.library.get_statistics(),
            "transfers": self.transfer.get_transfer_statistics(),
        }
