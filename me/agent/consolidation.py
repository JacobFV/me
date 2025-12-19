"""
Memory Consolidation - Sleep-like processing for the agent.

This module implements offline memory processing inspired by biological
sleep consolidation. The agent periodically "consolidates" its memories:
1. Compress episodic memories into generalized schemas
2. Prune low-value or redundant memories
3. Strengthen important connections
4. Extract patterns that weren't obvious in real-time
5. Update procedural knowledge from accumulated experience

Philosophy: The agent doesn't just accumulate memories infinitely.
Like biological systems, it must manage memory resources by:
- Compressing: Many similar episodes → one generalized schema
- Pruning: Remove memories below value threshold
- Strengthening: Rehearse and reinforce important memories
- Abstracting: Find higher-order patterns across experiences

This runs as a background daemon during low-activity periods,
but can also be triggered explicitly (e.g., end of session).
"""

from __future__ import annotations

from datetime import datetime, UTC, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
import json
import hashlib

from pydantic import BaseModel, Field


class ConsolidationPhase(str, Enum):
    """Phases of memory consolidation (like sleep stages)."""
    TAGGING = "tagging"  # Mark memories for processing
    COMPRESSION = "compression"  # Compress similar episodes
    PRUNING = "pruning"  # Remove low-value memories
    STRENGTHENING = "strengthening"  # Reinforce important memories
    ABSTRACTION = "abstraction"  # Extract higher-order patterns
    INTEGRATION = "integration"  # Integrate with existing knowledge


class MemoryValue(BaseModel):
    """Computed value of a memory for retention decisions."""
    memory_id: str
    recency_score: float = 0.0  # How recent
    frequency_score: float = 0.0  # How often accessed
    distinctiveness_score: float = 0.0  # How unique/novel
    utility_score: float = 0.0  # How useful for tasks
    emotional_score: float = 0.0  # How significant (surprise, error, success)
    connection_score: float = 0.0  # How connected to other memories
    composite_value: float = 0.0  # Weighted combination


class Schema(BaseModel):
    """A generalized schema extracted from multiple episodes."""
    id: str
    name: str
    description: str
    # Source episodes that were compressed into this schema
    source_episode_ids: list[str] = Field(default_factory=list)
    # Generalized structure
    common_elements: dict[str, Any] = Field(default_factory=dict)
    variable_slots: dict[str, list[str]] = Field(default_factory=dict)  # slot -> possible values
    typical_sequence: list[str] = Field(default_factory=list)
    success_conditions: list[str] = Field(default_factory=list)
    failure_patterns: list[str] = Field(default_factory=list)
    # Statistics
    instance_count: int = 0
    average_outcome: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PruneDecision(BaseModel):
    """Decision about whether to prune a memory."""
    memory_id: str
    decision: str  # "keep", "prune", "compress"
    reason: str
    value_score: float
    threshold: float


class ConsolidationReport(BaseModel):
    """Report from a consolidation session."""
    session_id: str
    started_at: datetime
    completed_at: datetime
    phases_completed: list[ConsolidationPhase]
    # Statistics
    memories_processed: int = 0
    memories_pruned: int = 0
    memories_compressed: int = 0
    schemas_created: int = 0
    schemas_updated: int = 0
    patterns_extracted: int = 0
    procedures_updated: int = 0
    # Quality metrics
    memory_reduction_ratio: float = 0.0  # How much smaller
    knowledge_density: float = 0.0  # Information per memory


class MemoryValuator:
    """
    Computes the retention value of memories.

    Uses multiple factors to determine which memories are worth keeping:
    - Recency: More recent = higher value (with decay)
    - Frequency: More accessed = higher value
    - Distinctiveness: Unique memories are valuable
    - Utility: Memories that helped accomplish tasks
    - Emotional significance: Surprises, errors, successes
    - Connections: Well-connected memories are valuable
    """

    def __init__(
        self,
        recency_weight: float = 0.2,
        frequency_weight: float = 0.15,
        distinctiveness_weight: float = 0.2,
        utility_weight: float = 0.25,
        emotional_weight: float = 0.1,
        connection_weight: float = 0.1,
    ):
        self.weights = {
            "recency": recency_weight,
            "frequency": frequency_weight,
            "distinctiveness": distinctiveness_weight,
            "utility": utility_weight,
            "emotional": emotional_weight,
            "connection": connection_weight,
        }
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def compute_value(
        self,
        memory_id: str,
        created_at: datetime,
        access_count: int = 0,
        last_accessed: datetime | None = None,
        similarity_to_others: float = 0.5,  # 0 = unique, 1 = duplicate
        task_utility: float = 0.0,  # -1 to 1
        surprise_level: float = 0.0,  # 0 to 1
        error_related: bool = False,
        success_related: bool = False,
        connection_count: int = 0,
    ) -> MemoryValue:
        """Compute the retention value of a memory."""
        now = datetime.now(UTC)

        # Recency: exponential decay over days
        age_days = (now - created_at).total_seconds() / 86400
        recency_score = 1.0 / (1.0 + age_days * 0.1)  # Slow decay

        # Frequency: log scale of access count
        import math
        frequency_score = math.log1p(access_count) / 5.0  # Normalize to ~1 for 148 accesses
        frequency_score = min(1.0, frequency_score)

        # Distinctiveness: inverse of similarity
        distinctiveness_score = 1.0 - similarity_to_others

        # Utility: normalize to 0-1
        utility_score = (task_utility + 1.0) / 2.0

        # Emotional: surprise + error/success bonuses
        emotional_score = surprise_level
        if error_related:
            emotional_score = min(1.0, emotional_score + 0.3)
        if success_related:
            emotional_score = min(1.0, emotional_score + 0.3)

        # Connections: log scale
        connection_score = math.log1p(connection_count) / 3.0
        connection_score = min(1.0, connection_score)

        # Composite
        composite = (
            self.weights["recency"] * recency_score +
            self.weights["frequency"] * frequency_score +
            self.weights["distinctiveness"] * distinctiveness_score +
            self.weights["utility"] * utility_score +
            self.weights["emotional"] * emotional_score +
            self.weights["connection"] * connection_score
        )

        return MemoryValue(
            memory_id=memory_id,
            recency_score=recency_score,
            frequency_score=frequency_score,
            distinctiveness_score=distinctiveness_score,
            utility_score=utility_score,
            emotional_score=emotional_score,
            connection_score=connection_score,
            composite_value=composite,
        )


class EpisodeCompressor:
    """
    Compresses multiple similar episodes into generalized schemas.

    When the agent has many episodes that follow similar patterns,
    they can be compressed into a single schema that captures:
    - The common structure
    - Variable elements (slots)
    - Typical sequences
    - Success/failure patterns
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.schemas_dir = body_dir / "memory" / "schemas"
        self.schemas_dir.mkdir(parents=True, exist_ok=True)

    def find_similar_episodes(
        self,
        episodes: list[dict[str, Any]],
        similarity_threshold: float = 0.7,
    ) -> list[list[dict[str, Any]]]:
        """Group episodes by similarity."""
        if not episodes:
            return []

        # Simple clustering by domain/task type
        clusters: dict[str, list[dict[str, Any]]] = {}

        for ep in episodes:
            # Create a key from domain and action types
            domain = ep.get("domain", "general")
            actions = ep.get("actions", [])
            action_types = tuple(sorted(set(
                a.get("type", "unknown") for a in actions[:5]  # First 5 actions
            )))
            key = f"{domain}:{action_types}"

            if key not in clusters:
                clusters[key] = []
            clusters[key].append(ep)

        # Return clusters with multiple episodes
        return [eps for eps in clusters.values() if len(eps) >= 2]

    def compress_to_schema(
        self,
        episodes: list[dict[str, Any]],
        schema_name: str,
    ) -> Schema:
        """Compress multiple episodes into a single schema."""
        # Extract common elements
        common_elements: dict[str, Any] = {}
        variable_slots: dict[str, list[str]] = {}

        # Find common domain
        domains = [ep.get("domain", "general") for ep in episodes]
        if len(set(domains)) == 1:
            common_elements["domain"] = domains[0]
        else:
            variable_slots["domain"] = list(set(domains))

        # Find common action patterns
        action_sequences = [
            [a.get("type", "unknown") for a in ep.get("actions", [])]
            for ep in episodes
        ]

        # Find longest common prefix
        if action_sequences:
            min_len = min(len(seq) for seq in action_sequences)
            common_prefix = []
            for i in range(min_len):
                actions_at_i = set(seq[i] for seq in action_sequences)
                if len(actions_at_i) == 1:
                    common_prefix.append(list(actions_at_i)[0])
                else:
                    variable_slots[f"action_{i}"] = list(actions_at_i)
                    break
            common_elements["action_prefix"] = common_prefix

        # Extract outcomes
        outcomes = [ep.get("outcome", {}) for ep in episodes]
        successes = [o.get("success", False) for o in outcomes]
        average_outcome = sum(1 for s in successes if s) / len(successes) if successes else 0.0

        # Build schema
        schema_id = hashlib.md5(schema_name.encode()).hexdigest()[:12]
        schema = Schema(
            id=schema_id,
            name=schema_name,
            description=f"Schema for {len(episodes)} similar episodes in {common_elements.get('domain', 'general')} domain",
            source_episode_ids=[ep.get("id", "") for ep in episodes],
            common_elements=common_elements,
            variable_slots=variable_slots,
            typical_sequence=common_elements.get("action_prefix", []),
            instance_count=len(episodes),
            average_outcome=average_outcome,
        )

        # Save schema
        self._save_schema(schema)

        return schema

    def _save_schema(self, schema: Schema) -> None:
        """Save schema to disk."""
        path = self.schemas_dir / f"{schema.id}.json"
        path.write_text(json.dumps(schema.model_dump(mode='json'), indent=2, default=str))

    def load_schemas(self) -> list[Schema]:
        """Load all schemas."""
        schemas = []
        for path in self.schemas_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                schemas.append(Schema.model_validate(data))
            except Exception:
                continue
        return schemas


class MemoryPruner:
    """
    Prunes low-value memories to manage storage.

    Pruning decisions consider:
    - Memory value (from MemoryValuator)
    - Whether the memory is covered by a schema
    - Minimum retention periods
    - Storage constraints
    """

    def __init__(
        self,
        value_threshold: float = 0.3,
        min_retention_hours: int = 24,
        max_memories: int = 10000,
    ):
        self.value_threshold = value_threshold
        self.min_retention = timedelta(hours=min_retention_hours)
        self.max_memories = max_memories
        self.valuator = MemoryValuator()

    def evaluate_for_pruning(
        self,
        memory_id: str,
        memory_data: dict[str, Any],
        covered_by_schema: bool = False,
    ) -> PruneDecision:
        """Evaluate whether a memory should be pruned."""
        created_at = memory_data.get("created_at", datetime.now(UTC))
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        # Check minimum retention
        age = datetime.now(UTC) - created_at
        if age < self.min_retention:
            return PruneDecision(
                memory_id=memory_id,
                decision="keep",
                reason="Below minimum retention period",
                value_score=1.0,
                threshold=self.value_threshold,
            )

        # Compute value
        value = self.valuator.compute_value(
            memory_id=memory_id,
            created_at=created_at,
            access_count=memory_data.get("access_count", 0),
            similarity_to_others=0.8 if covered_by_schema else 0.3,
            task_utility=memory_data.get("utility", 0.0),
            surprise_level=memory_data.get("surprise", 0.0),
            error_related=memory_data.get("is_error", False),
            success_related=memory_data.get("is_success", False),
            connection_count=len(memory_data.get("connections", [])),
        )

        # Decide
        if covered_by_schema and value.composite_value < self.value_threshold:
            return PruneDecision(
                memory_id=memory_id,
                decision="compress",
                reason="Low value and covered by schema",
                value_score=value.composite_value,
                threshold=self.value_threshold,
            )

        if value.composite_value < self.value_threshold * 0.5:
            return PruneDecision(
                memory_id=memory_id,
                decision="prune",
                reason="Value below threshold",
                value_score=value.composite_value,
                threshold=self.value_threshold,
            )

        return PruneDecision(
            memory_id=memory_id,
            decision="keep",
            reason="Value above threshold",
            value_score=value.composite_value,
            threshold=self.value_threshold,
        )


class PatternAbstractor:
    """
    Extracts higher-order patterns during consolidation.

    Unlike real-time pattern detection, this runs "offline" and can:
    - Find patterns across longer time periods
    - Detect meta-patterns (patterns of patterns)
    - Build hierarchical abstractions
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.patterns_dir = body_dir / "memory" / "patterns"
        self.patterns_dir.mkdir(parents=True, exist_ok=True)

    def extract_temporal_patterns(
        self,
        episodes: list[dict[str, Any]],
        time_window_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Find patterns that occur within time windows."""
        patterns = []

        # Sort episodes by time
        sorted_eps = sorted(
            episodes,
            key=lambda e: e.get("started_at", datetime.min)
        )

        # Find sequences within time windows
        for i, ep in enumerate(sorted_eps):
            window_eps = []
            ep_time = ep.get("started_at", datetime.min)
            if isinstance(ep_time, str):
                ep_time = datetime.fromisoformat(ep_time.replace("Z", "+00:00"))

            for j in range(i + 1, len(sorted_eps)):
                next_ep = sorted_eps[j]
                next_time = next_ep.get("started_at", datetime.min)
                if isinstance(next_time, str):
                    next_time = datetime.fromisoformat(next_time.replace("Z", "+00:00"))

                if (next_time - ep_time).total_seconds() <= time_window_hours * 3600:
                    window_eps.append(next_ep)
                else:
                    break

            if len(window_eps) >= 2:
                # Check for pattern in window
                domains = [e.get("domain", "") for e in [ep] + window_eps]
                if len(set(domains)) == 1:
                    patterns.append({
                        "type": "temporal_cluster",
                        "domain": domains[0],
                        "episode_count": len(window_eps) + 1,
                        "time_span_hours": time_window_hours,
                    })

        return patterns

    def extract_causal_patterns(
        self,
        episodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Find patterns that suggest causal relationships."""
        patterns = []

        # Look for action → outcome patterns
        action_outcomes: dict[str, list[bool]] = {}

        for ep in episodes:
            actions = ep.get("actions", [])
            outcome = ep.get("outcome", {}).get("success", False)

            for action in actions:
                action_type = action.get("type", "unknown")
                if action_type not in action_outcomes:
                    action_outcomes[action_type] = []
                action_outcomes[action_type].append(outcome)

        # Find actions with strong outcome correlations
        for action_type, outcomes in action_outcomes.items():
            if len(outcomes) >= 5:
                success_rate = sum(outcomes) / len(outcomes)
                if success_rate > 0.8:
                    patterns.append({
                        "type": "causal_positive",
                        "action": action_type,
                        "success_rate": success_rate,
                        "sample_size": len(outcomes),
                    })
                elif success_rate < 0.2:
                    patterns.append({
                        "type": "causal_negative",
                        "action": action_type,
                        "success_rate": success_rate,
                        "sample_size": len(outcomes),
                    })

        return patterns

    def build_abstraction_hierarchy(
        self,
        schemas: list[Schema],
    ) -> dict[str, Any]:
        """Build hierarchical abstractions from schemas."""
        hierarchy: dict[str, Any] = {
            "domains": {},
            "meta_schemas": [],
        }

        # Group schemas by domain
        for schema in schemas:
            domain = schema.common_elements.get("domain", "general")
            if domain not in hierarchy["domains"]:
                hierarchy["domains"][domain] = {
                    "schemas": [],
                    "common_patterns": [],
                }
            hierarchy["domains"][domain]["schemas"].append(schema.id)

        # Find meta-patterns (patterns across schemas)
        for domain, data in hierarchy["domains"].items():
            if len(data["schemas"]) >= 3:
                # This domain has enough schemas to potentially have meta-patterns
                domain_schemas = [s for s in schemas if s.common_elements.get("domain") == domain]

                # Look for common action prefixes across schemas
                prefixes = [s.typical_sequence for s in domain_schemas if s.typical_sequence]
                if prefixes:
                    # Find common prefix across all
                    min_len = min(len(p) for p in prefixes)
                    common = []
                    for i in range(min_len):
                        if len(set(p[i] for p in prefixes)) == 1:
                            common.append(prefixes[0][i])
                        else:
                            break

                    if common:
                        data["common_patterns"].append({
                            "type": "action_prefix",
                            "pattern": common,
                        })

        return hierarchy


class ConsolidationEngine:
    """
    Main engine for memory consolidation.

    Coordinates all consolidation activities:
    1. Tag memories for processing
    2. Compress similar episodes
    3. Prune low-value memories
    4. Strengthen important memories
    5. Extract patterns
    6. Integrate knowledge
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.valuator = MemoryValuator()
        self.compressor = EpisodeCompressor(body_dir)
        self.pruner = MemoryPruner()
        self.abstractor = PatternAbstractor(body_dir)

        self.consolidation_dir = body_dir / "memory" / "consolidation"
        self.consolidation_dir.mkdir(parents=True, exist_ok=True)

    def run_consolidation(
        self,
        episodes: list[dict[str, Any]],
        full_consolidation: bool = True,
    ) -> ConsolidationReport:
        """Run a consolidation session."""
        session_id = hashlib.md5(
            f"{datetime.now(UTC).isoformat()}".encode()
        ).hexdigest()[:12]

        started_at = datetime.now(UTC)
        phases_completed = []

        memories_processed = len(episodes)
        memories_pruned = 0
        memories_compressed = 0
        schemas_created = 0
        patterns_extracted = 0

        # Phase 1: Tagging
        phases_completed.append(ConsolidationPhase.TAGGING)
        tagged_for_compression = []
        tagged_for_pruning = []

        for ep in episodes:
            value = self.valuator.compute_value(
                memory_id=ep.get("id", ""),
                created_at=datetime.fromisoformat(
                    str(ep.get("started_at", datetime.now(UTC).isoformat())).replace("Z", "+00:00")
                ) if ep.get("started_at") else datetime.now(UTC),
                access_count=ep.get("access_count", 0),
                task_utility=ep.get("utility", 0.0),
            )

            if value.composite_value < 0.3:
                tagged_for_pruning.append(ep)
            elif value.composite_value < 0.5:
                tagged_for_compression.append(ep)

        # Phase 2: Compression
        if full_consolidation:
            phases_completed.append(ConsolidationPhase.COMPRESSION)
            similar_groups = self.compressor.find_similar_episodes(episodes)

            for group in similar_groups:
                if len(group) >= 3:  # Only compress groups of 3+
                    domain = group[0].get("domain", "general")
                    schema = self.compressor.compress_to_schema(
                        group,
                        f"{domain}_schema_{schemas_created}",
                    )
                    schemas_created += 1
                    memories_compressed += len(group)

        # Phase 3: Pruning
        phases_completed.append(ConsolidationPhase.PRUNING)
        schemas = self.compressor.load_schemas()
        schema_episode_ids = set()
        for schema in schemas:
            schema_episode_ids.update(schema.source_episode_ids)

        for ep in tagged_for_pruning:
            ep_id = ep.get("id", "")
            covered = ep_id in schema_episode_ids
            decision = self.pruner.evaluate_for_pruning(ep_id, ep, covered)
            if decision.decision == "prune":
                memories_pruned += 1

        # Phase 4: Strengthening (mark important memories)
        phases_completed.append(ConsolidationPhase.STRENGTHENING)
        # This would update access timestamps and boost values

        # Phase 5: Abstraction
        if full_consolidation:
            phases_completed.append(ConsolidationPhase.ABSTRACTION)
            temporal_patterns = self.abstractor.extract_temporal_patterns(episodes)
            causal_patterns = self.abstractor.extract_causal_patterns(episodes)
            patterns_extracted = len(temporal_patterns) + len(causal_patterns)

            if schemas:
                hierarchy = self.abstractor.build_abstraction_hierarchy(schemas)
                # Save hierarchy
                hierarchy_path = self.consolidation_dir / "hierarchy.json"
                hierarchy_path.write_text(json.dumps(hierarchy, indent=2, default=str))

        # Phase 6: Integration
        phases_completed.append(ConsolidationPhase.INTEGRATION)

        completed_at = datetime.now(UTC)

        # Calculate metrics
        original_count = memories_processed
        final_count = original_count - memories_pruned
        reduction_ratio = (original_count - final_count) / original_count if original_count > 0 else 0
        knowledge_density = (schemas_created + patterns_extracted) / max(1, final_count)

        report = ConsolidationReport(
            session_id=session_id,
            started_at=started_at,
            completed_at=completed_at,
            phases_completed=phases_completed,
            memories_processed=memories_processed,
            memories_pruned=memories_pruned,
            memories_compressed=memories_compressed,
            schemas_created=schemas_created,
            patterns_extracted=patterns_extracted,
            memory_reduction_ratio=reduction_ratio,
            knowledge_density=knowledge_density,
        )

        # Save report
        report_path = self.consolidation_dir / f"report-{session_id}.json"
        report_path.write_text(json.dumps(report.model_dump(mode='json'), indent=2, default=str))

        return report

    def get_consolidation_stats(self) -> dict[str, Any]:
        """Get statistics about consolidation history."""
        reports = []
        for path in self.consolidation_dir.glob("report-*.json"):
            try:
                data = json.loads(path.read_text())
                reports.append(ConsolidationReport.model_validate(data))
            except Exception:
                continue

        if not reports:
            return {"sessions": 0}

        total_pruned = sum(r.memories_pruned for r in reports)
        total_compressed = sum(r.memories_compressed for r in reports)
        total_schemas = sum(r.schemas_created for r in reports)

        return {
            "sessions": len(reports),
            "total_memories_pruned": total_pruned,
            "total_memories_compressed": total_compressed,
            "total_schemas_created": total_schemas,
            "average_reduction_ratio": sum(r.memory_reduction_ratio for r in reports) / len(reports),
            "last_consolidation": max(r.completed_at for r in reports).isoformat(),
        }


class ConsolidationDaemon:
    """
    Daemon that triggers consolidation during low-activity periods.

    Runs as a background process, monitoring activity levels and
    triggering consolidation when appropriate (like sleep).
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.engine = ConsolidationEngine(body_dir)
        self.last_activity: datetime = datetime.now(UTC)
        self.consolidation_cooldown = timedelta(hours=4)
        self.last_consolidation: datetime | None = None

    def record_activity(self) -> None:
        """Record that activity occurred."""
        self.last_activity = datetime.now(UTC)

    def should_consolidate(self) -> bool:
        """Check if consolidation should run."""
        now = datetime.now(UTC)

        # Check cooldown
        if self.last_consolidation:
            if now - self.last_consolidation < self.consolidation_cooldown:
                return False

        # Check inactivity period (30 minutes)
        inactivity = now - self.last_activity
        if inactivity < timedelta(minutes=30):
            return False

        return True

    def trigger_if_appropriate(self, episodes: list[dict[str, Any]]) -> ConsolidationReport | None:
        """Trigger consolidation if appropriate."""
        if not self.should_consolidate():
            return None

        if len(episodes) < 10:  # Not enough to consolidate
            return None

        self.last_consolidation = datetime.now(UTC)
        return self.engine.run_consolidation(episodes)


class MemoryConsolidationSystem:
    """
    Unified system for memory consolidation.

    Provides a single interface for:
    - Manual consolidation triggers
    - Automatic daemon-based consolidation
    - Schema and pattern access
    - Consolidation statistics
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.engine = ConsolidationEngine(body_dir)
        self.daemon = ConsolidationDaemon(body_dir)
        self.compressor = self.engine.compressor
        self.abstractor = self.engine.abstractor

    def consolidate(
        self,
        episodes: list[dict[str, Any]],
        full: bool = True,
    ) -> ConsolidationReport:
        """Run manual consolidation."""
        return self.engine.run_consolidation(episodes, full)

    def maybe_consolidate(
        self,
        episodes: list[dict[str, Any]],
    ) -> ConsolidationReport | None:
        """Run consolidation if conditions are appropriate."""
        return self.daemon.trigger_if_appropriate(episodes)

    def record_activity(self) -> None:
        """Record that activity occurred (resets inactivity timer)."""
        self.daemon.record_activity()

    def get_schemas(self) -> list[Schema]:
        """Get all schemas."""
        return self.compressor.load_schemas()

    def get_schema(self, schema_id: str) -> Schema | None:
        """Get a specific schema."""
        schemas = self.compressor.load_schemas()
        for schema in schemas:
            if schema.id == schema_id:
                return schema
        return None

    def get_abstraction_hierarchy(self) -> dict[str, Any]:
        """Get the current abstraction hierarchy."""
        hierarchy_path = self.engine.consolidation_dir / "hierarchy.json"
        if hierarchy_path.exists():
            return json.loads(hierarchy_path.read_text())
        return {"domains": {}, "meta_schemas": []}

    def get_stats(self) -> dict[str, Any]:
        """Get consolidation statistics."""
        return self.engine.get_consolidation_stats()

    def compute_memory_value(
        self,
        memory_id: str,
        memory_data: dict[str, Any],
    ) -> MemoryValue:
        """Compute the retention value of a memory."""
        created_at = memory_data.get("created_at", datetime.now(UTC))
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        return self.engine.valuator.compute_value(
            memory_id=memory_id,
            created_at=created_at,
            access_count=memory_data.get("access_count", 0),
            task_utility=memory_data.get("utility", 0.0),
            surprise_level=memory_data.get("surprise", 0.0),
            error_related=memory_data.get("is_error", False),
            success_related=memory_data.get("is_success", False),
            connection_count=len(memory_data.get("connections", [])),
        )
