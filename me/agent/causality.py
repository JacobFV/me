"""
Causal Reasoning - Understanding cause-effect relationships.

Intelligence requires understanding not just correlation but causation.
This module provides:
- Causal graph construction from observations
- Counterfactual reasoning (what if I had done X instead?)
- Intervention planning (how do I change Y?)
- Causal discovery from data

Philosophy: The world is not just patterns—it's mechanisms. Understanding
causality means understanding the deep structure of how things work, enabling
reasoning about unseen possibilities and hypothetical scenarios.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any


# =============================================================================
# Causal Graph
# =============================================================================

class CausalRelationType(str, Enum):
    """Types of causal relationships."""
    CAUSES = "causes"              # A causes B
    PREVENTS = "prevents"          # A prevents B
    ENABLES = "enables"            # A enables B (precondition)
    PROBABILISTIC = "probabilistic"  # A makes B more likely


@dataclass
class CausalEdge:
    """An edge in the causal graph."""
    cause: str
    effect: str
    relation_type: CausalRelationType
    confidence: float = 0.5  # 0-1, how confident we are
    strength: float = 0.5    # 0-1, how strong the effect
    evidence_count: int = 0
    observations: list[str] = field(default_factory=list)  # IDs of supporting observations
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "evidence_count": self.evidence_count,
            "observations": self.observations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CausalEdge":
        return cls(
            cause=data["cause"],
            effect=data["effect"],
            relation_type=CausalRelationType(data["relation_type"]),
            confidence=data.get("confidence", 0.5),
            strength=data.get("strength", 0.5),
            evidence_count=data.get("evidence_count", 0),
            observations=data.get("observations", []),
            metadata=data.get("metadata", {}),
        )


class CausalGraph:
    """
    A directed graph representing causal relationships.

    Nodes are variables (states, actions, events).
    Edges are causal relationships.
    """

    def __init__(self):
        self._nodes: set[str] = set()
        self._edges: dict[tuple[str, str], CausalEdge] = {}
        self._parents: dict[str, list[str]] = defaultdict(list)  # What causes this
        self._children: dict[str, list[str]] = defaultdict(list)  # What this causes

    def add_node(self, node: str):
        """Add a node (variable) to the graph."""
        self._nodes.add(node)

    def add_edge(self, edge: CausalEdge):
        """Add or update a causal edge."""
        key = (edge.cause, edge.effect)

        # Add nodes if they don't exist
        self.add_node(edge.cause)
        self.add_node(edge.effect)

        # Add/update edge
        if key in self._edges:
            # Update existing edge (increase confidence)
            existing = self._edges[key]
            existing.evidence_count += edge.evidence_count
            existing.observations.extend(edge.observations)

            # Update confidence (Bayesian-ish update)
            prior = existing.confidence
            new_evidence = edge.confidence
            existing.confidence = (prior + new_evidence) / 2

        else:
            self._edges[key] = edge
            self._parents[edge.effect].append(edge.cause)
            self._children[edge.cause].append(edge.effect)

    def get_edge(self, cause: str, effect: str) -> CausalEdge | None:
        """Get the causal edge between cause and effect."""
        return self._edges.get((cause, effect))

    def get_causes(self, effect: str) -> list[str]:
        """Get all direct causes of an effect."""
        return self._parents.get(effect, [])

    def get_effects(self, cause: str) -> list[str]:
        """Get all direct effects of a cause."""
        return self._children.get(cause, [])

    def get_ancestors(self, node: str, max_depth: int = 10) -> set[str]:
        """Get all ancestors (transitive causes) of a node."""
        ancestors = set()
        to_visit = [(node, 0)]
        visited = set()

        while to_visit:
            current, depth = to_visit.pop()

            if depth >= max_depth or current in visited:
                continue

            visited.add(current)

            for parent in self._parents.get(current, []):
                ancestors.add(parent)
                to_visit.append((parent, depth + 1))

        return ancestors

    def get_descendants(self, node: str, max_depth: int = 10) -> set[str]:
        """Get all descendants (transitive effects) of a node."""
        descendants = set()
        to_visit = [(node, 0)]
        visited = set()

        while to_visit:
            current, depth = to_visit.pop()

            if depth >= max_depth or current in visited:
                continue

            visited.add(current)

            for child in self._children.get(current, []):
                descendants.add(child)
                to_visit.append((child, depth + 1))

        return descendants

    def find_paths(self, start: str, end: str, max_length: int = 5) -> list[list[str]]:
        """Find all causal paths from start to end."""
        paths = []

        def _dfs(current: str, path: list[str]):
            if len(path) > max_length:
                return

            if current == end:
                paths.append(path.copy())
                return

            for child in self._children.get(current, []):
                if child not in path:  # Avoid cycles
                    path.append(child)
                    _dfs(child, path)
                    path.pop()

        _dfs(start, [start])
        return paths

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "nodes": list(self._nodes),
            "edges": [edge.to_dict() for edge in self._edges.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CausalGraph":
        """Deserialize from dictionary."""
        graph = cls()

        for node in data.get("nodes", []):
            graph.add_node(node)

        for edge_data in data.get("edges", []):
            edge = CausalEdge.from_dict(edge_data)
            graph.add_edge(edge)

        return graph


# =============================================================================
# Observations
# =============================================================================

@dataclass
class CausalObservation:
    """An observation of a potential causal relationship."""
    id: str
    timestamp: datetime
    action: str
    preconditions: dict[str, Any]
    outcome: dict[str, Any]
    success: bool
    duration_ms: float
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "preconditions": self.preconditions,
            "outcome": self.outcome,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CausalObservation":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=data["action"],
            preconditions=data["preconditions"],
            outcome=data["outcome"],
            success=data["success"],
            duration_ms=data["duration_ms"],
            context=data.get("context", {}),
        )


# =============================================================================
# Causal Discovery
# =============================================================================

class CausalDiscovery:
    """
    Discover causal relationships from observations.

    Uses simple heuristics:
    - Temporal precedence (cause before effect)
    - Consistency (same cause -> same effect)
    - Intervention (when I do X, Y happens)
    """

    def __init__(self):
        self._observations: list[CausalObservation] = []

    def add_observation(self, obs: CausalObservation):
        """Add an observation."""
        self._observations.append(obs)

    def infer_relationships(
        self,
        min_evidence: int = 2,
        min_confidence: float = 0.6,
    ) -> list[CausalEdge]:
        """
        Infer causal relationships from observations.

        Returns edges with sufficient evidence and confidence.
        """
        edges = []

        # Group observations by action
        by_action = defaultdict(list)
        for obs in self._observations:
            by_action[obs.action].append(obs)

        # For each action, look for consistent effects
        for action, obs_list in by_action.items():
            if len(obs_list) < min_evidence:
                continue

            # Count outcomes
            outcome_counts = defaultdict(int)
            for obs in obs_list:
                # Simplify outcome to key outcomes
                for key, value in obs.outcome.items():
                    outcome_counts[(key, value)] += 1

            # Find consistent outcomes
            for (key, value), count in outcome_counts.items():
                consistency = count / len(obs_list)

                if consistency >= min_confidence:
                    # Infer causal edge
                    edge = CausalEdge(
                        cause=action,
                        effect=f"{key}={value}",
                        relation_type=CausalRelationType.CAUSES,
                        confidence=consistency,
                        strength=consistency,
                        evidence_count=count,
                        observations=[obs.id for obs in obs_list],
                    )
                    edges.append(edge)

        return edges

    def detect_confounders(self) -> list[tuple[str, str, str]]:
        """
        Detect potential confounding variables.

        Returns triples (confounder, cause, effect) where confounder
        may be influencing both cause and effect.
        """
        # Simplified: look for variables that appear in many observations
        confounders = []

        variable_counts = defaultdict(int)
        for obs in self._observations:
            for key in obs.preconditions.keys():
                variable_counts[key] += 1

        # High-frequency variables might be confounders
        high_freq = [k for k, v in variable_counts.items() if v > len(self._observations) * 0.5]

        # Placeholder: return as potential confounders
        return []  # Simplified


# =============================================================================
# Counterfactual Reasoning
# =============================================================================

@dataclass
class Counterfactual:
    """A counterfactual scenario: what if X instead of Y?"""
    actual_action: str
    counterfactual_action: str
    actual_outcome: dict[str, Any]
    predicted_counterfactual_outcome: dict[str, Any]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual_action": self.actual_action,
            "counterfactual_action": self.counterfactual_action,
            "actual_outcome": self.actual_outcome,
            "predicted_counterfactual_outcome": self.predicted_counterfactual_outcome,
            "confidence": self.confidence,
        }


class CounterfactualReasoner:
    """
    Reason about counterfactuals using the causal graph.

    Answers questions like:
    - If I had done X instead of Y, what would have happened?
    - What caused Z to happen (vs. not happen)?
    """

    def __init__(self, graph: CausalGraph, discovery: CausalDiscovery):
        self.graph = graph
        self.discovery = discovery

    def reason_counterfactual(
        self,
        actual_action: str,
        counterfactual_action: str,
        context: dict[str, Any],
    ) -> Counterfactual:
        """
        Reason about what would have happened with a different action.

        Uses causal graph to predict counterfactual outcomes.
        """
        # Get effects of both actions
        actual_effects = self.graph.get_effects(actual_action)
        cf_effects = self.graph.get_effects(counterfactual_action)

        # Predict actual outcome (simplified: use most common effect)
        actual_outcome = {}
        for effect in actual_effects:
            edge = self.graph.get_edge(actual_action, effect)
            if edge:
                actual_outcome[effect] = edge.strength

        # Predict counterfactual outcome
        cf_outcome = {}
        for effect in cf_effects:
            edge = self.graph.get_edge(counterfactual_action, effect)
            if edge:
                cf_outcome[effect] = edge.strength

        # Confidence is average of edge confidences
        all_edges = [
            self.graph.get_edge(actual_action, e)
            for e in actual_effects
        ] + [
            self.graph.get_edge(counterfactual_action, e)
            for e in cf_effects
        ]
        valid_edges = [e for e in all_edges if e]
        confidence = sum(e.confidence for e in valid_edges) / max(1, len(valid_edges))

        return Counterfactual(
            actual_action=actual_action,
            counterfactual_action=counterfactual_action,
            actual_outcome=actual_outcome,
            predicted_counterfactual_outcome=cf_outcome,
            confidence=confidence,
        )

    def explain_outcome(self, outcome: str) -> list[str]:
        """
        Explain an outcome by tracing back to root causes.

        Returns a list of causes in order from proximal to distal.
        """
        causes = []

        # Get direct causes
        direct_causes = self.graph.get_causes(outcome)
        causes.extend(direct_causes)

        # Get ancestors (root causes)
        for cause in direct_causes:
            ancestors = self.graph.get_ancestors(cause, max_depth=3)
            causes.extend(ancestors)

        # Remove duplicates while preserving order
        seen = set()
        unique_causes = []
        for cause in causes:
            if cause not in seen:
                seen.add(cause)
                unique_causes.append(cause)

        return unique_causes


# =============================================================================
# Intervention Planning
# =============================================================================

@dataclass
class Intervention:
    """A planned intervention to achieve a goal."""
    goal: str
    actions: list[str]
    expected_effect: dict[str, Any]
    confidence: float
    side_effects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "actions": self.actions,
            "expected_effect": self.expected_effect,
            "confidence": self.confidence,
            "side_effects": self.side_effects,
        }


class InterventionPlanner:
    """
    Plan interventions to achieve goals.

    Uses the causal graph to find action sequences that lead to desired outcomes.
    """

    def __init__(self, graph: CausalGraph):
        self.graph = graph

    def plan_intervention(self, goal: str) -> Intervention | None:
        """
        Plan an intervention to achieve a goal.

        Returns the best action sequence if one exists.
        """
        # Find all actions that could lead to the goal
        causes = self.graph.get_causes(goal)

        if not causes:
            return None

        # Find the strongest causal path
        best_cause = None
        best_confidence = 0.0

        for cause in causes:
            edge = self.graph.get_edge(cause, goal)
            if edge and edge.confidence > best_confidence:
                best_confidence = edge.confidence
                best_cause = cause

        if not best_cause:
            return None

        # Build action sequence
        actions = [best_cause]

        # Check for preconditions (what enables this action?)
        preconditions = self.graph.get_causes(best_cause)
        actions = preconditions + actions

        # Predict side effects
        side_effects = []
        for action in actions:
            effects = self.graph.get_effects(action)
            for effect in effects:
                if effect != goal:
                    side_effects.append(effect)

        return Intervention(
            goal=goal,
            actions=actions,
            expected_effect={goal: 1.0},
            confidence=best_confidence,
            side_effects=side_effects,
        )


# =============================================================================
# Unified Causal Reasoner
# =============================================================================

class CausalReasoner:
    """
    Unified causal reasoning system.

    Combines:
    - Causal graph (structure)
    - Discovery (learning from observations)
    - Counterfactual reasoning (what if)
    - Intervention planning (how to achieve goals)
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.graph_path = self.model_dir / "causal_graph.json"
        self.observations_path = self.model_dir / "observations.jsonl"

        # Core components
        self.graph = CausalGraph()
        self.discovery = CausalDiscovery()

        # Load existing
        self._load()

    def observe(self, observation: CausalObservation):
        """Add a new observation."""
        self.discovery.add_observation(observation)

        # Append to file
        with open(self.observations_path, 'a') as f:
            f.write(json.dumps(observation.to_dict(), default=str) + '\n')

    def update_graph(self, min_evidence: int = 2):
        """
        Update causal graph based on accumulated observations.

        Should be called periodically to learn new relationships.
        """
        # Infer new edges
        new_edges = self.discovery.infer_relationships(min_evidence=min_evidence)

        # Add to graph
        for edge in new_edges:
            self.graph.add_edge(edge)

        # Save
        self.save()

    def reason_counterfactual(
        self,
        actual_action: str,
        counterfactual_action: str,
        context: dict[str, Any],
    ) -> Counterfactual:
        """Reason about a counterfactual scenario."""
        reasoner = CounterfactualReasoner(self.graph, self.discovery)
        return reasoner.reason_counterfactual(actual_action, counterfactual_action, context)

    def explain(self, outcome: str) -> list[str]:
        """Explain an outcome by tracing causes."""
        reasoner = CounterfactualReasoner(self.graph, self.discovery)
        return reasoner.explain_outcome(outcome)

    def plan_intervention(self, goal: str) -> Intervention | None:
        """Plan an intervention to achieve a goal."""
        planner = InterventionPlanner(self.graph)
        return planner.plan_intervention(goal)

    def save(self):
        """Save causal graph to disk."""
        with open(self.graph_path, 'w') as f:
            json.dump(self.graph.to_dict(), f, indent=2)

    def _load(self):
        """Load causal graph and observations from disk."""
        # Load graph
        if self.graph_path.exists():
            try:
                with open(self.graph_path, 'r') as f:
                    data = json.load(f)
                self.graph = CausalGraph.from_dict(data)
            except Exception:
                pass

        # Load observations
        if self.observations_path.exists():
            try:
                with open(self.observations_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        data = json.loads(line)
                        obs = CausalObservation.from_dict(data)
                        self.discovery.add_observation(obs)
            except Exception:
                pass
