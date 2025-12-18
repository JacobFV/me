"""
Experience Replay & Learning System.

Enables the agent to learn from past episodes by replaying and analyzing them,
comparing predictions vs outcomes, updating procedures, and extracting patterns.

Philosophy: Intelligence emerges from temporal embodiment - learning from the past
to better navigate the future. Experience is not just stored, it's metabolized.

Architecture:
    - ExperienceBuffer: Ring buffer of recent experiences for replay
    - PredictionTracker: Track predictions and compare to outcomes
    - ProcedureUpdater: Update procedure success rates and parameters
    - PatternExtractor: Find patterns across episodes
    - TheoryRefiner: Update theories based on evidence
    - ExperienceReplay: Unified interface for learning from experience
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Experience Buffer
# =============================================================================

class ExperienceType(str, Enum):
    """Types of experiences that can be replayed."""
    ACTION = "action"              # Took an action
    OBSERVATION = "observation"    # Observed something
    DECISION = "decision"          # Made a decision
    OUTCOME = "outcome"            # Experienced an outcome
    REFLECTION = "reflection"      # Reflected on something
    INTERACTION = "interaction"    # Interacted with user/agent
    ERROR = "error"                # Made an error
    SUCCESS = "success"            # Achieved something


class Experience(BaseModel):
    """A single experience to be replayed."""
    id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: ExperienceType
    context: dict[str, Any] = Field(default_factory=dict)
    content: str
    outcome: str | None = None
    reward: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    episode_id: str | None = None
    step_number: int | None = None


class ExperienceBuffer:
    """
    Ring buffer of recent experiences for replay.

    Stores experiences in a file-backed ring buffer that can be sampled
    for learning. Supports prioritized replay based on reward magnitude.
    """

    def __init__(self, buffer_dir: Path, max_size: int = 10000):
        self.buffer_dir = buffer_dir
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._buffer_file = buffer_dir / "experience_buffer.jsonl"
        self._index_file = buffer_dir / "buffer_index.json"
        self._load_index()

    def _load_index(self):
        """Load buffer index."""
        if self._index_file.exists():
            with open(self._index_file, 'r') as f:
                data = json.load(f)
                self._head = data.get("head", 0)
                self._tail = data.get("tail", 0)
                self._count = data.get("count", 0)
        else:
            self._head = 0
            self._tail = 0
            self._count = 0

    def _save_index(self):
        """Save buffer index."""
        with open(self._index_file, 'w') as f:
            json.dump({
                "head": self._head,
                "tail": self._tail,
                "count": self._count,
            }, f)

    def add(self, experience: Experience):
        """Add experience to buffer."""
        with open(self._buffer_file, 'a') as f:
            f.write(experience.model_dump_json() + '\n')

        self._count += 1
        if self._count > self.max_size:
            self._head += 1
            self._count = self.max_size

        self._save_index()

    def sample(self, n: int = 32, prioritized: bool = False) -> list[Experience]:
        """
        Sample n experiences from buffer.

        If prioritized, samples experiences with higher |reward| more often.
        """
        if not self._buffer_file.exists() or self._count == 0:
            return []

        import random

        experiences = self._read_all()
        if not experiences:
            return []

        if len(experiences) <= n:
            return experiences

        if prioritized:
            # Weight by absolute reward
            weights = [abs(e.reward) + 0.1 for e in experiences]
            total = sum(weights)
            weights = [w / total for w in weights]
            sampled_indices = random.choices(range(len(experiences)), weights=weights, k=n)
            return [experiences[i] for i in sampled_indices]
        else:
            return random.sample(experiences, n)

    def get_recent(self, n: int = 100) -> list[Experience]:
        """Get n most recent experiences."""
        experiences = self._read_all()
        return experiences[-n:] if experiences else []

    def get_by_type(self, exp_type: ExperienceType, limit: int = 100) -> list[Experience]:
        """Get experiences of a specific type."""
        experiences = self._read_all()
        filtered = [e for e in experiences if e.type == exp_type]
        return filtered[-limit:] if filtered else []

    def get_by_episode(self, episode_id: str) -> list[Experience]:
        """Get all experiences from a specific episode."""
        experiences = self._read_all()
        return [e for e in experiences if e.episode_id == episode_id]

    def _read_all(self) -> list[Experience]:
        """Read all experiences from buffer (respecting head/tail)."""
        if not self._buffer_file.exists():
            return []

        experiences = []
        with open(self._buffer_file, 'r') as f:
            for i, line in enumerate(f):
                if i < self._head:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    experiences.append(Experience.model_validate_json(line))
                except Exception:
                    pass

        return experiences

    def clear(self):
        """Clear the buffer."""
        if self._buffer_file.exists():
            self._buffer_file.unlink()
        self._head = 0
        self._tail = 0
        self._count = 0
        self._save_index()


# =============================================================================
# Prediction Tracker
# =============================================================================

class Prediction(BaseModel):
    """A prediction made by the agent."""
    id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    action: str
    predicted_outcome: str
    confidence: float = 0.5
    context: dict[str, Any] = Field(default_factory=dict)
    actual_outcome: str | None = None
    outcome_timestamp: datetime | None = None
    error_magnitude: float | None = None  # How wrong was the prediction


class PredictionTracker:
    """
    Track predictions and compare to actual outcomes.

    This enables the agent to calibrate its predictions and learn
    which types of predictions are reliable vs unreliable.
    """

    def __init__(self, tracker_dir: Path):
        self.tracker_dir = tracker_dir
        self.tracker_dir.mkdir(parents=True, exist_ok=True)
        self._predictions_file = tracker_dir / "predictions.jsonl"
        self._stats_file = tracker_dir / "prediction_stats.json"

    def record_prediction(self, prediction: Prediction):
        """Record a new prediction."""
        with open(self._predictions_file, 'a') as f:
            f.write(prediction.model_dump_json() + '\n')

    def record_outcome(self, prediction_id: str, actual_outcome: str, error_magnitude: float = 0.0):
        """Record the actual outcome for a prediction."""
        predictions = self._load_predictions()

        for pred in predictions:
            if pred.id == prediction_id:
                pred.actual_outcome = actual_outcome
                pred.outcome_timestamp = datetime.now(UTC)
                pred.error_magnitude = error_magnitude
                break

        self._save_predictions(predictions)
        self._update_stats(predictions)

    def get_unresolved(self) -> list[Prediction]:
        """Get predictions without outcomes yet."""
        predictions = self._load_predictions()
        return [p for p in predictions if p.actual_outcome is None]

    def get_accuracy_by_action(self) -> dict[str, float]:
        """Get prediction accuracy broken down by action type."""
        predictions = self._load_predictions()

        action_correct: dict[str, int] = defaultdict(int)
        action_total: dict[str, int] = defaultdict(int)

        for pred in predictions:
            if pred.actual_outcome is not None:
                action_total[pred.action] += 1
                # Consider prediction correct if error is small
                if pred.error_magnitude is not None and pred.error_magnitude < 0.3:
                    action_correct[pred.action] += 1

        return {
            action: correct / total if total > 0 else 0.0
            for action, total in action_total.items()
            for correct in [action_correct[action]]
        }

    def get_calibration(self) -> dict[str, Any]:
        """
        Get calibration statistics.

        Returns how well the agent's confidence matches actual accuracy.
        """
        predictions = self._load_predictions()

        # Bucket by confidence
        buckets: dict[str, list[Prediction]] = defaultdict(list)
        for pred in predictions:
            if pred.actual_outcome is not None:
                bucket = f"{int(pred.confidence * 10) * 10}-{int(pred.confidence * 10) * 10 + 10}"
                buckets[bucket].append(pred)

        calibration = {}
        for bucket, preds in buckets.items():
            correct = sum(1 for p in preds if p.error_magnitude is not None and p.error_magnitude < 0.3)
            calibration[bucket] = {
                "count": len(preds),
                "actual_accuracy": correct / len(preds) if preds else 0,
                "avg_confidence": sum(p.confidence for p in preds) / len(preds) if preds else 0,
            }

        return calibration

    def _load_predictions(self) -> list[Prediction]:
        """Load all predictions."""
        if not self._predictions_file.exists():
            return []

        predictions = []
        with open(self._predictions_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    predictions.append(Prediction.model_validate_json(line))
                except Exception:
                    pass

        return predictions

    def _save_predictions(self, predictions: list[Prediction]):
        """Save all predictions."""
        with open(self._predictions_file, 'w') as f:
            for pred in predictions:
                f.write(pred.model_dump_json() + '\n')

    def _update_stats(self, predictions: list[Prediction]):
        """Update aggregated statistics."""
        resolved = [p for p in predictions if p.actual_outcome is not None]
        if not resolved:
            return

        correct = sum(1 for p in resolved if p.error_magnitude is not None and p.error_magnitude < 0.3)

        stats = {
            "total_predictions": len(predictions),
            "resolved_predictions": len(resolved),
            "overall_accuracy": correct / len(resolved) if resolved else 0,
            "avg_confidence": sum(p.confidence for p in resolved) / len(resolved) if resolved else 0,
            "avg_error": sum(p.error_magnitude or 0 for p in resolved) / len(resolved) if resolved else 0,
            "updated_at": datetime.now(UTC).isoformat(),
        }

        with open(self._stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


# =============================================================================
# Procedure Updater
# =============================================================================

class ProcedureStats(BaseModel):
    """Statistics about a procedure's performance."""
    procedure_name: str
    total_uses: int = 0
    successful_uses: int = 0
    failed_uses: int = 0
    avg_duration_ms: float = 0.0
    avg_reward: float = 0.0
    last_used: datetime | None = None
    contexts_used: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


class ProcedureUpdater:
    """
    Update procedure success rates and parameters based on outcomes.

    Tracks which procedures work well in which contexts,
    and suggests improvements or deprecation.
    """

    def __init__(self, updater_dir: Path):
        self.updater_dir = updater_dir
        self.updater_dir.mkdir(parents=True, exist_ok=True)
        self._stats_file = updater_dir / "procedure_stats.json"

    def record_use(
        self,
        procedure_name: str,
        success: bool,
        duration_ms: float,
        reward: float,
        context: str,
        failure_reason: str | None = None,
    ):
        """Record a procedure use."""
        stats = self._load_stats()

        if procedure_name not in stats:
            stats[procedure_name] = ProcedureStats(procedure_name=procedure_name)

        proc_stats = stats[procedure_name]
        proc_stats.total_uses += 1
        if success:
            proc_stats.successful_uses += 1
        else:
            proc_stats.failed_uses += 1
            if failure_reason:
                proc_stats.failure_reasons.append(failure_reason)
                # Keep only last 10 failure reasons
                proc_stats.failure_reasons = proc_stats.failure_reasons[-10:]

        # Update running averages
        n = proc_stats.total_uses
        proc_stats.avg_duration_ms = (proc_stats.avg_duration_ms * (n - 1) + duration_ms) / n
        proc_stats.avg_reward = (proc_stats.avg_reward * (n - 1) + reward) / n
        proc_stats.last_used = datetime.now(UTC)

        # Track context (keep unique)
        if context not in proc_stats.contexts_used:
            proc_stats.contexts_used.append(context)
            proc_stats.contexts_used = proc_stats.contexts_used[-20:]

        self._save_stats(stats)

    def get_success_rate(self, procedure_name: str) -> float:
        """Get success rate for a procedure."""
        stats = self._load_stats()
        if procedure_name not in stats:
            return 0.0

        proc_stats = stats[procedure_name]
        if proc_stats.total_uses == 0:
            return 0.0

        return proc_stats.successful_uses / proc_stats.total_uses

    def get_recommended_procedures(self, context: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Get recommended procedures for a context.

        Returns procedures ranked by success rate weighted by recency.
        """
        stats = self._load_stats()

        scored = []
        now = datetime.now(UTC)

        for name, proc_stats in stats.items():
            if proc_stats.total_uses == 0:
                continue

            # Base score is success rate
            success_rate = proc_stats.successful_uses / proc_stats.total_uses

            # Weight by recency (decay over 30 days)
            if proc_stats.last_used:
                days_ago = (now - proc_stats.last_used).total_seconds() / 86400
                recency_weight = max(0.1, 1.0 - days_ago / 30)
            else:
                recency_weight = 0.1

            # Bonus if context matches
            context_match = 1.0 if context in proc_stats.contexts_used else 0.5

            final_score = success_rate * recency_weight * context_match
            scored.append((name, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_procedures_to_deprecate(self, min_uses: int = 5, max_success_rate: float = 0.2) -> list[str]:
        """Get procedures that should be deprecated due to poor performance."""
        stats = self._load_stats()

        to_deprecate = []
        for name, proc_stats in stats.items():
            if proc_stats.total_uses >= min_uses:
                success_rate = proc_stats.successful_uses / proc_stats.total_uses
                if success_rate <= max_success_rate:
                    to_deprecate.append(name)

        return to_deprecate

    def get_all_stats(self) -> dict[str, ProcedureStats]:
        """Get all procedure statistics."""
        return self._load_stats()

    def _load_stats(self) -> dict[str, ProcedureStats]:
        """Load procedure statistics."""
        if not self._stats_file.exists():
            return {}

        try:
            with open(self._stats_file, 'r') as f:
                data = json.load(f)
            return {name: ProcedureStats.model_validate(stats) for name, stats in data.items()}
        except Exception:
            return {}

    def _save_stats(self, stats: dict[str, ProcedureStats]):
        """Save procedure statistics."""
        with open(self._stats_file, 'w') as f:
            json.dump({name: s.model_dump(mode='json') for name, s in stats.items()}, f, indent=2, default=str)


# =============================================================================
# Pattern Extractor
# =============================================================================

class Pattern(BaseModel):
    """A pattern extracted from experiences."""
    id: str
    name: str
    description: str
    pattern_type: str  # e.g., "success_pattern", "failure_pattern", "sequence"
    conditions: list[str] = Field(default_factory=list)
    consequences: list[str] = Field(default_factory=list)
    support: int = 0  # Number of experiences supporting this pattern
    confidence: float = 0.0
    examples: list[str] = Field(default_factory=list)
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PatternExtractor:
    """
    Find patterns across episodes and experiences.

    Uses simple heuristics to identify:
    - Common sequences of actions
    - Conditions that lead to success/failure
    - Recurring situations
    """

    def __init__(self, extractor_dir: Path):
        self.extractor_dir = extractor_dir
        self.extractor_dir.mkdir(parents=True, exist_ok=True)
        self._patterns_file = extractor_dir / "patterns.json"

    def extract_from_experiences(self, experiences: list[Experience]) -> list[Pattern]:
        """Extract patterns from a list of experiences."""
        patterns = []

        # Extract success patterns
        success_exps = [e for e in experiences if e.type == ExperienceType.SUCCESS]
        if success_exps:
            success_pattern = self._extract_success_pattern(success_exps, experiences)
            if success_pattern:
                patterns.append(success_pattern)

        # Extract failure patterns
        error_exps = [e for e in experiences if e.type == ExperienceType.ERROR]
        if error_exps:
            failure_pattern = self._extract_failure_pattern(error_exps, experiences)
            if failure_pattern:
                patterns.append(failure_pattern)

        # Extract action sequences
        action_exps = [e for e in experiences if e.type == ExperienceType.ACTION]
        if len(action_exps) >= 3:
            sequence_patterns = self._extract_sequences(action_exps)
            patterns.extend(sequence_patterns)

        # Save new patterns
        self._merge_and_save(patterns)

        return patterns

    def _extract_success_pattern(
        self,
        success_exps: list[Experience],
        all_exps: list[Experience]
    ) -> Pattern | None:
        """Extract pattern from successful experiences."""
        if len(success_exps) < 2:
            return None

        # Find common contexts
        common_contexts = self._find_common_keys(
            [e.context for e in success_exps]
        )

        # Find actions that preceded success
        preceding_actions = []
        for succ in success_exps:
            if succ.step_number and succ.step_number > 0:
                prev = [e for e in all_exps
                       if e.step_number == succ.step_number - 1
                       and e.type == ExperienceType.ACTION]
                if prev:
                    preceding_actions.append(prev[0].content[:100])

        import uuid
        return Pattern(
            id=str(uuid.uuid4()),
            name="success_conditions",
            description=f"Pattern from {len(success_exps)} successes",
            pattern_type="success_pattern",
            conditions=list(common_contexts)[:5],
            consequences=preceding_actions[:3],
            support=len(success_exps),
            confidence=min(1.0, len(success_exps) / 10),
            examples=[e.content[:100] for e in success_exps[:3]],
        )

    def _extract_failure_pattern(
        self,
        error_exps: list[Experience],
        all_exps: list[Experience]
    ) -> Pattern | None:
        """Extract pattern from failure experiences."""
        if len(error_exps) < 2:
            return None

        # Find common error contexts
        common_contexts = self._find_common_keys(
            [e.context for e in error_exps]
        )

        # Find what actions led to errors
        preceding_actions = []
        for err in error_exps:
            if err.step_number and err.step_number > 0:
                prev = [e for e in all_exps
                       if e.step_number == err.step_number - 1
                       and e.type == ExperienceType.ACTION]
                if prev:
                    preceding_actions.append(prev[0].content[:100])

        import uuid
        return Pattern(
            id=str(uuid.uuid4()),
            name="failure_conditions",
            description=f"Pattern from {len(error_exps)} failures",
            pattern_type="failure_pattern",
            conditions=list(common_contexts)[:5],
            consequences=preceding_actions[:3],
            support=len(error_exps),
            confidence=min(1.0, len(error_exps) / 10),
            examples=[e.content[:100] for e in error_exps[:3]],
        )

    def _extract_sequences(self, action_exps: list[Experience]) -> list[Pattern]:
        """Extract common action sequences."""
        patterns = []

        # Sort by timestamp
        sorted_exps = sorted(action_exps, key=lambda e: e.timestamp)

        # Find repeating sequences of length 2 and 3
        for seq_len in [2, 3]:
            sequences: dict[tuple[str, ...], int] = defaultdict(int)

            for i in range(len(sorted_exps) - seq_len + 1):
                seq = tuple(e.content[:50] for e in sorted_exps[i:i+seq_len])
                sequences[seq] += 1

            # Keep sequences that appear at least twice
            for seq, count in sequences.items():
                if count >= 2:
                    import uuid
                    patterns.append(Pattern(
                        id=str(uuid.uuid4()),
                        name=f"sequence_{seq_len}",
                        description=f"Sequence of {seq_len} actions appearing {count} times",
                        pattern_type="sequence",
                        conditions=list(seq),
                        support=count,
                        confidence=min(1.0, count / 5),
                    ))

        return patterns

    def _find_common_keys(self, contexts: list[dict[str, Any]]) -> set[str]:
        """Find keys that appear in most contexts."""
        if not contexts:
            return set()

        key_counts: dict[str, int] = defaultdict(int)
        for ctx in contexts:
            for key in ctx.keys():
                key_counts[key] += 1

        threshold = len(contexts) * 0.5
        return {k for k, v in key_counts.items() if v >= threshold}

    def _merge_and_save(self, new_patterns: list[Pattern]):
        """Merge new patterns with existing and save."""
        existing = self.get_all_patterns()

        # Simple merge: add new patterns, update support for similar ones
        pattern_map = {p.name: p for p in existing}

        for new_p in new_patterns:
            if new_p.name in pattern_map:
                # Update support
                pattern_map[new_p.name].support += new_p.support
                pattern_map[new_p.name].confidence = min(1.0, pattern_map[new_p.name].support / 20)
            else:
                pattern_map[new_p.name] = new_p

        # Save
        with open(self._patterns_file, 'w') as f:
            json.dump(
                [p.model_dump(mode='json') for p in pattern_map.values()],
                f, indent=2, default=str
            )

    def get_all_patterns(self) -> list[Pattern]:
        """Get all discovered patterns."""
        if not self._patterns_file.exists():
            return []

        try:
            with open(self._patterns_file, 'r') as f:
                data = json.load(f)
            return [Pattern.model_validate(p) for p in data]
        except Exception:
            return []

    def get_patterns_for_context(self, context: dict[str, Any]) -> list[Pattern]:
        """Get patterns relevant to a context."""
        patterns = self.get_all_patterns()
        context_keys = set(context.keys())

        relevant = []
        for pattern in patterns:
            pattern_keys = set(pattern.conditions)
            if pattern_keys & context_keys:
                relevant.append(pattern)

        return relevant


# =============================================================================
# Theory Refiner
# =============================================================================

class Evidence(BaseModel):
    """Evidence for or against a theory."""
    id: str
    theory_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    supports: bool  # True if evidence supports theory, False if contradicts
    description: str
    strength: float = 0.5  # How strong is this evidence
    source: str = "experience"  # Where did this evidence come from


class TheoryRefiner:
    """
    Update theories based on evidence.

    Tracks evidence for each theory and updates confidence accordingly.
    """

    def __init__(self, refiner_dir: Path):
        self.refiner_dir = refiner_dir
        self.refiner_dir.mkdir(parents=True, exist_ok=True)
        self._evidence_file = refiner_dir / "evidence.jsonl"
        self._theory_confidence_file = refiner_dir / "theory_confidence.json"

    def add_evidence(self, evidence: Evidence):
        """Add new evidence."""
        with open(self._evidence_file, 'a') as f:
            f.write(evidence.model_dump_json() + '\n')

        self._update_confidence(evidence.theory_id)

    def get_evidence_for_theory(self, theory_id: str) -> list[Evidence]:
        """Get all evidence for a theory."""
        all_evidence = self._load_all_evidence()
        return [e for e in all_evidence if e.theory_id == theory_id]

    def get_theory_confidence(self, theory_id: str) -> float:
        """Get current confidence in a theory."""
        confidences = self._load_confidences()
        return confidences.get(theory_id, 0.5)

    def get_theories_to_revise(self, confidence_threshold: float = 0.3) -> list[str]:
        """Get theories with low confidence that need revision."""
        confidences = self._load_confidences()
        return [tid for tid, conf in confidences.items() if conf < confidence_threshold]

    def get_contradicted_theories(self) -> list[str]:
        """Get theories with more contradicting than supporting evidence."""
        all_evidence = self._load_all_evidence()

        support_count: dict[str, int] = defaultdict(int)
        contradict_count: dict[str, int] = defaultdict(int)

        for ev in all_evidence:
            if ev.supports:
                support_count[ev.theory_id] += 1
            else:
                contradict_count[ev.theory_id] += 1

        contradicted = []
        for theory_id in set(support_count.keys()) | set(contradict_count.keys()):
            if contradict_count[theory_id] > support_count[theory_id]:
                contradicted.append(theory_id)

        return contradicted

    def _update_confidence(self, theory_id: str):
        """Update confidence for a theory based on all evidence."""
        evidence = self.get_evidence_for_theory(theory_id)
        if not evidence:
            return

        # Weighted evidence calculation
        support_weight = sum(e.strength for e in evidence if e.supports)
        contradict_weight = sum(e.strength for e in evidence if not e.supports)

        total_weight = support_weight + contradict_weight
        if total_weight == 0:
            confidence = 0.5
        else:
            confidence = support_weight / total_weight

        # Save
        confidences = self._load_confidences()
        confidences[theory_id] = confidence
        self._save_confidences(confidences)

    def _load_all_evidence(self) -> list[Evidence]:
        """Load all evidence."""
        if not self._evidence_file.exists():
            return []

        evidence = []
        with open(self._evidence_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evidence.append(Evidence.model_validate_json(line))
                except Exception:
                    pass

        return evidence

    def _load_confidences(self) -> dict[str, float]:
        """Load theory confidences."""
        if not self._theory_confidence_file.exists():
            return {}

        try:
            with open(self._theory_confidence_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_confidences(self, confidences: dict[str, float]):
        """Save theory confidences."""
        with open(self._theory_confidence_file, 'w') as f:
            json.dump(confidences, f, indent=2)


# =============================================================================
# Experience Replay (Unified Interface)
# =============================================================================

class ExperienceReplay:
    """
    Unified interface for learning from experience.

    Combines experience buffer, prediction tracking, procedure updating,
    pattern extraction, and theory refinement into a coherent learning system.
    """

    def __init__(self, learning_dir: Path):
        self.learning_dir = learning_dir
        self.learning_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.buffer = ExperienceBuffer(learning_dir / "buffer")
        self.predictions = PredictionTracker(learning_dir / "predictions")
        self.procedures = ProcedureUpdater(learning_dir / "procedures")
        self.patterns = PatternExtractor(learning_dir / "patterns")
        self.theories = TheoryRefiner(learning_dir / "theories")

    def record_experience(self, experience: Experience):
        """Record a new experience."""
        self.buffer.add(experience)

    def record_prediction(self, prediction: Prediction):
        """Record a prediction."""
        self.predictions.record_prediction(prediction)

    def record_outcome(self, prediction_id: str, actual_outcome: str, error: float = 0.0):
        """Record outcome for a prediction."""
        self.predictions.record_outcome(prediction_id, actual_outcome, error)

    def record_procedure_use(
        self,
        procedure_name: str,
        success: bool,
        duration_ms: float,
        reward: float,
        context: str,
        failure_reason: str | None = None,
    ):
        """Record a procedure use."""
        self.procedures.record_use(
            procedure_name, success, duration_ms, reward, context, failure_reason
        )

    def add_theory_evidence(self, evidence: Evidence):
        """Add evidence for a theory."""
        self.theories.add_evidence(evidence)

    def replay_and_learn(self, batch_size: int = 32) -> dict[str, Any]:
        """
        Replay a batch of experiences and extract learnings.

        Returns summary of what was learned.
        """
        # Sample experiences
        experiences = self.buffer.sample(batch_size, prioritized=True)
        if not experiences:
            return {"status": "no_experiences", "learned": False}

        # Extract patterns
        patterns = self.patterns.extract_from_experiences(experiences)

        # Get prediction accuracy
        accuracy = self.predictions.get_accuracy_by_action()

        # Get procedure recommendations
        procedures_to_deprecate = self.procedures.get_procedures_to_deprecate()

        # Get theories needing revision
        contradicted_theories = self.theories.get_contradicted_theories()

        return {
            "status": "learned",
            "learned": True,
            "experiences_replayed": len(experiences),
            "patterns_found": len(patterns),
            "new_patterns": [p.name for p in patterns],
            "prediction_accuracy": accuracy,
            "procedures_to_deprecate": procedures_to_deprecate,
            "contradicted_theories": contradicted_theories,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_learning_summary(self) -> dict[str, Any]:
        """Get overall learning statistics."""
        return {
            "buffer_size": self.buffer._count,
            "patterns_discovered": len(self.patterns.get_all_patterns()),
            "prediction_calibration": self.predictions.get_calibration(),
            "procedure_stats": {
                name: {
                    "success_rate": stats.successful_uses / stats.total_uses if stats.total_uses > 0 else 0,
                    "total_uses": stats.total_uses,
                }
                for name, stats in self.procedures.get_all_stats().items()
            },
            "theories_needing_revision": self.theories.get_theories_to_revise(),
        }

    def suggest_improvements(self, context: str) -> dict[str, Any]:
        """
        Suggest improvements based on learned patterns.

        Given current context, what have we learned that's relevant?
        """
        suggestions = {
            "recommended_procedures": [],
            "patterns_to_watch": [],
            "avoid_procedures": [],
            "relevant_theories": [],
        }

        # Get recommended procedures
        recommendations = self.procedures.get_recommended_procedures(context)
        suggestions["recommended_procedures"] = [
            {"name": name, "score": score}
            for name, score in recommendations
        ]

        # Get procedures to avoid
        suggestions["avoid_procedures"] = self.procedures.get_procedures_to_deprecate()

        # Get relevant patterns
        patterns = self.patterns.get_patterns_for_context({"context": context})
        suggestions["patterns_to_watch"] = [
            {"name": p.name, "type": p.pattern_type, "confidence": p.confidence}
            for p in patterns
        ]

        return suggestions
