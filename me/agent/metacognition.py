"""
Meta-Cognitive Monitoring System.

Enables the agent to monitor its own cognitive processes, detect when it's stuck,
track strategy effectiveness, and adapt its approach dynamically.

Philosophy: Self-awareness IS metacognition. The agent that knows its own
cognitive patterns can improve them. This is the recursive loop of intelligence.

Architecture:
    - PerformanceMonitor: Track cognitive performance metrics
    - StuckDetector: Detect circular reasoning and loops
    - StrategyTracker: Track which strategies work
    - AdaptationEngine: Suggest alternative approaches
    - NoveltyDetector: Track exploration vs exploitation
    - MetaCognitiveMonitor: Unified interface
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Performance Monitor
# =============================================================================

class CognitiveMetric(str, Enum):
    """Types of cognitive metrics to track."""
    TASK_COMPLETION_RATE = "task_completion_rate"
    AVG_TASK_DURATION = "avg_task_duration"
    ERROR_RATE = "error_rate"
    RETRY_RATE = "retry_rate"
    TOOL_SUCCESS_RATE = "tool_success_rate"
    PREDICTION_ACCURACY = "prediction_accuracy"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    NOVELTY_RATE = "novelty_rate"


class PerformanceSnapshot(BaseModel):
    """Snapshot of performance metrics at a point in time."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, float] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class PerformanceMonitor:
    """
    Track cognitive performance metrics over time.

    Maintains rolling windows of metrics to detect trends and anomalies.
    """

    def __init__(self, monitor_dir: Path, window_size: int = 100):
        self.monitor_dir = monitor_dir
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = monitor_dir / "performance_metrics.jsonl"
        self._summary_file = monitor_dir / "performance_summary.json"
        self._window_size = window_size

        # In-memory rolling windows
        self._windows: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._load()

    def _load(self):
        """Load recent metrics into windows."""
        if not self._metrics_file.exists():
            return

        try:
            with open(self._metrics_file, 'r') as f:
                lines = f.readlines()[-self._window_size:]

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    snapshot = PerformanceSnapshot.model_validate_json(line)
                    for name, value in snapshot.metrics.items():
                        self._windows[name].append(value)
                except Exception:
                    pass
        except Exception:
            pass

    def record(self, metric: str | CognitiveMetric, value: float, context: dict[str, Any] | None = None):
        """Record a metric value."""
        metric_name = metric.value if isinstance(metric, CognitiveMetric) else metric
        self._windows[metric_name].append(value)

        # Persist
        snapshot = PerformanceSnapshot(
            metrics={metric_name: value},
            context=context or {},
        )

        with open(self._metrics_file, 'a') as f:
            f.write(snapshot.model_dump_json() + '\n')

        self._update_summary()

    def get_current(self, metric: str | CognitiveMetric) -> float | None:
        """Get current (most recent) value of a metric."""
        metric_name = metric.value if isinstance(metric, CognitiveMetric) else metric
        if metric_name in self._windows and self._windows[metric_name]:
            return self._windows[metric_name][-1]
        return None

    def get_average(self, metric: str | CognitiveMetric, n: int | None = None) -> float:
        """Get average of a metric over last n values (or entire window)."""
        metric_name = metric.value if isinstance(metric, CognitiveMetric) else metric
        if metric_name not in self._windows or not self._windows[metric_name]:
            return 0.0

        values = list(self._windows[metric_name])
        if n:
            values = values[-n:]

        return sum(values) / len(values) if values else 0.0

    def get_trend(self, metric: str | CognitiveMetric) -> str:
        """Get trend direction for a metric (improving, degrading, stable)."""
        metric_name = metric.value if isinstance(metric, CognitiveMetric) else metric
        if metric_name not in self._windows or len(self._windows[metric_name]) < 10:
            return "unknown"

        values = list(self._windows[metric_name])
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        diff = second_avg - first_avg
        threshold = abs(first_avg) * 0.1 if first_avg != 0 else 0.1

        if diff > threshold:
            return "improving"
        elif diff < -threshold:
            return "degrading"
        else:
            return "stable"

    def detect_anomaly(self, metric: str | CognitiveMetric, value: float, std_threshold: float = 2.0) -> bool:
        """Detect if a value is anomalous (beyond threshold standard deviations)."""
        metric_name = metric.value if isinstance(metric, CognitiveMetric) else metric
        if metric_name not in self._windows or len(self._windows[metric_name]) < 10:
            return False

        values = list(self._windows[metric_name])
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5

        if std == 0:
            return False

        z_score = abs(value - mean) / std
        return z_score > std_threshold

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get summary of all metrics."""
        summary = {}
        for metric_name, window in self._windows.items():
            if not window:
                continue

            values = list(window)
            summary[metric_name] = {
                "current": values[-1],
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "trend": self.get_trend(metric_name),
            }

        return summary

    def _update_summary(self):
        """Update summary file."""
        summary = self.get_all_metrics()
        summary["updated_at"] = datetime.now(UTC).isoformat()

        with open(self._summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# Stuck Detector
# =============================================================================

class StuckType(str, Enum):
    """Types of stuck states."""
    LOOP = "loop"                  # Repeating same actions
    OSCILLATION = "oscillation"    # Going back and forth
    STAGNATION = "stagnation"      # No progress
    ERROR_LOOP = "error_loop"      # Repeatedly hitting same error
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Running out of tokens/time


class StuckDetection(BaseModel):
    """Record of a stuck detection."""
    id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stuck_type: StuckType
    description: str
    evidence: list[str] = Field(default_factory=list)
    severity: float = 0.5  # 0-1 scale
    suggested_actions: list[str] = Field(default_factory=list)
    resolved: bool = False
    resolved_at: datetime | None = None


class StuckDetector:
    """
    Detect when the agent is stuck in loops or making no progress.

    Uses pattern matching on recent actions and states to identify:
    - Repeated action sequences
    - Oscillating states
    - Lack of progress toward goals
    - Repeated errors
    """

    def __init__(self, detector_dir: Path, history_size: int = 50):
        self.detector_dir = detector_dir
        self.detector_dir.mkdir(parents=True, exist_ok=True)
        self._detections_file = detector_dir / "stuck_detections.jsonl"
        self._history_size = history_size

        # Track recent items
        self._recent_actions: deque[str] = deque(maxlen=history_size)
        self._recent_states: deque[str] = deque(maxlen=history_size)
        self._recent_errors: deque[str] = deque(maxlen=history_size)

    def record_action(self, action: str):
        """Record an action for loop detection."""
        self._recent_actions.append(action)

    def record_state(self, state: str):
        """Record a state summary for stagnation detection."""
        self._recent_states.append(state)

    def record_error(self, error: str):
        """Record an error for error loop detection."""
        self._recent_errors.append(error)

    def check_all(self) -> list[StuckDetection]:
        """Run all stuck detection checks."""
        detections = []

        # Check for action loops
        loop = self._detect_action_loop()
        if loop:
            detections.append(loop)

        # Check for oscillations
        oscillation = self._detect_oscillation()
        if oscillation:
            detections.append(oscillation)

        # Check for stagnation
        stagnation = self._detect_stagnation()
        if stagnation:
            detections.append(stagnation)

        # Check for error loops
        error_loop = self._detect_error_loop()
        if error_loop:
            detections.append(error_loop)

        # Persist detections
        for detection in detections:
            with open(self._detections_file, 'a') as f:
                f.write(detection.model_dump_json() + '\n')

        return detections

    def _detect_action_loop(self, min_repetitions: int = 3) -> StuckDetection | None:
        """Detect repeated action sequences."""
        if len(self._recent_actions) < min_repetitions * 2:
            return None

        actions = list(self._recent_actions)

        # Check for repeated single actions
        last = actions[-1]
        count = sum(1 for a in actions[-10:] if a == last)
        if count >= min_repetitions:
            import uuid
            return StuckDetection(
                id=str(uuid.uuid4()),
                stuck_type=StuckType.LOOP,
                description=f"Action '{last}' repeated {count} times in last 10 actions",
                evidence=actions[-10:],
                severity=min(1.0, count / 10),
                suggested_actions=[
                    "Try a different approach",
                    "Reconsider the goal",
                    "Ask for help",
                ],
            )

        # Check for repeated sequences
        for seq_len in range(2, 5):
            if len(actions) < seq_len * min_repetitions:
                continue

            last_seq = tuple(actions[-seq_len:])
            count = 0

            for i in range(len(actions) - seq_len, -1, -seq_len):
                seq = tuple(actions[i:i+seq_len])
                if seq == last_seq:
                    count += 1
                else:
                    break

            if count >= min_repetitions:
                import uuid
                return StuckDetection(
                    id=str(uuid.uuid4()),
                    stuck_type=StuckType.LOOP,
                    description=f"Sequence of {seq_len} actions repeated {count} times",
                    evidence=list(last_seq),
                    severity=min(1.0, count / 5),
                    suggested_actions=[
                        "Break the pattern",
                        "Try alternative actions",
                        "Revisit assumptions",
                    ],
                )

        return None

    def _detect_oscillation(self, window: int = 10) -> StuckDetection | None:
        """Detect back-and-forth oscillation between states."""
        if len(self._recent_states) < window:
            return None

        states = list(self._recent_states)[-window:]
        unique_states = set(states)

        # Oscillation: only 2 unique states alternating
        if len(unique_states) == 2:
            transitions = 0
            for i in range(1, len(states)):
                if states[i] != states[i-1]:
                    transitions += 1

            # High transitions relative to states = oscillation
            if transitions >= len(states) * 0.6:
                import uuid
                return StuckDetection(
                    id=str(uuid.uuid4()),
                    stuck_type=StuckType.OSCILLATION,
                    description=f"Oscillating between {len(unique_states)} states with {transitions} transitions",
                    evidence=list(unique_states),
                    severity=min(1.0, transitions / window),
                    suggested_actions=[
                        "Find a third option",
                        "Commit to one direction",
                        "Re-evaluate the decision criteria",
                    ],
                )

        return None

    def _detect_stagnation(self, window: int = 10) -> StuckDetection | None:
        """Detect lack of state change (stagnation)."""
        if len(self._recent_states) < window:
            return None

        states = list(self._recent_states)[-window:]
        unique_states = set(states)

        # Stagnation: same state repeated
        if len(unique_states) == 1:
            import uuid
            return StuckDetection(
                id=str(uuid.uuid4()),
                stuck_type=StuckType.STAGNATION,
                description=f"State unchanged for {window} steps",
                evidence=list(unique_states),
                severity=0.7,
                suggested_actions=[
                    "Take any action to change state",
                    "Gather more information",
                    "Try a random exploration",
                ],
            )

        return None

    def _detect_error_loop(self, min_repetitions: int = 2) -> StuckDetection | None:
        """Detect repeated errors."""
        if len(self._recent_errors) < min_repetitions:
            return None

        errors = list(self._recent_errors)

        # Count error frequencies
        error_counts: dict[str, int] = defaultdict(int)
        for error in errors:
            # Normalize error (remove line numbers, etc.)
            normalized = re.sub(r'\d+', 'N', error)[:100]
            error_counts[normalized] += 1

        # Check for repeated errors
        for error, count in error_counts.items():
            if count >= min_repetitions:
                import uuid
                return StuckDetection(
                    id=str(uuid.uuid4()),
                    stuck_type=StuckType.ERROR_LOOP,
                    description=f"Error pattern repeated {count} times: {error[:50]}",
                    evidence=[error],
                    severity=min(1.0, count / 5),
                    suggested_actions=[
                        "Address the root cause",
                        "Try a different approach",
                        "Check preconditions",
                    ],
                )

        return None

    def mark_resolved(self, detection_id: str):
        """Mark a stuck detection as resolved."""
        # Would update the detection in storage
        pass

    def get_unresolved(self) -> list[StuckDetection]:
        """Get unresolved stuck detections."""
        if not self._detections_file.exists():
            return []

        detections = []
        with open(self._detections_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    detection = StuckDetection.model_validate_json(line)
                    if not detection.resolved:
                        detections.append(detection)
                except Exception:
                    pass

        return detections


# =============================================================================
# Strategy Tracker
# =============================================================================

class Strategy(BaseModel):
    """A cognitive strategy used by the agent."""
    id: str
    name: str
    description: str
    uses: int = 0
    successes: int = 0
    failures: int = 0
    total_reward: float = 0.0
    contexts_used: list[str] = Field(default_factory=list)
    avg_duration_ms: float = 0.0
    last_used: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StrategyTracker:
    """
    Track which strategies work in which contexts.

    Strategies are higher-level approaches (e.g., "divide and conquer",
    "ask for clarification", "try simplest solution first").
    """

    def __init__(self, tracker_dir: Path):
        self.tracker_dir = tracker_dir
        self.tracker_dir.mkdir(parents=True, exist_ok=True)
        self._strategies_file = tracker_dir / "strategies.json"
        self._load()

    def _load(self):
        """Load strategies from disk."""
        self._strategies: dict[str, Strategy] = {}

        if self._strategies_file.exists():
            try:
                with open(self._strategies_file, 'r') as f:
                    data = json.load(f)
                for strat_data in data:
                    strat = Strategy.model_validate(strat_data)
                    self._strategies[strat.id] = strat
            except Exception:
                pass

    def _save(self):
        """Save strategies to disk."""
        with open(self._strategies_file, 'w') as f:
            json.dump(
                [s.model_dump(mode='json') for s in self._strategies.values()],
                f, indent=2, default=str
            )

    def register_strategy(self, name: str, description: str) -> Strategy:
        """Register a new strategy."""
        import uuid
        strategy = Strategy(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
        )
        self._strategies[strategy.id] = strategy
        self._save()
        return strategy

    def record_use(
        self,
        strategy_id: str,
        success: bool,
        reward: float,
        duration_ms: float,
        context: str,
    ):
        """Record strategy use."""
        if strategy_id not in self._strategies:
            return

        strategy = self._strategies[strategy_id]
        strategy.uses += 1
        if success:
            strategy.successes += 1
        else:
            strategy.failures += 1

        strategy.total_reward += reward

        # Update running averages
        n = strategy.uses
        strategy.avg_duration_ms = (strategy.avg_duration_ms * (n - 1) + duration_ms) / n

        strategy.last_used = datetime.now(UTC)

        if context not in strategy.contexts_used:
            strategy.contexts_used.append(context)
            strategy.contexts_used = strategy.contexts_used[-20:]

        self._save()

    def get_best_for_context(self, context: str, top_k: int = 3) -> list[Strategy]:
        """Get best strategies for a context."""
        scored = []

        for strategy in self._strategies.values():
            if strategy.uses == 0:
                continue

            # Base score is success rate
            success_rate = strategy.successes / strategy.uses

            # Context match bonus
            context_bonus = 0.2 if context in strategy.contexts_used else 0

            # Recency bonus
            recency_bonus = 0
            if strategy.last_used:
                days_ago = (datetime.now(UTC) - strategy.last_used).total_seconds() / 86400
                recency_bonus = max(0, 0.1 * (1 - days_ago / 30))

            score = success_rate + context_bonus + recency_bonus
            scored.append((strategy, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:top_k]]

    def get_struggling_strategies(self, min_uses: int = 5, max_success_rate: float = 0.3) -> list[Strategy]:
        """Get strategies that are struggling."""
        struggling = []

        for strategy in self._strategies.values():
            if strategy.uses >= min_uses:
                success_rate = strategy.successes / strategy.uses
                if success_rate <= max_success_rate:
                    struggling.append(strategy)

        return struggling

    def get_all_statistics(self) -> dict[str, Any]:
        """Get statistics for all strategies."""
        strategies = list(self._strategies.values())

        return {
            "total_strategies": len(strategies),
            "total_uses": sum(s.uses for s in strategies),
            "overall_success_rate": (
                sum(s.successes for s in strategies) / sum(s.uses for s in strategies)
                if sum(s.uses for s in strategies) > 0 else 0
            ),
            "strategies": [
                {
                    "name": s.name,
                    "uses": s.uses,
                    "success_rate": s.successes / s.uses if s.uses > 0 else 0,
                    "avg_reward": s.total_reward / s.uses if s.uses > 0 else 0,
                }
                for s in strategies
            ],
        }


# =============================================================================
# Adaptation Engine
# =============================================================================

class Adaptation(BaseModel):
    """A suggested adaptation to the agent's approach."""
    id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trigger: str  # What triggered this suggestion
    current_approach: str
    suggested_approach: str
    rationale: str
    priority: float = 0.5
    adopted: bool = False
    outcome: str | None = None


class AdaptationEngine:
    """
    Suggest alternative approaches when current ones aren't working.

    Uses performance data and stuck detections to recommend changes.
    """

    def __init__(
        self,
        engine_dir: Path,
        performance: PerformanceMonitor,
        stuck: StuckDetector,
        strategies: StrategyTracker,
    ):
        self.engine_dir = engine_dir
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        self._adaptations_file = engine_dir / "adaptations.jsonl"
        self._performance = performance
        self._stuck = stuck
        self._strategies = strategies

    def suggest_adaptations(self) -> list[Adaptation]:
        """Generate adaptation suggestions based on current state."""
        adaptations = []

        # Check stuck detections
        stuck_detections = self._stuck.get_unresolved()
        for detection in stuck_detections:
            adaptation = self._adapt_for_stuck(detection)
            if adaptation:
                adaptations.append(adaptation)

        # Check performance trends
        metrics = self._performance.get_all_metrics()
        for metric_name, data in metrics.items():
            if data.get("trend") == "degrading":
                adaptation = self._adapt_for_degrading(metric_name, data)
                if adaptation:
                    adaptations.append(adaptation)

        # Check struggling strategies
        struggling = self._strategies.get_struggling_strategies()
        for strategy in struggling:
            adaptation = self._adapt_for_struggling_strategy(strategy)
            if adaptation:
                adaptations.append(adaptation)

        # Persist
        for adaptation in adaptations:
            with open(self._adaptations_file, 'a') as f:
                f.write(adaptation.model_dump_json() + '\n')

        return adaptations

    def _adapt_for_stuck(self, detection: StuckDetection) -> Adaptation | None:
        """Generate adaptation for a stuck detection."""
        import uuid

        suggestions = {
            StuckType.LOOP: {
                "approach": "random_exploration",
                "rationale": "Break repetitive pattern by trying random alternatives",
            },
            StuckType.OSCILLATION: {
                "approach": "commitment_with_timeout",
                "rationale": "Commit to one option with a time limit for re-evaluation",
            },
            StuckType.STAGNATION: {
                "approach": "information_gathering",
                "rationale": "Gather more information before acting",
            },
            StuckType.ERROR_LOOP: {
                "approach": "root_cause_analysis",
                "rationale": "Step back and analyze the root cause of errors",
            },
        }

        suggestion = suggestions.get(detection.stuck_type)
        if not suggestion:
            return None

        return Adaptation(
            id=str(uuid.uuid4()),
            trigger=f"stuck:{detection.stuck_type.value}",
            current_approach="current_failing_approach",
            suggested_approach=suggestion["approach"],
            rationale=suggestion["rationale"],
            priority=detection.severity,
        )

    def _adapt_for_degrading(self, metric: str, data: dict[str, Any]) -> Adaptation | None:
        """Generate adaptation for degrading metric."""
        import uuid

        suggestions = {
            "task_completion_rate": {
                "approach": "simplify_tasks",
                "rationale": "Break tasks into smaller, achievable steps",
            },
            "error_rate": {
                "approach": "defensive_coding",
                "rationale": "Add more validation and error handling",
            },
            "prediction_accuracy": {
                "approach": "calibration_update",
                "rationale": "Update mental models based on recent evidence",
            },
        }

        suggestion = suggestions.get(metric)
        if not suggestion:
            return None

        return Adaptation(
            id=str(uuid.uuid4()),
            trigger=f"degrading:{metric}",
            current_approach=f"current_{metric}_approach",
            suggested_approach=suggestion["approach"],
            rationale=suggestion["rationale"],
            priority=0.6,
        )

    def _adapt_for_struggling_strategy(self, strategy: Strategy) -> Adaptation | None:
        """Generate adaptation for struggling strategy."""
        import uuid

        success_rate = strategy.successes / strategy.uses if strategy.uses > 0 else 0

        return Adaptation(
            id=str(uuid.uuid4()),
            trigger=f"struggling_strategy:{strategy.name}",
            current_approach=strategy.name,
            suggested_approach="alternative_strategy",
            rationale=f"Strategy '{strategy.name}' has only {success_rate:.0%} success rate",
            priority=0.5,
        )

    def record_outcome(self, adaptation_id: str, adopted: bool, outcome: str):
        """Record the outcome of an adaptation suggestion."""
        # Would update the adaptation in storage
        pass


# =============================================================================
# Novelty Detector
# =============================================================================

class NoveltyDetector:
    """
    Track exploration vs exploitation.

    Monitors how much the agent is trying new things vs repeating known patterns.
    """

    def __init__(self, detector_dir: Path, memory_size: int = 1000):
        self.detector_dir = detector_dir
        self.detector_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = detector_dir / "action_history.jsonl"
        self._memory_size = memory_size
        self._seen_actions: set[str] = set()
        self._load()

    def _load(self):
        """Load action history."""
        if not self._history_file.exists():
            return

        try:
            with open(self._history_file, 'r') as f:
                lines = f.readlines()[-self._memory_size:]

            for line in lines:
                line = line.strip()
                if line:
                    self._seen_actions.add(line)
        except Exception:
            pass

    def record_action(self, action: str) -> bool:
        """
        Record an action and return whether it's novel.

        Returns True if this is a new action, False if seen before.
        """
        # Normalize action
        normalized = self._normalize(action)

        is_novel = normalized not in self._seen_actions

        self._seen_actions.add(normalized)

        # Persist
        with open(self._history_file, 'a') as f:
            f.write(normalized + '\n')

        # Trim if too large
        if len(self._seen_actions) > self._memory_size:
            self._seen_actions = set(list(self._seen_actions)[-self._memory_size:])

        return is_novel

    def _normalize(self, action: str) -> str:
        """Normalize action for comparison."""
        # Remove specific values, keep structure
        normalized = re.sub(r'\d+', 'N', action)
        normalized = re.sub(r'"[^"]*"', '"S"', normalized)
        return normalized[:200]

    def get_novelty_rate(self, window: int = 100) -> float:
        """Get novelty rate over recent actions."""
        if not self._history_file.exists():
            return 1.0

        try:
            with open(self._history_file, 'r') as f:
                lines = f.readlines()[-window:]

            seen_in_window: set[str] = set()
            novel_count = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line not in seen_in_window:
                    novel_count += 1
                    seen_in_window.add(line)

            return novel_count / len(lines) if lines else 1.0

        except Exception:
            return 1.0

    def suggest_exploration(self) -> bool:
        """Suggest whether to explore vs exploit."""
        novelty_rate = self.get_novelty_rate()

        # If novelty is low, suggest exploration
        return novelty_rate < 0.3


# =============================================================================
# Meta-Cognitive Monitor (Unified Interface)
# =============================================================================

class MetaCognitiveMonitor:
    """
    Unified interface for meta-cognitive monitoring.

    Combines performance tracking, stuck detection, strategy tracking,
    adaptation suggestions, and novelty detection.
    """

    def __init__(self, metacog_dir: Path):
        self.metacog_dir = metacog_dir
        self.metacog_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.performance = PerformanceMonitor(metacog_dir / "performance")
        self.stuck = StuckDetector(metacog_dir / "stuck")
        self.strategies = StrategyTracker(metacog_dir / "strategies")
        self.novelty = NoveltyDetector(metacog_dir / "novelty")
        self.adaptation = AdaptationEngine(
            metacog_dir / "adaptation",
            self.performance,
            self.stuck,
            self.strategies,
        )

    def record_step(
        self,
        action: str,
        state: str,
        success: bool,
        duration_ms: float,
        reward: float,
    ):
        """Record a step for all monitoring components."""
        # Record action for stuck and novelty detection
        self.stuck.record_action(action)
        self.stuck.record_state(state)
        is_novel = self.novelty.record_action(action)

        # Record performance metrics
        self.performance.record(CognitiveMetric.TASK_COMPLETION_RATE, 1.0 if success else 0.0)
        self.performance.record(CognitiveMetric.AVG_TASK_DURATION, duration_ms)
        self.performance.record(CognitiveMetric.NOVELTY_RATE, 1.0 if is_novel else 0.0)

    def record_error(self, error: str):
        """Record an error."""
        self.stuck.record_error(error)
        self.performance.record(CognitiveMetric.ERROR_RATE, 1.0)

    def record_prediction(self, predicted: str, actual: str, accuracy: float):
        """Record a prediction and its accuracy."""
        self.performance.record(CognitiveMetric.PREDICTION_ACCURACY, accuracy)

    def check_cognitive_state(self) -> dict[str, Any]:
        """
        Check overall cognitive state.

        Returns summary of performance, stuck states, and suggestions.
        """
        # Check for stuck states
        stuck_detections = self.stuck.check_all()

        # Get adaptation suggestions
        adaptations = self.adaptation.suggest_adaptations()

        # Get performance summary
        metrics = self.performance.get_all_metrics()

        # Get novelty rate
        novelty_rate = self.novelty.get_novelty_rate()

        return {
            "status": "attention_needed" if stuck_detections else "ok",
            "stuck_detections": [d.model_dump() for d in stuck_detections],
            "adaptations_suggested": [a.model_dump() for a in adaptations],
            "performance_metrics": metrics,
            "novelty_rate": novelty_rate,
            "should_explore": self.novelty.suggest_exploration(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_recommended_strategy(self, context: str) -> Strategy | None:
        """Get recommended strategy for current context."""
        strategies = self.strategies.get_best_for_context(context)
        return strategies[0] if strategies else None

    def register_strategy(self, name: str, description: str) -> Strategy:
        """Register a new strategy."""
        return self.strategies.register_strategy(name, description)

    def record_strategy_use(
        self,
        strategy_id: str,
        success: bool,
        reward: float,
        duration_ms: float,
        context: str,
    ):
        """Record strategy use."""
        self.strategies.record_use(strategy_id, success, reward, duration_ms, context)

    def get_full_report(self) -> dict[str, Any]:
        """Get comprehensive metacognitive report."""
        return {
            "cognitive_state": self.check_cognitive_state(),
            "strategy_statistics": self.strategies.get_all_statistics(),
            "performance_trends": {
                metric: self.performance.get_trend(metric)
                for metric in CognitiveMetric
            },
            "generated_at": datetime.now(UTC).isoformat(),
        }
