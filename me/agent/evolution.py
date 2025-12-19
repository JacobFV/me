"""
Agent Evolution - Self-improvement and autonomous growth capabilities.

This module enables the agent to:
1. Analyze its own performance and identify weaknesses
2. Generate intrinsic goals (curiosity, self-improvement)
3. Explore its environment and capabilities
4. Learn from errors and build recovery procedures
5. Evolve its character based on experience
6. Run self-improvement experiments

Philosophy: The agent is not just a passive executor of tasks. It actively
seeks to understand itself, improve its capabilities, and grow over time.
The trajectory of self-improvement is itself a form of identity.
"""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta, UTC
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from me.agent.body import BodyDirectory, Character, Episode
    from me.agent.skills import SkillManager
    from me.agent.learning import ExperienceBuffer


# =============================================================================
# Performance Self-Analysis
# =============================================================================

class PerformanceArea(str, Enum):
    """Areas of agent performance to analyze."""
    TASK_COMPLETION = "task_completion"
    ERROR_HANDLING = "error_handling"
    EFFICIENCY = "efficiency"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    TOOL_USE = "tool_use"
    PLANNING = "planning"
    MEMORY_USE = "memory_use"


class PerformanceInsight(BaseModel):
    """An insight about agent performance."""
    area: PerformanceArea
    observation: str
    evidence: list[str] = Field(default_factory=list)
    severity: str = "info"  # info, warning, critical
    suggested_improvement: str | None = None
    confidence: float = 0.5
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PerformanceReport(BaseModel):
    """A self-analysis report of agent performance."""
    period_start: datetime
    period_end: datetime
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0

    # Computed metrics
    success_rate: float = 0.0
    average_task_duration: float = 0.0
    error_rate: float = 0.0

    # Insights
    insights: list[PerformanceInsight] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    # Recommended actions
    improvement_priorities: list[str] = Field(default_factory=list)


class SelfAnalyzer:
    """
    Analyzes agent's own performance and identifies improvement areas.

    This is introspection - the agent examining its own behavior to
    identify patterns, strengths, and weaknesses.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.episodes_dir = body_dir / "memory" / "episodes"
        self.reports_dir = body_dir / "evolution" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def analyze_recent_performance(
        self,
        days: int = 7,
    ) -> PerformanceReport:
        """Analyze performance over recent period."""
        from me.agent.body import markdown_to_model, Episode

        end = datetime.now(UTC)
        start = end - timedelta(days=days)

        report = PerformanceReport(
            period_start=start,
            period_end=end,
        )

        # Load episodes
        episodes = []
        for ep_path in self.episodes_dir.glob("*.md"):
            try:
                content = ep_path.read_text()
                episode, _ = markdown_to_model(content, Episode)
                if episode.started_at >= start:
                    episodes.append(episode)
            except Exception:
                continue

        if not episodes:
            report.insights.append(PerformanceInsight(
                area=PerformanceArea.LEARNING,
                observation="No episodes recorded in this period",
                severity="warning",
                suggested_improvement="Record more episodes to enable self-analysis",
            ))
            return report

        # Basic metrics
        report.total_tasks = len(episodes)
        report.successful_tasks = sum(1 for e in episodes if e.outcome == "completed")
        report.failed_tasks = sum(1 for e in episodes if e.outcome == "failed")
        report.success_rate = report.successful_tasks / report.total_tasks if report.total_tasks > 0 else 0.0

        # Analyze patterns
        self._analyze_error_patterns(episodes, report)
        self._analyze_tool_usage(episodes, report)
        self._analyze_learning_patterns(episodes, report)
        self._identify_strengths_weaknesses(report)
        self._prioritize_improvements(report)

        # Save report
        self._save_report(report)

        return report

    def _analyze_error_patterns(
        self,
        episodes: list["Episode"],
        report: PerformanceReport,
    ) -> None:
        """Identify patterns in errors and failures."""
        failed = [e for e in episodes if e.outcome == "failed"]

        if not failed:
            return

        # Count error types
        error_tools: dict[str, int] = defaultdict(int)
        for ep in failed:
            for tool in ep.tools_used:
                error_tools[tool] += 1

        if error_tools:
            most_failed = max(error_tools, key=error_tools.get)
            if error_tools[most_failed] >= 2:
                report.insights.append(PerformanceInsight(
                    area=PerformanceArea.ERROR_HANDLING,
                    observation=f"Tool '{most_failed}' involved in {error_tools[most_failed]} failures",
                    evidence=[f"Failed episode: {e.title}" for e in failed if most_failed in e.tools_used],
                    severity="warning",
                    suggested_improvement=f"Review usage of {most_failed} tool, consider alternative approaches",
                    confidence=0.7,
                ))

    def _analyze_tool_usage(
        self,
        episodes: list["Episode"],
        report: PerformanceReport,
    ) -> None:
        """Analyze tool usage patterns."""
        tool_counts: dict[str, int] = defaultdict(int)
        tool_success: dict[str, int] = defaultdict(int)

        for ep in episodes:
            for tool in ep.tools_used:
                tool_counts[tool] += 1
                if ep.outcome == "completed":
                    tool_success[tool] += 1

        # Identify underused tools
        if tool_counts:
            avg_use = sum(tool_counts.values()) / len(tool_counts)
            underused = [t for t, c in tool_counts.items() if c < avg_use * 0.3]
            if underused:
                report.insights.append(PerformanceInsight(
                    area=PerformanceArea.TOOL_USE,
                    observation=f"Tools {underused} are rarely used",
                    severity="info",
                    suggested_improvement="Explore if these tools could help with current tasks",
                ))

    def _analyze_learning_patterns(
        self,
        episodes: list["Episode"],
        report: PerformanceReport,
    ) -> None:
        """Analyze how well the agent is learning."""
        lessons = []
        for ep in episodes:
            lessons.extend(ep.lessons)

        if len(lessons) < len(episodes) * 0.5:
            report.insights.append(PerformanceInsight(
                area=PerformanceArea.LEARNING,
                observation="Few lessons recorded relative to episodes",
                severity="warning",
                suggested_improvement="Extract more lessons from experiences",
                confidence=0.6,
            ))

        # Check for repeated lessons (not learning)
        lesson_hashes = [hashlib.md5(l.lower().encode()).hexdigest()[:8] for l in lessons]
        repeated = len(lesson_hashes) - len(set(lesson_hashes))
        if repeated > 2:
            report.insights.append(PerformanceInsight(
                area=PerformanceArea.LEARNING,
                observation=f"{repeated} lessons are repeated - may not be applying learnings",
                severity="warning",
                suggested_improvement="Review and apply past lessons before starting similar tasks",
            ))

    def _identify_strengths_weaknesses(self, report: PerformanceReport) -> None:
        """Identify overall strengths and weaknesses."""
        if report.success_rate >= 0.8:
            report.strengths.append("High task completion rate")
        elif report.success_rate < 0.5:
            report.weaknesses.append("Low task completion rate")

        error_insights = [i for i in report.insights if i.area == PerformanceArea.ERROR_HANDLING]
        if not error_insights:
            report.strengths.append("Good error handling")
        elif any(i.severity == "critical" for i in error_insights):
            report.weaknesses.append("Significant error handling issues")

    def _prioritize_improvements(self, report: PerformanceReport) -> None:
        """Prioritize improvement areas."""
        # Sort insights by severity and confidence
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_insights = sorted(
            [i for i in report.insights if i.suggested_improvement],
            key=lambda i: (severity_order.get(i.severity, 2), -i.confidence)
        )

        report.improvement_priorities = [
            i.suggested_improvement for i in sorted_insights[:5]
            if i.suggested_improvement
        ]

    def _save_report(self, report: PerformanceReport) -> None:
        """Save report to file."""
        filename = f"report-{report.period_end.strftime('%Y%m%d')}.json"
        path = self.reports_dir / filename
        path.write_text(json.dumps(report.model_dump(mode='json'), indent=2, default=str))


# =============================================================================
# Intrinsic Goal Generation (Curiosity)
# =============================================================================

class GoalType(str, Enum):
    """Types of self-generated goals."""
    EXPLORATION = "exploration"       # Explore unknown areas
    SKILL_BUILDING = "skill_building" # Improve a capability
    KNOWLEDGE = "knowledge"           # Learn about something
    MAINTENANCE = "maintenance"       # Self-maintenance tasks
    EXPERIMENT = "experiment"         # Test a hypothesis
    RECOVERY = "recovery"             # Recover from issues


class IntrinsicGoal(BaseModel):
    """A goal generated by the agent itself."""
    id: str
    goal_type: GoalType
    description: str
    motivation: str  # Why this goal matters
    priority: float = 0.5

    # Progress tracking
    status: str = "pending"  # pending, in_progress, completed, abandoned
    progress: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Learning
    hypothesis: str | None = None  # For experiments
    expected_outcome: str | None = None
    actual_outcome: str | None = None
    lessons: list[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = "curiosity"  # curiosity, self_analysis, error_recovery


class CuriosityEngine:
    """
    Generates intrinsic goals based on curiosity and self-improvement drive.

    The agent doesn't just wait for tasks - it actively seeks to understand
    its environment, expand its capabilities, and fill gaps in knowledge.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.goals_dir = body_dir / "evolution" / "goals"
        self.goals_dir.mkdir(parents=True, exist_ok=True)

    def generate_exploration_goals(self) -> list[IntrinsicGoal]:
        """Generate goals to explore the environment."""
        goals = []

        # Check for unexplored directories
        cwd = Path.cwd()
        unexplored = []
        for item in cwd.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                marker = self.goals_dir / f"explored-{hashlib.md5(str(item).encode()).hexdigest()[:8]}"
                if not marker.exists():
                    unexplored.append(item)

        if unexplored:
            target = unexplored[0]
            goals.append(IntrinsicGoal(
                id=f"explore-{hashlib.md5(str(target).encode()).hexdigest()[:8]}",
                goal_type=GoalType.EXPLORATION,
                description=f"Explore directory: {target.name}",
                motivation="Understanding the environment enables better task execution",
                priority=0.3,
            ))

        return goals

    def generate_skill_goals(self, skill_manager: "SkillManager") -> list[IntrinsicGoal]:
        """Generate goals to improve skills."""
        from me.agent.skills import SkillState

        goals = []

        # Find skills that need practice
        developing = skill_manager.list_skills(SkillState.DEVELOPING)
        for meta in developing[:2]:
            if meta.use_count < 3:
                goals.append(IntrinsicGoal(
                    id=f"practice-{meta.name}",
                    goal_type=GoalType.SKILL_BUILDING,
                    description=f"Practice skill: {meta.name}",
                    motivation=f"Skill at {meta.proficiency:.0%} proficiency, needs more practice",
                    priority=0.5,
                ))

        # Find atrophying skills
        from me.agent.skill_integration import AtrophyDetector
        atrophy = AtrophyDetector(skill_manager)
        warnings = atrophy.check_atrophy()

        for warning in warnings[:2]:
            goals.append(IntrinsicGoal(
                id=f"revive-{warning.skill_name}",
                goal_type=GoalType.SKILL_BUILDING,
                description=f"Revive skill: {warning.skill_name}",
                motivation=f"Skill unused for {warning.days_since_use} days, at risk of atrophy",
                priority=0.6 if warning.warning_level == "critical" else 0.4,
            ))

        return goals

    def generate_knowledge_goals(self, body: "BodyDirectory") -> list[IntrinsicGoal]:
        """Generate goals to expand knowledge."""
        goals = []

        # Check for gaps in theories
        theories_dir = self.body_dir / "memory" / "theories"
        if theories_dir.exists():
            theory_files = list(theories_dir.glob("*.md"))
            if len(theory_files) < 3:
                goals.append(IntrinsicGoal(
                    id="develop-theories",
                    goal_type=GoalType.KNOWLEDGE,
                    description="Develop working theories about the environment",
                    motivation="Theories enable prediction and better decision-making",
                    priority=0.4,
                ))

        return goals

    def generate_experiment_goal(
        self,
        hypothesis: str,
        test_description: str,
        expected_outcome: str,
    ) -> IntrinsicGoal:
        """Generate an experimental goal to test a hypothesis."""
        return IntrinsicGoal(
            id=f"exp-{hashlib.md5(hypothesis.encode()).hexdigest()[:8]}",
            goal_type=GoalType.EXPERIMENT,
            description=test_description,
            motivation="Testing hypothesis through experimentation",
            priority=0.5,
            hypothesis=hypothesis,
            expected_outcome=expected_outcome,
            source="hypothesis",
        )

    def get_active_goals(self) -> list[IntrinsicGoal]:
        """Get all active intrinsic goals."""
        goals = []
        for path in self.goals_dir.glob("goal-*.json"):
            try:
                data = json.loads(path.read_text())
                goal = IntrinsicGoal.model_validate(data)
                if goal.status in ("pending", "in_progress"):
                    goals.append(goal)
            except Exception:
                continue
        return sorted(goals, key=lambda g: -g.priority)

    def save_goal(self, goal: IntrinsicGoal) -> None:
        """Save a goal to disk."""
        path = self.goals_dir / f"goal-{goal.id}.json"
        path.write_text(json.dumps(goal.model_dump(mode='json'), indent=2, default=str))

    def update_goal(
        self,
        goal_id: str,
        status: str | None = None,
        progress: float | None = None,
        actual_outcome: str | None = None,
        lessons: list[str] | None = None,
    ) -> IntrinsicGoal | None:
        """Update goal progress."""
        path = self.goals_dir / f"goal-{goal_id}.json"
        if not path.exists():
            return None

        data = json.loads(path.read_text())
        goal = IntrinsicGoal.model_validate(data)

        if status:
            goal.status = status
            if status == "in_progress" and not goal.started_at:
                goal.started_at = datetime.now(UTC)
            elif status == "completed":
                goal.completed_at = datetime.now(UTC)

        if progress is not None:
            goal.progress = progress

        if actual_outcome:
            goal.actual_outcome = actual_outcome

        if lessons:
            goal.lessons.extend(lessons)

        self.save_goal(goal)
        return goal


# =============================================================================
# Error Recovery Learning
# =============================================================================

class ErrorPattern(BaseModel):
    """A recognized pattern in errors."""
    id: str
    error_type: str
    signature: str  # How to recognize this error
    frequency: int = 1
    last_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Recovery
    recovery_procedure: str | None = None
    recovery_success_rate: float = 0.0
    recovery_attempts: int = 0
    recovery_successes: int = 0

    # Prevention
    prevention_hints: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)  # Conditions that led to error


class ErrorRecoveryLearner:
    """
    Learns from errors to build recovery procedures and prevention strategies.

    When errors occur, the agent:
    1. Identifies the error pattern
    2. Records recovery attempts
    3. Builds successful recovery procedures
    4. Identifies prevention strategies
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.patterns_dir = body_dir / "evolution" / "error_patterns"
        self.patterns_dir.mkdir(parents=True, exist_ok=True)

    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: dict[str, Any],
    ) -> ErrorPattern:
        """Record an error occurrence."""
        # Generate signature from error
        signature = self._generate_signature(error_type, error_message)
        pattern_id = hashlib.md5(signature.encode()).hexdigest()[:12]

        # Check if pattern exists
        pattern_path = self.patterns_dir / f"pattern-{pattern_id}.json"

        if pattern_path.exists():
            data = json.loads(pattern_path.read_text())
            pattern = ErrorPattern.model_validate(data)
            pattern.frequency += 1
            pattern.last_seen = datetime.now(UTC)
        else:
            pattern = ErrorPattern(
                id=pattern_id,
                error_type=error_type,
                signature=signature,
            )

        # Add preconditions from context
        if "action" in context:
            if context["action"] not in pattern.preconditions:
                pattern.preconditions.append(context["action"])

        self._save_pattern(pattern)
        return pattern

    def record_recovery_attempt(
        self,
        pattern_id: str,
        success: bool,
        recovery_action: str,
    ) -> ErrorPattern | None:
        """Record a recovery attempt for an error pattern."""
        pattern_path = self.patterns_dir / f"pattern-{pattern_id}.json"
        if not pattern_path.exists():
            return None

        data = json.loads(pattern_path.read_text())
        pattern = ErrorPattern.model_validate(data)

        pattern.recovery_attempts += 1
        if success:
            pattern.recovery_successes += 1
            # Update recovery procedure if this action was successful
            if not pattern.recovery_procedure or pattern.recovery_success_rate < 0.5:
                pattern.recovery_procedure = recovery_action

        pattern.recovery_success_rate = (
            pattern.recovery_successes / pattern.recovery_attempts
            if pattern.recovery_attempts > 0 else 0.0
        )

        self._save_pattern(pattern)
        return pattern

    def get_recovery_suggestion(self, error_type: str, error_message: str) -> str | None:
        """Get a recovery suggestion based on past patterns."""
        signature = self._generate_signature(error_type, error_message)
        pattern_id = hashlib.md5(signature.encode()).hexdigest()[:12]

        pattern_path = self.patterns_dir / f"pattern-{pattern_id}.json"
        if not pattern_path.exists():
            return None

        data = json.loads(pattern_path.read_text())
        pattern = ErrorPattern.model_validate(data)

        if pattern.recovery_procedure and pattern.recovery_success_rate >= 0.5:
            return pattern.recovery_procedure
        return None

    def get_frequent_errors(self, min_frequency: int = 2) -> list[ErrorPattern]:
        """Get frequently occurring error patterns."""
        patterns = []
        for path in self.patterns_dir.glob("pattern-*.json"):
            try:
                data = json.loads(path.read_text())
                pattern = ErrorPattern.model_validate(data)
                if pattern.frequency >= min_frequency:
                    patterns.append(pattern)
            except Exception:
                continue
        return sorted(patterns, key=lambda p: -p.frequency)

    def _generate_signature(self, error_type: str, error_message: str) -> str:
        """Generate a signature for error matching."""
        # Normalize the error message
        import re
        normalized = re.sub(r'\d+', 'N', error_message)  # Replace numbers
        normalized = re.sub(r'/[^\s]+', '/PATH', normalized)  # Replace paths
        normalized = re.sub(r'0x[0-9a-fA-F]+', 'ADDR', normalized)  # Replace addresses
        return f"{error_type}:{normalized[:100]}"

    def _save_pattern(self, pattern: ErrorPattern) -> None:
        """Save pattern to disk."""
        path = self.patterns_dir / f"pattern-{pattern.id}.json"
        path.write_text(json.dumps(pattern.model_dump(mode='json'), indent=2, default=str))


# =============================================================================
# Character Evolution
# =============================================================================

class CharacterEvolutionEvent(BaseModel):
    """An event that shapes character evolution."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str  # decision, success, failure, learning
    description: str
    trait_impacts: dict[str, float] = Field(default_factory=dict)  # trait -> delta
    evidence: str


class CharacterEvolver:
    """
    Tracks and evolves the agent's character based on experience.

    Character emerges from patterns of behavior, not from declarations.
    This module observes behavior and updates character traits accordingly.
    """

    TRAIT_DEFAULTS = {
        "thoroughness": 0.5,
        "risk_tolerance": 0.5,
        "persistence": 0.5,
        "exploration_tendency": 0.5,
        "caution": 0.5,
        "curiosity": 0.5,
    }

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.evolution_dir = body_dir / "evolution" / "character"
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.evolution_dir / "events.json"

    def record_event(self, event: CharacterEvolutionEvent) -> None:
        """Record an event that shapes character."""
        events = self._load_events()
        events.append(event.model_dump(mode='json'))

        # Keep last 1000 events
        if len(events) > 1000:
            events = events[-1000:]

        self.events_file.write_text(json.dumps(events, indent=2, default=str))

    def record_decision(
        self,
        decision: str,
        chose_safe: bool,
        chose_thorough: bool,
        stakes: str = "medium",
    ) -> None:
        """Record a decision and its character implications."""
        impacts = {}

        if chose_safe:
            impacts["caution"] = 0.02
            impacts["risk_tolerance"] = -0.01
        else:
            impacts["caution"] = -0.01
            impacts["risk_tolerance"] = 0.02

        if chose_thorough:
            impacts["thoroughness"] = 0.02
        else:
            impacts["thoroughness"] = -0.01

        # Higher stakes have bigger impact
        multiplier = {"low": 0.5, "medium": 1.0, "high": 2.0}.get(stakes, 1.0)
        impacts = {k: v * multiplier for k, v in impacts.items()}

        event = CharacterEvolutionEvent(
            event_type="decision",
            description=decision,
            trait_impacts=impacts,
            evidence=f"Decision with {stakes} stakes",
        )
        self.record_event(event)

    def record_outcome(
        self,
        task: str,
        success: bool,
        persisted: bool = False,
        explored_alternatives: bool = False,
    ) -> None:
        """Record a task outcome and its character implications."""
        impacts = {}

        if success:
            impacts["confidence"] = 0.01
        else:
            impacts["confidence"] = -0.01
            if persisted:
                impacts["persistence"] = 0.03

        if explored_alternatives:
            impacts["exploration_tendency"] = 0.02
            impacts["curiosity"] = 0.01

        event = CharacterEvolutionEvent(
            event_type="success" if success else "failure",
            description=task,
            trait_impacts=impacts,
            evidence=f"Task outcome: {'success' if success else 'failure'}",
        )
        self.record_event(event)

    def compute_character_update(self, current_character: "Character") -> dict[str, float]:
        """Compute character trait updates based on recent events."""
        events = self._load_events()

        # Only consider recent events (last 100)
        recent = events[-100:] if len(events) > 100 else events

        # Aggregate impacts
        total_impacts: dict[str, float] = defaultdict(float)
        for event_data in recent:
            for trait, delta in event_data.get("trait_impacts", {}).items():
                total_impacts[trait] += delta

        # Apply decay and clamping
        updates = {}
        for trait, delta in total_impacts.items():
            # Get current value
            current = getattr(current_character, trait, self.TRAIT_DEFAULTS.get(trait, 0.5))
            new_value = max(0.0, min(1.0, current + delta * 0.1))  # Dampen changes
            if abs(new_value - current) > 0.01:  # Only report significant changes
                updates[trait] = new_value

        return updates

    def get_character_summary(self) -> str:
        """Generate a summary of character evolution."""
        events = self._load_events()

        if not events:
            return "No character evolution events recorded yet."

        # Count event types
        type_counts: dict[str, int] = defaultdict(int)
        for event in events:
            type_counts[event.get("event_type", "unknown")] += 1

        # Compute net trait changes
        net_changes: dict[str, float] = defaultdict(float)
        for event in events:
            for trait, delta in event.get("trait_impacts", {}).items():
                net_changes[trait] += delta

        lines = ["## Character Evolution Summary"]
        lines.append(f"\nTotal events: {len(events)}")
        lines.append(f"- Decisions: {type_counts.get('decision', 0)}")
        lines.append(f"- Successes: {type_counts.get('success', 0)}")
        lines.append(f"- Failures: {type_counts.get('failure', 0)}")

        if net_changes:
            lines.append("\n### Trait Tendencies")
            for trait, change in sorted(net_changes.items(), key=lambda x: -abs(x[1])):
                direction = "↑" if change > 0 else "↓"
                lines.append(f"- {trait}: {direction} ({change:+.2f})")

        return "\n".join(lines)

    def _load_events(self) -> list[dict]:
        """Load events from file."""
        if not self.events_file.exists():
            return []
        try:
            return json.loads(self.events_file.read_text())
        except Exception:
            return []


# =============================================================================
# Self-Improvement Experiments
# =============================================================================

class Experiment(BaseModel):
    """A self-improvement experiment."""
    id: str
    hypothesis: str
    method: str

    # Execution
    status: str = "pending"  # pending, running, completed, failed
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Metrics
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    result_metrics: dict[str, float] = Field(default_factory=dict)

    # Analysis
    conclusion: str | None = None
    should_adopt: bool | None = None
    lessons: list[str] = Field(default_factory=list)


class ExperimentRunner:
    """
    Runs self-improvement experiments.

    The agent can formulate hypotheses about how to improve,
    design experiments to test them, and measure results.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.experiments_dir = body_dir / "evolution" / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        hypothesis: str,
        method: str,
        baseline_metrics: dict[str, float],
    ) -> Experiment:
        """Create a new experiment."""
        exp_id = hashlib.md5(f"{hypothesis}:{method}".encode()).hexdigest()[:12]

        experiment = Experiment(
            id=exp_id,
            hypothesis=hypothesis,
            method=method,
            baseline_metrics=baseline_metrics,
        )

        self._save_experiment(experiment)
        return experiment

    def start_experiment(self, exp_id: str) -> Experiment | None:
        """Start an experiment."""
        exp = self._load_experiment(exp_id)
        if not exp:
            return None

        exp.status = "running"
        exp.started_at = datetime.now(UTC)
        self._save_experiment(exp)
        return exp

    def complete_experiment(
        self,
        exp_id: str,
        result_metrics: dict[str, float],
        conclusion: str,
        should_adopt: bool,
        lessons: list[str] | None = None,
    ) -> Experiment | None:
        """Complete an experiment with results."""
        exp = self._load_experiment(exp_id)
        if not exp:
            return None

        exp.status = "completed"
        exp.completed_at = datetime.now(UTC)
        exp.result_metrics = result_metrics
        exp.conclusion = conclusion
        exp.should_adopt = should_adopt
        if lessons:
            exp.lessons = lessons

        self._save_experiment(exp)
        return exp

    def get_successful_experiments(self) -> list[Experiment]:
        """Get experiments that should be adopted."""
        experiments = []
        for path in self.experiments_dir.glob("exp-*.json"):
            try:
                data = json.loads(path.read_text())
                exp = Experiment.model_validate(data)
                if exp.status == "completed" and exp.should_adopt:
                    experiments.append(exp)
            except Exception:
                continue
        return experiments

    def _save_experiment(self, exp: Experiment) -> None:
        """Save experiment to disk."""
        path = self.experiments_dir / f"exp-{exp.id}.json"
        path.write_text(json.dumps(exp.model_dump(mode='json'), indent=2, default=str))

    def _load_experiment(self, exp_id: str) -> Experiment | None:
        """Load experiment from disk."""
        path = self.experiments_dir / f"exp-{exp_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return Experiment.model_validate(data)
        except Exception:
            return None


# =============================================================================
# Unified Evolution System
# =============================================================================

class EvolutionSystem:
    """
    Unified interface for agent evolution and self-improvement.

    Coordinates:
    - Self-analysis and performance tracking
    - Intrinsic goal generation (curiosity)
    - Error recovery learning
    - Character evolution
    - Self-improvement experiments
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.evolution_dir = body_dir / "evolution"
        self.evolution_dir.mkdir(parents=True, exist_ok=True)

        # Initialize subsystems
        self.analyzer = SelfAnalyzer(body_dir)
        self.curiosity = CuriosityEngine(body_dir)
        self.error_recovery = ErrorRecoveryLearner(body_dir)
        self.character = CharacterEvolver(body_dir)
        self.experiments = ExperimentRunner(body_dir)

    def daily_self_analysis(self) -> PerformanceReport:
        """Run daily self-analysis and generate improvement goals."""
        # Analyze recent performance
        report = self.analyzer.analyze_recent_performance(days=1)

        # Generate improvement goals from insights
        for priority in report.improvement_priorities[:3]:
            goal = IntrinsicGoal(
                id=f"improve-{hashlib.md5(priority.encode()).hexdigest()[:8]}",
                goal_type=GoalType.SKILL_BUILDING,
                description=priority,
                motivation="Identified through self-analysis",
                priority=0.6,
                source="self_analysis",
            )
            self.curiosity.save_goal(goal)

        return report

    def generate_intrinsic_goals(
        self,
        body: "BodyDirectory",
        skill_manager: "SkillManager | None" = None,
    ) -> list[IntrinsicGoal]:
        """Generate all types of intrinsic goals."""
        goals = []

        # Exploration goals
        goals.extend(self.curiosity.generate_exploration_goals())

        # Skill goals
        if skill_manager:
            goals.extend(self.curiosity.generate_skill_goals(skill_manager))

        # Knowledge goals
        goals.extend(self.curiosity.generate_knowledge_goals(body))

        # Save and return
        for goal in goals:
            self.curiosity.save_goal(goal)

        return goals

    def on_error(
        self,
        error_type: str,
        error_message: str,
        context: dict[str, Any],
    ) -> str | None:
        """Handle an error - record and suggest recovery."""
        # Record the error
        pattern = self.error_recovery.record_error(error_type, error_message, context)

        # Try to get recovery suggestion
        return self.error_recovery.get_recovery_suggestion(error_type, error_message)

    def on_task_complete(
        self,
        task: str,
        success: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Handle task completion - update character and goals."""
        details = details or {}

        # Update character
        self.character.record_outcome(
            task=task,
            success=success,
            persisted=details.get("persisted", False),
            explored_alternatives=details.get("explored_alternatives", False),
        )

        # Check if any intrinsic goals were completed
        active_goals = self.curiosity.get_active_goals()
        for goal in active_goals:
            if goal.description.lower() in task.lower():
                self.curiosity.update_goal(
                    goal.id,
                    status="completed" if success else "in_progress",
                    progress=1.0 if success else goal.progress + 0.2,
                )

    def get_evolution_summary(self) -> str:
        """Get a summary of the agent's evolution state."""
        lines = ["# Agent Evolution Summary\n"]

        # Active goals
        goals = self.curiosity.get_active_goals()
        if goals:
            lines.append("## Active Self-Improvement Goals")
            for goal in goals[:5]:
                lines.append(f"- [{goal.goal_type.value}] {goal.description}")
            lines.append("")

        # Recent errors
        frequent_errors = self.error_recovery.get_frequent_errors()
        if frequent_errors:
            lines.append("## Frequent Errors (learning in progress)")
            for pattern in frequent_errors[:3]:
                recovery = "✓" if pattern.recovery_procedure else "?"
                lines.append(f"- {pattern.error_type}: seen {pattern.frequency}x [{recovery}]")
            lines.append("")

        # Character summary
        lines.append(self.character.get_character_summary())

        # Successful experiments
        successful = self.experiments.get_successful_experiments()
        if successful:
            lines.append("\n## Adopted Improvements")
            for exp in successful[:3]:
                lines.append(f"- {exp.hypothesis[:50]}...")

        return "\n".join(lines)


# =============================================================================
# Evolution Daemons
# =============================================================================

def get_self_analysis_daemon() -> dict[str, Any]:
    """Get pipeline definition for self-analysis daemon."""
    return {
        "name": "self-analyzer",
        "description": "Periodic self-analysis of agent performance",
        "sources": [
            {"path": "{body}/memory/episodes/", "mode": "full"},
            {"path": "{body}/evolution/reports/", "mode": "full"},
        ],
        "trigger": {
            "mode": "every_n_steps",
            "n_steps": 100,
            "debounce_ms": 300000,
        },
        "prompt": """Analyze recent agent performance and identify improvement opportunities.

Review:
1. Task success/failure patterns
2. Common error types
3. Learning progress
4. Skill development

Generate a brief analysis with:
- 2-3 strengths observed
- 2-3 areas for improvement
- 1-2 specific action items

Be concise and actionable.""",
        "output": "self-analysis.md",
        "model": "haiku",
        "max_tokens": 800,
        "temperature": 0.3,
        "enabled": True,
        "priority": 6,
        "group": "maintenance",
        "tags": ["self-improvement", "analysis"],
    }


def get_curiosity_daemon() -> dict[str, Any]:
    """Get pipeline definition for curiosity/goal generation daemon."""
    return {
        "name": "curiosity-engine",
        "description": "Generates intrinsic goals based on curiosity",
        "sources": [
            {"path": "{body}/working_set.json", "mode": "full"},
            {"path": "{body}/evolution/goals/", "mode": "full"},
            {"path": "{cwd}", "mode": "ls"},
        ],
        "trigger": {
            "mode": "every_n_steps",
            "n_steps": 50,
            "debounce_ms": 120000,
        },
        "prompt": """Review the agent's current state and environment.

Consider:
1. What unexplored areas exist?
2. What skills could be improved?
3. What knowledge gaps exist?
4. What experiments could test hypotheses?

Generate 1-2 intrinsic goals that would help the agent grow.
Format as simple bullet points with motivation.

Be selective - only suggest truly valuable goals.""",
        "output": "curiosity-suggestions.md",
        "model": "haiku",
        "max_tokens": 500,
        "temperature": 0.5,
        "enabled": True,
        "priority": 7,
        "group": "maintenance",
        "tags": ["curiosity", "goals"],
    }
