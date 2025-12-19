"""
Strategic Reasoning - Long-horizon planning with resource awareness.

This module enables the agent to think strategically about:
1. Long-term goals that span multiple sessions
2. Resource management (tokens, time, context)
3. Opportunity costs and trade-offs
4. Multi-step planning with dependencies
5. Strategic pivots when plans fail

Philosophy: An agent that only thinks one step ahead will miss
opportunities and waste resources. Strategic reasoning allows the
agent to maintain coherent long-term behavior while being adaptive
to changing circumstances.

This is different from tactical planning (immediate task breakdown).
Strategic reasoning considers:
- What should I be doing vs what am I being asked to do?
- Is this the best use of my resources right now?
- What opportunities am I giving up by pursuing this path?
- How does this fit into my broader development trajectory?
"""

from __future__ import annotations

from datetime import datetime, UTC, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
import json
import hashlib

from pydantic import BaseModel, Field


class ResourceType(str, Enum):
    """Types of resources the agent manages."""
    TOKENS = "tokens"  # LLM tokens
    TIME = "time"  # Wall clock time
    CONTEXT = "context"  # Context window space
    MEMORY = "memory"  # Memory storage
    ATTENTION = "attention"  # Focus/priority
    ENERGY = "energy"  # Abstract processing capacity


class StrategicHorizon(str, Enum):
    """Time horizons for strategic thinking."""
    IMMEDIATE = "immediate"  # This interaction
    SESSION = "session"  # This session
    SHORT_TERM = "short_term"  # Days
    MEDIUM_TERM = "medium_term"  # Weeks
    LONG_TERM = "long_term"  # Months


class ResourceBudget(BaseModel):
    """Budget for a specific resource."""
    resource_type: ResourceType
    total: float
    consumed: float = 0.0
    reserved: float = 0.0  # Reserved for future use

    @property
    def available(self) -> float:
        return self.total - self.consumed - self.reserved

    @property
    def utilization(self) -> float:
        return self.consumed / self.total if self.total > 0 else 0.0


class ResourceState(BaseModel):
    """Current state of all resources."""
    budgets: dict[ResourceType, ResourceBudget] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def get_budget(self, resource_type: ResourceType) -> ResourceBudget | None:
        return self.budgets.get(resource_type)

    def consume(self, resource_type: ResourceType, amount: float) -> bool:
        """Consume resources, returns False if insufficient."""
        budget = self.budgets.get(resource_type)
        if not budget:
            return True  # No budget tracking for this resource

        if budget.available >= amount:
            budget.consumed += amount
            return True
        return False

    def reserve(self, resource_type: ResourceType, amount: float) -> bool:
        """Reserve resources for future use."""
        budget = self.budgets.get(resource_type)
        if not budget:
            return True

        if budget.available >= amount:
            budget.reserved += amount
            return True
        return False


class StrategicGoal(BaseModel):
    """A long-term goal that spans multiple sessions."""
    id: str
    name: str
    description: str
    horizon: StrategicHorizon
    # Progress tracking
    target_metric: str  # What to measure
    current_value: float = 0.0
    target_value: float = 1.0
    # Priority and constraints
    priority: float = 0.5  # 0-1
    deadline: datetime | None = None
    # Dependencies
    depends_on: list[str] = Field(default_factory=list)
    enables: list[str] = Field(default_factory=list)
    # Resource requirements
    estimated_resources: dict[ResourceType, float] = Field(default_factory=dict)
    # Status
    status: str = "active"  # active, paused, completed, abandoned
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_progress: datetime | None = None

    @property
    def progress(self) -> float:
        if self.target_value == 0:
            return 1.0
        return min(1.0, self.current_value / self.target_value)


class OpportunityCost(BaseModel):
    """The cost of not pursuing an alternative."""
    forgone_option: str
    description: str
    estimated_value: float
    resources_that_could_be_reallocated: dict[ResourceType, float] = Field(default_factory=dict)


class StrategicOption(BaseModel):
    """An option being considered in strategic reasoning."""
    id: str
    description: str
    # Value assessment
    expected_value: float
    confidence: float  # How certain we are about the value
    variance: float  # How variable the outcome could be
    # Resource requirements
    resource_costs: dict[ResourceType, float] = Field(default_factory=dict)
    # Trade-offs
    opportunity_costs: list[OpportunityCost] = Field(default_factory=list)
    # Alignment
    goal_alignment: dict[str, float] = Field(default_factory=dict)  # goal_id -> alignment score
    # Timing
    time_sensitivity: float = 0.0  # 0 = flexible, 1 = urgent


class StrategicDecision(BaseModel):
    """A recorded strategic decision."""
    id: str
    timestamp: datetime
    context: str
    options_considered: list[StrategicOption]
    chosen_option: str
    rationale: str
    expected_outcome: str
    # For learning
    actual_outcome: str | None = None
    outcome_value: float | None = None
    lessons_learned: list[str] = Field(default_factory=list)


class ResourceTracker:
    """
    Tracks resource consumption across the agent's activities.

    Provides real-time visibility into resource utilization
    and enables resource-aware decision making.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.state_file = body_dir / "strategic" / "resources.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state: ResourceState | None = None
        self._load_state()

    def _load_state(self) -> None:
        """Load resource state from file."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self._state = ResourceState.model_validate(data)
            except Exception:
                self._init_default_state()
        else:
            self._init_default_state()

    def _init_default_state(self) -> None:
        """Initialize with default resource budgets."""
        self._state = ResourceState(
            budgets={
                ResourceType.TOKENS: ResourceBudget(
                    resource_type=ResourceType.TOKENS,
                    total=100000,  # Default token budget per session
                ),
                ResourceType.TIME: ResourceBudget(
                    resource_type=ResourceType.TIME,
                    total=3600,  # 1 hour default
                ),
                ResourceType.CONTEXT: ResourceBudget(
                    resource_type=ResourceType.CONTEXT,
                    total=100000,  # Context window
                ),
                ResourceType.ATTENTION: ResourceBudget(
                    resource_type=ResourceType.ATTENTION,
                    total=100,  # Abstract attention units
                ),
            }
        )

    def _save_state(self) -> None:
        """Save resource state."""
        if self._state:
            self.state_file.write_text(
                json.dumps(self._state.model_dump(mode='json'), indent=2, default=str)
            )

    @property
    def state(self) -> ResourceState:
        if not self._state:
            self._load_state()
        return self._state  # type: ignore

    def consume(self, resource_type: ResourceType, amount: float) -> bool:
        """Record resource consumption."""
        result = self.state.consume(resource_type, amount)
        self._save_state()
        return result

    def reserve(self, resource_type: ResourceType, amount: float) -> bool:
        """Reserve resources for future use."""
        result = self.state.reserve(resource_type, amount)
        self._save_state()
        return result

    def get_utilization(self) -> dict[ResourceType, float]:
        """Get utilization across all resources."""
        return {
            rt: budget.utilization
            for rt, budget in self.state.budgets.items()
        }

    def get_availability(self) -> dict[ResourceType, float]:
        """Get available amount of each resource."""
        return {
            rt: budget.available
            for rt, budget in self.state.budgets.items()
        }

    def reset_session(self) -> None:
        """Reset for new session."""
        self._init_default_state()
        self._save_state()

    def set_budget(self, resource_type: ResourceType, total: float) -> None:
        """Set budget for a resource."""
        self.state.budgets[resource_type] = ResourceBudget(
            resource_type=resource_type,
            total=total,
        )
        self._save_state()


class GoalManager:
    """
    Manages long-term strategic goals.

    Handles goal lifecycle, progress tracking, and goal graph
    (dependencies and enablements).
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.goals_dir = body_dir / "strategic" / "goals"
        self.goals_dir.mkdir(parents=True, exist_ok=True)

    def create_goal(
        self,
        name: str,
        description: str,
        horizon: StrategicHorizon,
        target_metric: str,
        target_value: float = 1.0,
        priority: float = 0.5,
        deadline: datetime | None = None,
        depends_on: list[str] | None = None,
        estimated_resources: dict[ResourceType, float] | None = None,
    ) -> StrategicGoal:
        """Create a new strategic goal."""
        goal_id = hashlib.md5(f"{name}:{datetime.now(UTC).isoformat()}".encode()).hexdigest()[:12]

        goal = StrategicGoal(
            id=goal_id,
            name=name,
            description=description,
            horizon=horizon,
            target_metric=target_metric,
            target_value=target_value,
            priority=priority,
            deadline=deadline,
            depends_on=depends_on or [],
            estimated_resources=estimated_resources or {},
        )

        self._save_goal(goal)
        return goal

    def _save_goal(self, goal: StrategicGoal) -> None:
        """Save goal to disk."""
        path = self.goals_dir / f"{goal.id}.json"
        path.write_text(json.dumps(goal.model_dump(mode='json'), indent=2, default=str))

    def get_goal(self, goal_id: str) -> StrategicGoal | None:
        """Get a goal by ID."""
        path = self.goals_dir / f"{goal_id}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return StrategicGoal.model_validate(data)
            except Exception:
                return None
        return None

    def get_active_goals(self) -> list[StrategicGoal]:
        """Get all active goals."""
        goals = []
        for path in self.goals_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                goal = StrategicGoal.model_validate(data)
                if goal.status == "active":
                    goals.append(goal)
            except Exception:
                continue
        return sorted(goals, key=lambda g: -g.priority)

    def update_progress(self, goal_id: str, new_value: float) -> None:
        """Update goal progress."""
        goal = self.get_goal(goal_id)
        if goal:
            goal.current_value = new_value
            goal.last_progress = datetime.now(UTC)
            if goal.progress >= 1.0:
                goal.status = "completed"
            self._save_goal(goal)

    def get_goal_graph(self) -> dict[str, list[str]]:
        """Get goal dependency graph."""
        goals = self.get_active_goals()
        graph: dict[str, list[str]] = {}
        for goal in goals:
            graph[goal.id] = goal.depends_on
        return graph

    def get_ready_goals(self) -> list[StrategicGoal]:
        """Get goals whose dependencies are met."""
        goals = self.get_active_goals()
        completed = {
            g.id for g in self.get_all_goals() if g.status == "completed"
        }

        ready = []
        for goal in goals:
            if all(dep in completed for dep in goal.depends_on):
                ready.append(goal)

        return ready

    def get_all_goals(self) -> list[StrategicGoal]:
        """Get all goals (any status)."""
        goals = []
        for path in self.goals_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                goals.append(StrategicGoal.model_validate(data))
            except Exception:
                continue
        return goals


class OptionEvaluator:
    """
    Evaluates strategic options considering multiple factors.

    Computes expected value considering:
    - Direct value of the option
    - Resource costs
    - Opportunity costs
    - Goal alignment
    - Risk/variance
    """

    def __init__(
        self,
        risk_aversion: float = 0.5,  # 0 = risk-seeking, 1 = risk-averse
        goal_weight: float = 0.3,
        resource_weight: float = 0.2,
        opportunity_weight: float = 0.2,
    ):
        self.risk_aversion = risk_aversion
        self.goal_weight = goal_weight
        self.resource_weight = resource_weight
        self.opportunity_weight = opportunity_weight

    def compute_adjusted_value(
        self,
        option: StrategicOption,
        resource_state: ResourceState,
        active_goals: list[StrategicGoal],
    ) -> float:
        """Compute risk-adjusted value of an option."""
        # Base expected value with risk adjustment
        # Higher variance → lower value for risk-averse agent
        risk_penalty = self.risk_aversion * option.variance
        adjusted_base = option.expected_value * option.confidence - risk_penalty

        # Resource feasibility
        resource_score = 1.0
        for resource_type, cost in option.resource_costs.items():
            budget = resource_state.get_budget(resource_type)
            if budget:
                if cost > budget.available:
                    resource_score *= 0.1  # Heavy penalty for infeasible
                else:
                    # Penalize based on how much of available we're using
                    utilization = cost / budget.available if budget.available > 0 else 1.0
                    resource_score *= (1 - 0.5 * utilization)

        # Goal alignment score
        goal_score = 0.0
        if active_goals and option.goal_alignment:
            for goal in active_goals:
                alignment = option.goal_alignment.get(goal.id, 0.0)
                goal_score += alignment * goal.priority
            goal_score /= sum(g.priority for g in active_goals)

        # Opportunity cost
        opportunity_penalty = sum(
            oc.estimated_value for oc in option.opportunity_costs
        ) * self.opportunity_weight

        # Combine factors
        final_value = (
            adjusted_base * (1 - self.goal_weight - self.resource_weight) +
            goal_score * self.goal_weight +
            resource_score * self.resource_weight -
            opportunity_penalty
        )

        return final_value

    def rank_options(
        self,
        options: list[StrategicOption],
        resource_state: ResourceState,
        active_goals: list[StrategicGoal],
    ) -> list[tuple[StrategicOption, float]]:
        """Rank options by adjusted value."""
        scored = [
            (opt, self.compute_adjusted_value(opt, resource_state, active_goals))
            for opt in options
        ]
        return sorted(scored, key=lambda x: -x[1])


class StrategicPlanner:
    """
    Plans strategic paths toward goals.

    Creates multi-step plans that consider:
    - Goal dependencies
    - Resource constraints
    - Timing and deadlines
    - Parallel vs sequential execution
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.goal_manager = GoalManager(body_dir)
        self.resource_tracker = ResourceTracker(body_dir)

    def create_strategic_plan(
        self,
        goals: list[StrategicGoal] | None = None,
    ) -> dict[str, Any]:
        """Create a strategic plan to achieve goals."""
        if goals is None:
            goals = self.goal_manager.get_ready_goals()

        if not goals:
            return {"phases": [], "message": "No ready goals to plan for"}

        # Group goals by horizon
        by_horizon: dict[StrategicHorizon, list[StrategicGoal]] = {}
        for goal in goals:
            if goal.horizon not in by_horizon:
                by_horizon[goal.horizon] = []
            by_horizon[goal.horizon].append(goal)

        # Create phased plan
        phases = []

        # Immediate goals first
        if StrategicHorizon.IMMEDIATE in by_horizon:
            phases.append({
                "name": "Immediate Actions",
                "goals": [g.model_dump() for g in by_horizon[StrategicHorizon.IMMEDIATE]],
                "parallel": True,  # Can do these in parallel
            })

        # Session goals
        if StrategicHorizon.SESSION in by_horizon:
            phases.append({
                "name": "Session Goals",
                "goals": [g.model_dump() for g in by_horizon[StrategicHorizon.SESSION]],
                "parallel": False,  # Likely sequential
            })

        # Longer-term goals as background
        long_term_goals = []
        for horizon in [StrategicHorizon.SHORT_TERM, StrategicHorizon.MEDIUM_TERM, StrategicHorizon.LONG_TERM]:
            if horizon in by_horizon:
                long_term_goals.extend(by_horizon[horizon])

        if long_term_goals:
            phases.append({
                "name": "Long-term Progress",
                "goals": [g.model_dump() for g in long_term_goals],
                "note": "Make incremental progress on these when opportunities arise",
            })

        return {
            "phases": phases,
            "resource_state": self.resource_tracker.get_availability(),
        }

    def should_pivot(
        self,
        current_goal: StrategicGoal,
        progress_rate: float,
        resources_consumed: dict[ResourceType, float],
    ) -> tuple[bool, str]:
        """Determine if we should pivot from current goal."""
        # Check if we're making enough progress
        if progress_rate < 0.1:  # Less than 10% progress rate
            return True, "Progress rate too low"

        # Check resource efficiency
        expected_total = sum(current_goal.estimated_resources.values())
        consumed_total = sum(resources_consumed.values())

        if expected_total > 0:
            efficiency = current_goal.progress / (consumed_total / expected_total)
            if efficiency < 0.5:
                return True, "Resource efficiency too low"

        # Check deadline
        if current_goal.deadline:
            remaining = (current_goal.deadline - datetime.now(UTC)).total_seconds()
            remaining_progress = 1.0 - current_goal.progress
            if remaining_progress > 0 and remaining > 0:
                required_rate = remaining_progress / remaining
                if progress_rate < required_rate * 0.5:
                    return True, "Will miss deadline at current rate"

        return False, "Continue current path"


class DecisionLogger:
    """
    Logs strategic decisions for learning and accountability.

    Maintains a record of:
    - What options were considered
    - Why the chosen option was selected
    - What the expected vs actual outcomes were
    - Lessons learned
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.decisions_dir = body_dir / "strategic" / "decisions"
        self.decisions_dir.mkdir(parents=True, exist_ok=True)

    def log_decision(
        self,
        context: str,
        options: list[StrategicOption],
        chosen_id: str,
        rationale: str,
        expected_outcome: str,
    ) -> StrategicDecision:
        """Log a strategic decision."""
        decision_id = hashlib.md5(
            f"{context}:{datetime.now(UTC).isoformat()}".encode()
        ).hexdigest()[:12]

        decision = StrategicDecision(
            id=decision_id,
            timestamp=datetime.now(UTC),
            context=context,
            options_considered=options,
            chosen_option=chosen_id,
            rationale=rationale,
            expected_outcome=expected_outcome,
        )

        self._save_decision(decision)
        return decision

    def _save_decision(self, decision: StrategicDecision) -> None:
        """Save decision to disk."""
        path = self.decisions_dir / f"{decision.id}.json"
        path.write_text(json.dumps(decision.model_dump(mode='json'), indent=2, default=str))

    def record_outcome(
        self,
        decision_id: str,
        actual_outcome: str,
        outcome_value: float,
        lessons: list[str],
    ) -> None:
        """Record the outcome of a decision."""
        path = self.decisions_dir / f"{decision_id}.json"
        if path.exists():
            data = json.loads(path.read_text())
            data["actual_outcome"] = actual_outcome
            data["outcome_value"] = outcome_value
            data["lessons_learned"] = lessons
            path.write_text(json.dumps(data, indent=2, default=str))

    def get_recent_decisions(self, limit: int = 10) -> list[StrategicDecision]:
        """Get recent decisions."""
        decisions = []
        for path in self.decisions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                decisions.append(StrategicDecision.model_validate(data))
            except Exception:
                continue

        return sorted(decisions, key=lambda d: d.timestamp, reverse=True)[:limit]

    def analyze_decision_quality(self) -> dict[str, Any]:
        """Analyze quality of past decisions."""
        decisions = self.get_recent_decisions(100)

        if not decisions:
            return {"decisions_analyzed": 0}

        decisions_with_outcomes = [
            d for d in decisions if d.outcome_value is not None
        ]

        if not decisions_with_outcomes:
            return {
                "decisions_analyzed": len(decisions),
                "decisions_with_outcomes": 0,
            }

        outcome_values = [d.outcome_value for d in decisions_with_outcomes]
        avg_outcome = sum(outcome_values) / len(outcome_values) if outcome_values else 0

        # Collect all lessons
        all_lessons = []
        for d in decisions_with_outcomes:
            all_lessons.extend(d.lessons_learned)

        return {
            "decisions_analyzed": len(decisions),
            "decisions_with_outcomes": len(decisions_with_outcomes),
            "average_outcome_value": avg_outcome,
            "lessons_collected": len(all_lessons),
            "unique_lessons": len(set(all_lessons)),
        }


class StrategicReasoningSystem:
    """
    Unified system for strategic reasoning.

    Provides a single interface for:
    - Goal management
    - Resource tracking
    - Option evaluation
    - Strategic planning
    - Decision logging
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.resource_tracker = ResourceTracker(body_dir)
        self.goal_manager = GoalManager(body_dir)
        self.evaluator = OptionEvaluator()
        self.planner = StrategicPlanner(body_dir)
        self.decision_logger = DecisionLogger(body_dir)

    def create_goal(
        self,
        name: str,
        description: str,
        horizon: StrategicHorizon = StrategicHorizon.SESSION,
        target_metric: str = "completion",
        target_value: float = 1.0,
        priority: float = 0.5,
        **kwargs: Any,
    ) -> StrategicGoal:
        """Create a strategic goal."""
        return self.goal_manager.create_goal(
            name=name,
            description=description,
            horizon=horizon,
            target_metric=target_metric,
            target_value=target_value,
            priority=priority,
            **kwargs,
        )

    def evaluate_options(
        self,
        options: list[StrategicOption],
    ) -> list[tuple[StrategicOption, float]]:
        """Evaluate and rank strategic options."""
        return self.evaluator.rank_options(
            options,
            self.resource_tracker.state,
            self.goal_manager.get_active_goals(),
        )

    def make_decision(
        self,
        context: str,
        options: list[StrategicOption],
    ) -> tuple[StrategicOption, StrategicDecision]:
        """Make and log a strategic decision."""
        # Rank options
        ranked = self.evaluate_options(options)

        if not ranked:
            raise ValueError("No options to evaluate")

        best_option, best_value = ranked[0]

        # Build rationale
        rationale_parts = [f"Selected '{best_option.description}' (value: {best_value:.2f})"]
        if len(ranked) > 1:
            rationale_parts.append(f"over {len(ranked) - 1} alternatives")
            runner_up = ranked[1][0]
            rationale_parts.append(f"(next best: '{runner_up.description}' at {ranked[1][1]:.2f})")

        rationale = " ".join(rationale_parts)

        # Log decision
        decision = self.decision_logger.log_decision(
            context=context,
            options=options,
            chosen_id=best_option.id,
            rationale=rationale,
            expected_outcome=f"Expected value: {best_option.expected_value}",
        )

        return best_option, decision

    def get_strategic_plan(self) -> dict[str, Any]:
        """Get current strategic plan."""
        return self.planner.create_strategic_plan()

    def check_pivot(
        self,
        goal_id: str,
        progress_rate: float,
        resources_consumed: dict[ResourceType, float],
    ) -> tuple[bool, str]:
        """Check if we should pivot from a goal."""
        goal = self.goal_manager.get_goal(goal_id)
        if not goal:
            return False, "Goal not found"

        return self.planner.should_pivot(goal, progress_rate, resources_consumed)

    def consume_resources(self, resource_type: ResourceType, amount: float) -> bool:
        """Record resource consumption."""
        return self.resource_tracker.consume(resource_type, amount)

    def get_resource_status(self) -> dict[str, Any]:
        """Get current resource status."""
        return {
            "utilization": self.resource_tracker.get_utilization(),
            "availability": self.resource_tracker.get_availability(),
        }

    def get_goal_progress(self) -> list[dict[str, Any]]:
        """Get progress on all active goals."""
        goals = self.goal_manager.get_active_goals()
        return [
            {
                "id": g.id,
                "name": g.name,
                "progress": g.progress,
                "priority": g.priority,
                "horizon": g.horizon.value,
            }
            for g in goals
        ]

    def record_decision_outcome(
        self,
        decision_id: str,
        outcome: str,
        value: float,
        lessons: list[str],
    ) -> None:
        """Record outcome of a past decision."""
        self.decision_logger.record_outcome(decision_id, outcome, value, lessons)

    def get_decision_stats(self) -> dict[str, Any]:
        """Get decision-making statistics."""
        return self.decision_logger.analyze_decision_quality()
