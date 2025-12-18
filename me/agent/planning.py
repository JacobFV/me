"""
Strategic Reasoning and Planning System.

Enables the agent to decompose goals hierarchically, plan sequences of actions,
evaluate plans, and reason strategically about how to achieve objectives.

Philosophy: Intelligence requires foresight. The ability to project into the
future, consider alternatives, and choose paths that lead to desired outcomes
is the essence of strategic reasoning.

Architecture:
    - Goal: Hierarchical goal structure with dependencies
    - GoalGraph: DAG of goals with decomposition relationships
    - Plan: Sequence of actions to achieve a goal
    - Planner: Generate plans from goals using various strategies
    - PlanEvaluator: Score and compare plans
    - StrategicReasoner: High-level strategic thinking
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field


# =============================================================================
# Goal Model
# =============================================================================

class GoalStatus(str, Enum):
    """Status of a goal."""
    PENDING = "pending"        # Not started
    ACTIVE = "active"          # Currently being pursued
    BLOCKED = "blocked"        # Blocked by dependency or obstacle
    COMPLETED = "completed"    # Successfully achieved
    FAILED = "failed"          # Failed to achieve
    ABANDONED = "abandoned"    # Deliberately abandoned


class GoalPriority(str, Enum):
    """Priority level of a goal."""
    CRITICAL = "critical"      # Must be done, highest priority
    HIGH = "high"              # Important, should be done soon
    MEDIUM = "medium"          # Normal priority
    LOW = "low"                # Nice to have, can be deferred
    BACKGROUND = "background"  # Long-term, no urgency


class Goal(BaseModel):
    """
    A goal in the hierarchical goal structure.

    Goals can be decomposed into subgoals, have dependencies,
    and track their status and progress.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.MEDIUM

    # Hierarchy
    parent_id: str | None = None
    subgoal_ids: list[str] = Field(default_factory=list)

    # Dependencies and blockers
    depends_on: list[str] = Field(default_factory=list)  # Goal IDs that must complete first
    blocked_by: list[str] = Field(default_factory=list)  # Current blockers (reasons)

    # Success criteria
    success_criteria: list[str] = Field(default_factory=list)
    completion_percentage: float = 0.0

    # Planning
    estimated_effort: float = 1.0  # Abstract effort units
    actual_effort: float = 0.0
    deadline: datetime | None = None

    # Context
    context: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def is_leaf(self) -> bool:
        """Check if this is a leaf goal (no subgoals)."""
        return len(self.subgoal_ids) == 0

    def is_actionable(self, completed_goals: set[str]) -> bool:
        """Check if goal can be worked on (dependencies met, not blocked)."""
        if self.status not in [GoalStatus.PENDING, GoalStatus.ACTIVE]:
            return False

        # Check dependencies
        for dep_id in self.depends_on:
            if dep_id not in completed_goals:
                return False

        # Check blockers
        if self.blocked_by:
            return False

        return True


# =============================================================================
# Goal Graph
# =============================================================================

class GoalGraph:
    """
    Directed acyclic graph of goals with decomposition relationships.

    Supports:
    - Goal decomposition (parent -> subgoals)
    - Dependencies between goals
    - Finding actionable goals
    - Computing critical path
    """

    def __init__(self, graph_dir: Path):
        self.graph_dir = graph_dir
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self._goals_file = graph_dir / "goals.json"
        self._load()

    def _load(self):
        """Load goals from disk."""
        self._goals: dict[str, Goal] = {}

        if self._goals_file.exists():
            try:
                with open(self._goals_file, 'r') as f:
                    data = json.load(f)
                for goal_data in data:
                    goal = Goal.model_validate(goal_data)
                    self._goals[goal.id] = goal
            except Exception:
                pass

    def _save(self):
        """Save goals to disk."""
        with open(self._goals_file, 'w') as f:
            json.dump(
                [g.model_dump(mode='json') for g in self._goals.values()],
                f, indent=2, default=str
            )

    def add_goal(self, goal: Goal) -> Goal:
        """Add a goal to the graph."""
        self._goals[goal.id] = goal

        # Update parent if specified
        if goal.parent_id and goal.parent_id in self._goals:
            parent = self._goals[goal.parent_id]
            if goal.id not in parent.subgoal_ids:
                parent.subgoal_ids.append(goal.id)

        self._save()
        return goal

    def get_goal(self, goal_id: str) -> Goal | None:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def update_goal(self, goal: Goal):
        """Update an existing goal."""
        self._goals[goal.id] = goal
        self._save()

    def delete_goal(self, goal_id: str, cascade: bool = False):
        """
        Delete a goal.

        If cascade=True, also deletes all subgoals.
        """
        if goal_id not in self._goals:
            return

        goal = self._goals[goal_id]

        if cascade:
            for subgoal_id in goal.subgoal_ids.copy():
                self.delete_goal(subgoal_id, cascade=True)

        # Remove from parent
        if goal.parent_id and goal.parent_id in self._goals:
            parent = self._goals[goal.parent_id]
            if goal_id in parent.subgoal_ids:
                parent.subgoal_ids.remove(goal_id)

        del self._goals[goal_id]
        self._save()

    def decompose_goal(self, parent_id: str, subgoals: list[Goal]) -> list[Goal]:
        """
        Decompose a goal into subgoals.

        Returns the created subgoals.
        """
        if parent_id not in self._goals:
            return []

        created = []

        for subgoal in subgoals:
            subgoal.parent_id = parent_id
            # add_goal will handle adding to parent's subgoal_ids
            self.add_goal(subgoal)
            created.append(subgoal)

        self._save()
        return created

    def get_root_goals(self) -> list[Goal]:
        """Get all top-level goals (no parent)."""
        return [g for g in self._goals.values() if g.parent_id is None]

    def get_subgoals(self, goal_id: str) -> list[Goal]:
        """Get immediate subgoals of a goal."""
        if goal_id not in self._goals:
            return []

        goal = self._goals[goal_id]
        return [self._goals[sid] for sid in goal.subgoal_ids if sid in self._goals]

    def get_all_subgoals(self, goal_id: str) -> list[Goal]:
        """Get all subgoals recursively."""
        if goal_id not in self._goals:
            return []

        result = []
        queue = list(self._goals[goal_id].subgoal_ids)

        while queue:
            sid = queue.pop(0)
            if sid in self._goals:
                subgoal = self._goals[sid]
                result.append(subgoal)
                queue.extend(subgoal.subgoal_ids)

        return result

    def get_actionable_goals(self) -> list[Goal]:
        """Get all goals that can currently be worked on."""
        completed = {
            gid for gid, g in self._goals.items()
            if g.status == GoalStatus.COMPLETED
        }

        actionable = []
        for goal in self._goals.values():
            if goal.is_actionable(completed) and goal.is_leaf():
                actionable.append(goal)

        # Sort by priority
        priority_order = {
            GoalPriority.CRITICAL: 0,
            GoalPriority.HIGH: 1,
            GoalPriority.MEDIUM: 2,
            GoalPriority.LOW: 3,
            GoalPriority.BACKGROUND: 4,
        }
        actionable.sort(key=lambda g: priority_order.get(g.priority, 2))

        return actionable

    def get_blocked_goals(self) -> list[tuple[Goal, list[str]]]:
        """Get all blocked goals with their blockers."""
        blocked = []
        for goal in self._goals.values():
            if goal.status == GoalStatus.BLOCKED or goal.blocked_by:
                blocked.append((goal, goal.blocked_by))
        return blocked

    def compute_completion(self, goal_id: str) -> float:
        """Compute completion percentage based on subgoal completion."""
        if goal_id not in self._goals:
            return 0.0

        goal = self._goals[goal_id]

        if goal.is_leaf():
            return goal.completion_percentage

        # Aggregate from subgoals
        subgoals = self.get_subgoals(goal_id)
        if not subgoals:
            return goal.completion_percentage

        total_weight = sum(sg.estimated_effort for sg in subgoals)
        if total_weight == 0:
            return 0.0

        weighted_completion = sum(
            self.compute_completion(sg.id) * sg.estimated_effort
            for sg in subgoals
        )

        return weighted_completion / total_weight

    def update_status(self, goal_id: str, status: GoalStatus):
        """Update goal status and propagate changes."""
        if goal_id not in self._goals:
            return

        goal = self._goals[goal_id]
        goal.status = status

        if status == GoalStatus.ACTIVE and not goal.started_at:
            goal.started_at = datetime.now(UTC)
        elif status == GoalStatus.COMPLETED:
            goal.completed_at = datetime.now(UTC)
            goal.completion_percentage = 100.0

            # Check if parent can be completed
            if goal.parent_id:
                self._check_parent_completion(goal.parent_id)

        self._save()

    def _check_parent_completion(self, parent_id: str):
        """Check if parent goal is complete when all subgoals complete."""
        if parent_id not in self._goals:
            return

        parent = self._goals[parent_id]
        subgoals = self.get_subgoals(parent_id)

        if not subgoals:
            return

        all_complete = all(
            sg.status == GoalStatus.COMPLETED
            for sg in subgoals
        )

        if all_complete:
            parent.status = GoalStatus.COMPLETED
            parent.completed_at = datetime.now(UTC)
            parent.completion_percentage = 100.0

            # Recurse up
            if parent.parent_id:
                self._check_parent_completion(parent.parent_id)

    def get_critical_path(self, goal_id: str) -> list[Goal]:
        """
        Get critical path to goal completion.

        Returns sequence of goals that must complete for the goal to complete.
        """
        if goal_id not in self._goals:
            return []

        # Get all dependencies and subgoals
        to_complete: list[Goal] = []
        visited: set[str] = set()

        def collect(gid: str):
            if gid in visited or gid not in self._goals:
                return
            visited.add(gid)

            goal = self._goals[gid]

            # Add dependencies first
            for dep_id in goal.depends_on:
                collect(dep_id)

            # Add subgoals
            for sub_id in goal.subgoal_ids:
                collect(sub_id)

            # Add this goal if not complete
            if goal.status != GoalStatus.COMPLETED:
                to_complete.append(goal)

        collect(goal_id)
        return to_complete

    def get_statistics(self) -> dict[str, Any]:
        """Get goal graph statistics."""
        goals = list(self._goals.values())

        return {
            "total_goals": len(goals),
            "by_status": {
                s.value: len([g for g in goals if g.status == s])
                for s in GoalStatus
            },
            "by_priority": {
                p.value: len([g for g in goals if g.priority == p])
                for p in GoalPriority
            },
            "root_goals": len(self.get_root_goals()),
            "actionable_goals": len(self.get_actionable_goals()),
            "blocked_goals": len(self.get_blocked_goals()),
        }


# =============================================================================
# Plan Model
# =============================================================================

class PlanStep(BaseModel):
    """A single step in a plan."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str = ""
    estimated_duration_ms: float = 0.0
    dependencies: list[str] = Field(default_factory=list)  # Step IDs
    completed: bool = False
    outcome: str | None = None


class PlanStatus(str, Enum):
    """Status of a plan."""
    DRAFT = "draft"            # Being developed
    READY = "ready"            # Ready to execute
    EXECUTING = "executing"    # Currently executing
    COMPLETED = "completed"    # Successfully completed
    FAILED = "failed"          # Failed during execution
    ABANDONED = "abandoned"    # Deliberately abandoned


class Plan(BaseModel):
    """
    A plan to achieve a goal.

    Plans are sequences of steps with dependencies, estimates,
    and tracking of execution.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    goal_id: str | None = None
    status: PlanStatus = PlanStatus.DRAFT

    steps: list[PlanStep] = Field(default_factory=list)

    # Estimates
    total_estimated_duration_ms: float = 0.0
    confidence: float = 0.5  # How confident are we in this plan

    # Execution tracking
    current_step_index: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    actual_duration_ms: float = 0.0

    # Alternatives
    alternative_plan_ids: list[str] = Field(default_factory=list)

    # Context
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def get_next_steps(self) -> list[PlanStep]:
        """Get steps that can be executed next (dependencies met)."""
        completed = {s.id for s in self.steps if s.completed}

        ready = []
        for step in self.steps:
            if step.completed:
                continue
            # Check if all dependencies are complete
            if all(dep in completed for dep in step.dependencies):
                ready.append(step)

        return ready

    def mark_step_complete(self, step_id: str, outcome: str = "success"):
        """Mark a step as complete."""
        for step in self.steps:
            if step.id == step_id:
                step.completed = True
                step.outcome = outcome
                break

        # Check if plan is complete
        if all(s.completed for s in self.steps):
            self.status = PlanStatus.COMPLETED
            self.completed_at = datetime.now(UTC)


# =============================================================================
# Planner
# =============================================================================

class PlanningStrategy(str, Enum):
    """Strategies for generating plans."""
    FORWARD = "forward"        # Start from current state, work toward goal
    BACKWARD = "backward"      # Start from goal, work backward to current state
    HIERARCHICAL = "hierarchical"  # Decompose into subplans
    MEANS_ENDS = "means_ends"  # Match means to ends


class Planner:
    """
    Generate plans from goals.

    Supports multiple planning strategies and plan refinement.
    """

    def __init__(self, planner_dir: Path, goal_graph: GoalGraph):
        self.planner_dir = planner_dir
        self.planner_dir.mkdir(parents=True, exist_ok=True)
        self._plans_file = planner_dir / "plans.json"
        self._goal_graph = goal_graph
        self._load()

    def _load(self):
        """Load plans from disk."""
        self._plans: dict[str, Plan] = {}

        if self._plans_file.exists():
            try:
                with open(self._plans_file, 'r') as f:
                    data = json.load(f)
                for plan_data in data:
                    plan = Plan.model_validate(plan_data)
                    self._plans[plan.id] = plan
            except Exception:
                pass

    def _save(self):
        """Save plans to disk."""
        with open(self._plans_file, 'w') as f:
            json.dump(
                [p.model_dump(mode='json') for p in self._plans.values()],
                f, indent=2, default=str
            )

    def create_plan(
        self,
        goal_id: str,
        strategy: PlanningStrategy = PlanningStrategy.FORWARD,
        context: dict[str, Any] | None = None,
    ) -> Plan | None:
        """
        Create a plan to achieve a goal.

        Uses the specified strategy to generate a sequence of steps.
        """
        goal = self._goal_graph.get_goal(goal_id)
        if not goal:
            return None

        if strategy == PlanningStrategy.FORWARD:
            steps = self._plan_forward(goal, context or {})
        elif strategy == PlanningStrategy.BACKWARD:
            steps = self._plan_backward(goal, context or {})
        elif strategy == PlanningStrategy.HIERARCHICAL:
            steps = self._plan_hierarchical(goal, context or {})
        else:
            steps = self._plan_forward(goal, context or {})

        if not steps:
            return None

        plan = Plan(
            name=f"Plan for: {goal.name}",
            description=f"Generated plan to achieve goal: {goal.description}",
            goal_id=goal_id,
            steps=steps,
            total_estimated_duration_ms=sum(s.estimated_duration_ms for s in steps),
            confidence=self._estimate_confidence(steps),
            context=context or {},
        )

        self._plans[plan.id] = plan
        self._save()
        return plan

    def _plan_forward(self, goal: Goal, context: dict[str, Any]) -> list[PlanStep]:
        """Forward planning: current state -> goal."""
        steps = []

        # Check if goal has subgoals
        subgoals = self._goal_graph.get_subgoals(goal.id)

        if subgoals:
            # Create steps for each subgoal
            for sg in subgoals:
                steps.append(PlanStep(
                    action=f"achieve_{sg.id}",
                    description=f"Achieve subgoal: {sg.name}",
                    expected_outcome=sg.name,
                    estimated_duration_ms=sg.estimated_effort * 1000,
                ))
        else:
            # Leaf goal: create single step
            steps.append(PlanStep(
                action=f"achieve_{goal.id}",
                description=f"Achieve goal: {goal.name}",
                expected_outcome=goal.name,
                estimated_duration_ms=goal.estimated_effort * 1000,
            ))

        return steps

    def _plan_backward(self, goal: Goal, context: dict[str, Any]) -> list[PlanStep]:
        """Backward planning: goal -> current state (reversed)."""
        # Start from goal and work backward
        steps = []

        # Get dependencies that must be satisfied
        for dep_id in goal.depends_on:
            dep_goal = self._goal_graph.get_goal(dep_id)
            if dep_goal and dep_goal.status != GoalStatus.COMPLETED:
                steps.append(PlanStep(
                    action=f"satisfy_dependency_{dep_id}",
                    description=f"Satisfy dependency: {dep_goal.name}",
                    expected_outcome=f"{dep_goal.name} completed",
                    estimated_duration_ms=dep_goal.estimated_effort * 1000,
                ))

        # Then achieve the goal itself
        steps.append(PlanStep(
            action=f"achieve_{goal.id}",
            description=f"Achieve goal: {goal.name}",
            expected_outcome=goal.name,
            estimated_duration_ms=goal.estimated_effort * 1000,
            dependencies=[s.id for s in steps],
        ))

        return steps

    def _plan_hierarchical(self, goal: Goal, context: dict[str, Any]) -> list[PlanStep]:
        """Hierarchical planning: decompose into subplans."""
        steps = []

        # Recursively plan for all subgoals
        all_subgoals = self._goal_graph.get_all_subgoals(goal.id)

        for sg in all_subgoals:
            if sg.status != GoalStatus.COMPLETED:
                step = PlanStep(
                    action=f"execute_subgoal_{sg.id}",
                    description=f"Execute: {sg.name}",
                    expected_outcome=f"{sg.name} completed",
                    estimated_duration_ms=sg.estimated_effort * 1000,
                )

                # Add dependencies based on goal dependencies
                dep_steps = [
                    s.id for s in steps
                    if any(sg.depends_on and dep_id in s.action for dep_id in sg.depends_on)
                ]
                step.dependencies = dep_steps

                steps.append(step)

        return steps

    def _estimate_confidence(self, steps: list[PlanStep]) -> float:
        """Estimate confidence in plan success."""
        if not steps:
            return 0.0

        # Simple heuristic: shorter plans have higher confidence
        base_confidence = max(0.3, 1.0 - len(steps) * 0.05)

        # Reduce confidence for steps with many dependencies
        avg_deps = sum(len(s.dependencies) for s in steps) / len(steps)
        dep_penalty = min(0.3, avg_deps * 0.05)

        return max(0.1, base_confidence - dep_penalty)

    def get_plan(self, plan_id: str) -> Plan | None:
        """Get a plan by ID."""
        return self._plans.get(plan_id)

    def get_plans_for_goal(self, goal_id: str) -> list[Plan]:
        """Get all plans for a goal."""
        return [p for p in self._plans.values() if p.goal_id == goal_id]

    def refine_plan(self, plan_id: str, refinements: dict[str, Any]) -> Plan | None:
        """
        Refine an existing plan.

        Refinements can add/remove/modify steps.
        """
        if plan_id not in self._plans:
            return None

        plan = self._plans[plan_id]

        # Apply refinements
        if "add_steps" in refinements:
            for step_data in refinements["add_steps"]:
                plan.steps.append(PlanStep.model_validate(step_data))

        if "remove_steps" in refinements:
            plan.steps = [s for s in plan.steps if s.id not in refinements["remove_steps"]]

        if "reorder" in refinements:
            # Reorder based on new indices
            order = refinements["reorder"]
            plan.steps = [plan.steps[i] for i in order if i < len(plan.steps)]

        # Recalculate estimates
        plan.total_estimated_duration_ms = sum(s.estimated_duration_ms for s in plan.steps)
        plan.confidence = self._estimate_confidence(plan.steps)

        self._save()
        return plan

    def start_plan(self, plan_id: str) -> bool:
        """Start executing a plan."""
        if plan_id not in self._plans:
            return False

        plan = self._plans[plan_id]
        plan.status = PlanStatus.EXECUTING
        plan.started_at = datetime.now(UTC)
        self._save()
        return True

    def complete_step(self, plan_id: str, step_id: str, outcome: str = "success") -> bool:
        """Mark a step as complete."""
        if plan_id not in self._plans:
            return False

        plan = self._plans[plan_id]
        plan.mark_step_complete(step_id, outcome)
        self._save()
        return True


# =============================================================================
# Plan Evaluator
# =============================================================================

class PlanEvaluator:
    """
    Evaluate and compare plans.

    Scores plans based on multiple criteria and recommends the best option.
    """

    def __init__(self, goal_graph: GoalGraph):
        self._goal_graph = goal_graph

    def evaluate_plan(self, plan: Plan) -> dict[str, Any]:
        """
        Evaluate a plan and return scores.

        Considers:
        - Feasibility (can we actually do this?)
        - Efficiency (how long will it take?)
        - Risk (what could go wrong?)
        - Completeness (does it achieve the goal?)
        """
        scores = {
            "feasibility": self._score_feasibility(plan),
            "efficiency": self._score_efficiency(plan),
            "risk": self._score_risk(plan),
            "completeness": self._score_completeness(plan),
        }

        # Overall score is weighted average
        weights = {"feasibility": 0.3, "efficiency": 0.2, "risk": 0.2, "completeness": 0.3}
        overall = sum(scores[k] * weights[k] for k in scores)

        return {
            "scores": scores,
            "overall": overall,
            "recommendation": "proceed" if overall > 0.6 else "reconsider",
        }

    def _score_feasibility(self, plan: Plan) -> float:
        """Score how feasible the plan is."""
        # Based on plan confidence and step count
        return plan.confidence

    def _score_efficiency(self, plan: Plan) -> float:
        """Score plan efficiency."""
        if not plan.steps:
            return 0.0

        # Prefer shorter plans
        step_penalty = min(0.5, len(plan.steps) * 0.05)
        return max(0.1, 1.0 - step_penalty)

    def _score_risk(self, plan: Plan) -> float:
        """Score plan risk (lower is better, but we return risk as 1-risk)."""
        # More dependencies = more risk
        if not plan.steps:
            return 1.0

        total_deps = sum(len(s.dependencies) for s in plan.steps)
        risk = min(0.8, total_deps * 0.1)
        return 1.0 - risk

    def _score_completeness(self, plan: Plan) -> float:
        """Score whether plan achieves the goal."""
        if not plan.goal_id:
            return 0.5

        goal = self._goal_graph.get_goal(plan.goal_id)
        if not goal:
            return 0.5

        # Check if plan steps cover success criteria
        if not goal.success_criteria:
            return 0.7  # Assume decent if no explicit criteria

        covered = 0
        for criterion in goal.success_criteria:
            for step in plan.steps:
                if criterion.lower() in step.description.lower():
                    covered += 1
                    break

        return covered / len(goal.success_criteria) if goal.success_criteria else 0.7

    def compare_plans(self, plans: list[Plan]) -> list[tuple[Plan, dict[str, Any]]]:
        """Compare multiple plans and rank them."""
        evaluated = [(p, self.evaluate_plan(p)) for p in plans]
        evaluated.sort(key=lambda x: x[1]["overall"], reverse=True)
        return evaluated

    def recommend_plan(self, goal_id: str, plans: list[Plan]) -> Plan | None:
        """Recommend the best plan for a goal."""
        if not plans:
            return None

        ranked = self.compare_plans(plans)
        best = ranked[0]

        if best[1]["overall"] > 0.5:
            return best[0]

        return None


# =============================================================================
# Strategic Reasoner
# =============================================================================

class StrategicInsight(BaseModel):
    """An insight from strategic analysis."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    insight_type: str  # e.g., "opportunity", "threat", "bottleneck"
    description: str
    related_goals: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    suggested_actions: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StrategicReasoner:
    """
    High-level strategic thinking.

    Analyzes goals, plans, and context to provide strategic insights
    and recommendations.
    """

    def __init__(
        self,
        reasoner_dir: Path,
        goal_graph: GoalGraph,
        planner: Planner,
    ):
        self.reasoner_dir = reasoner_dir
        self.reasoner_dir.mkdir(parents=True, exist_ok=True)
        self._insights_file = reasoner_dir / "insights.jsonl"
        self._goal_graph = goal_graph
        self._planner = planner

    def analyze_situation(self) -> dict[str, Any]:
        """
        Analyze current strategic situation.

        Returns overview of goals, plans, and identified issues.
        """
        goal_stats = self._goal_graph.get_statistics()
        actionable = self._goal_graph.get_actionable_goals()
        blocked = self._goal_graph.get_blocked_goals()

        # Identify issues
        issues = []
        if blocked:
            issues.append(f"{len(blocked)} goals are blocked")

        if goal_stats["by_status"].get("pending", 0) > goal_stats["by_status"].get("active", 0) * 2:
            issues.append("Many pending goals, consider prioritization")

        return {
            "goal_overview": goal_stats,
            "actionable_goals": [g.name for g in actionable],
            "blocked_goals": [(g.name, b) for g, b in blocked],
            "issues": issues,
            "analyzed_at": datetime.now(UTC).isoformat(),
        }

    def identify_opportunities(self) -> list[StrategicInsight]:
        """Identify strategic opportunities."""
        insights = []

        # Check for goals that could be parallelized
        actionable = self._goal_graph.get_actionable_goals()
        if len(actionable) > 1:
            insights.append(StrategicInsight(
                insight_type="opportunity",
                description=f"Can pursue {len(actionable)} goals in parallel",
                related_goals=[g.id for g in actionable],
                confidence=0.8,
                suggested_actions=["Prioritize goals", "Allocate resources to parallel execution"],
            ))

        # Check for goals with high estimated effort that could be decomposed
        for goal in self._goal_graph._goals.values():
            if goal.estimated_effort > 10 and goal.is_leaf():
                insights.append(StrategicInsight(
                    insight_type="opportunity",
                    description=f"Large goal '{goal.name}' could be decomposed",
                    related_goals=[goal.id],
                    confidence=0.7,
                    suggested_actions=["Decompose into smaller subgoals"],
                ))

        self._persist_insights(insights)
        return insights

    def identify_risks(self) -> list[StrategicInsight]:
        """Identify strategic risks."""
        insights = []

        # Check for goals with tight deadlines
        now = datetime.now(UTC)
        for goal in self._goal_graph._goals.values():
            if goal.deadline and goal.status not in [GoalStatus.COMPLETED, GoalStatus.ABANDONED]:
                time_remaining = (goal.deadline - now).total_seconds()
                if time_remaining < 0:
                    insights.append(StrategicInsight(
                        insight_type="threat",
                        description=f"Goal '{goal.name}' is past deadline",
                        related_goals=[goal.id],
                        confidence=1.0,
                        suggested_actions=["Assess impact", "Negotiate deadline", "Escalate"],
                    ))
                elif time_remaining < 86400 * 2:  # 2 days
                    insights.append(StrategicInsight(
                        insight_type="threat",
                        description=f"Goal '{goal.name}' deadline in <2 days",
                        related_goals=[goal.id],
                        confidence=0.9,
                        suggested_actions=["Prioritize", "Focus resources"],
                    ))

        # Check for blocked goals
        blocked = self._goal_graph.get_blocked_goals()
        for goal, blockers in blocked:
            insights.append(StrategicInsight(
                insight_type="bottleneck",
                description=f"Goal '{goal.name}' blocked by: {', '.join(blockers)}",
                related_goals=[goal.id],
                confidence=0.95,
                suggested_actions=["Address blockers", "Find workaround"],
            ))

        self._persist_insights(insights)
        return insights

    def recommend_focus(self) -> dict[str, Any]:
        """Recommend what to focus on."""
        actionable = self._goal_graph.get_actionable_goals()

        if not actionable:
            return {
                "recommendation": "No actionable goals",
                "suggested_action": "Decompose pending goals or resolve blockers",
            }

        # Recommend highest priority actionable goal
        top_goal = actionable[0]

        # Check if there's a plan for it
        plans = self._planner.get_plans_for_goal(top_goal.id)

        return {
            "recommendation": f"Focus on: {top_goal.name}",
            "goal_id": top_goal.id,
            "priority": top_goal.priority.value,
            "has_plan": len(plans) > 0,
            "next_action": plans[0].get_next_steps()[0].action if plans and plans[0].get_next_steps() else "Create plan",
        }

    def suggest_goal_decomposition(self, goal_id: str) -> list[str]:
        """Suggest how to decompose a goal."""
        goal = self._goal_graph.get_goal(goal_id)
        if not goal:
            return []

        # Generate suggested subgoals based on goal name/description
        suggestions = []

        # Generic decomposition patterns
        if "implement" in goal.name.lower():
            suggestions.extend([
                "Design solution",
                "Implement core functionality",
                "Write tests",
                "Document changes",
            ])
        elif "fix" in goal.name.lower():
            suggestions.extend([
                "Reproduce the issue",
                "Identify root cause",
                "Implement fix",
                "Verify fix",
            ])
        elif "research" in goal.name.lower():
            suggestions.extend([
                "Gather information",
                "Analyze findings",
                "Synthesize conclusions",
            ])
        else:
            suggestions.extend([
                "Understand requirements",
                "Plan approach",
                "Execute",
                "Verify completion",
            ])

        return suggestions

    def _persist_insights(self, insights: list[StrategicInsight]):
        """Persist insights to file."""
        with open(self._insights_file, 'a') as f:
            for insight in insights:
                f.write(insight.model_dump_json() + '\n')


# =============================================================================
# Strategic Planning System (Unified Interface)
# =============================================================================

class StrategicPlanningSystem:
    """
    Unified interface for strategic reasoning and planning.
    """

    def __init__(self, planning_dir: Path):
        self.planning_dir = planning_dir
        self.planning_dir.mkdir(parents=True, exist_ok=True)

        self.goals = GoalGraph(planning_dir / "goals")
        self.planner = Planner(planning_dir / "plans", self.goals)
        self.evaluator = PlanEvaluator(self.goals)
        self.strategist = StrategicReasoner(
            planning_dir / "strategy", self.goals, self.planner
        )

    def add_goal(
        self,
        name: str,
        description: str = "",
        priority: GoalPriority = GoalPriority.MEDIUM,
        parent_id: str | None = None,
        success_criteria: list[str] | None = None,
    ) -> Goal:
        """Add a new goal."""
        goal = Goal(
            name=name,
            description=description,
            priority=priority,
            parent_id=parent_id,
            success_criteria=success_criteria or [],
        )
        return self.goals.add_goal(goal)

    def decompose_goal(self, goal_id: str, subgoal_names: list[str]) -> list[Goal]:
        """Decompose a goal into subgoals."""
        subgoals = [Goal(name=name) for name in subgoal_names]
        return self.goals.decompose_goal(goal_id, subgoals)

    def create_plan(self, goal_id: str, strategy: PlanningStrategy = PlanningStrategy.FORWARD) -> Plan | None:
        """Create a plan for a goal."""
        return self.planner.create_plan(goal_id, strategy)

    def evaluate_plans(self, goal_id: str) -> list[tuple[Plan, dict[str, Any]]]:
        """Evaluate all plans for a goal."""
        plans = self.planner.get_plans_for_goal(goal_id)
        return self.evaluator.compare_plans(plans)

    def get_strategic_analysis(self) -> dict[str, Any]:
        """Get full strategic analysis."""
        return {
            "situation": self.strategist.analyze_situation(),
            "opportunities": [i.model_dump() for i in self.strategist.identify_opportunities()],
            "risks": [i.model_dump() for i in self.strategist.identify_risks()],
            "recommended_focus": self.strategist.recommend_focus(),
        }

    def complete_goal(self, goal_id: str):
        """Mark a goal as complete."""
        self.goals.update_status(goal_id, GoalStatus.COMPLETED)

    def block_goal(self, goal_id: str, reason: str):
        """Block a goal with a reason."""
        goal = self.goals.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.BLOCKED
            goal.blocked_by.append(reason)
            self.goals.update_goal(goal)

    def get_next_actions(self) -> list[dict[str, Any]]:
        """Get recommended next actions."""
        actionable = self.goals.get_actionable_goals()
        actions = []

        for goal in actionable[:3]:  # Top 3
            plans = self.planner.get_plans_for_goal(goal.id)
            if plans:
                next_steps = plans[0].get_next_steps()
                if next_steps:
                    actions.append({
                        "goal": goal.name,
                        "goal_id": goal.id,
                        "action": next_steps[0].action,
                        "description": next_steps[0].description,
                    })
            else:
                actions.append({
                    "goal": goal.name,
                    "goal_id": goal.id,
                    "action": "create_plan",
                    "description": "No plan exists, create one",
                })

        return actions
