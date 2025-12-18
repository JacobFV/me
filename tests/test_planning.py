"""Tests for strategic reasoning and planning system."""

import tempfile
from datetime import datetime, UTC, timedelta
from pathlib import Path

import pytest

from me.agent.planning import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalGraph,
    Plan,
    PlanStep,
    PlanStatus,
    Planner,
    PlanningStrategy,
    PlanEvaluator,
    StrategicReasoner,
    StrategicInsight,
    StrategicPlanningSystem,
)


# =============================================================================
# Goal Model Tests
# =============================================================================

class TestGoalModel:
    def test_create_goal(self):
        goal = Goal(name="Test Goal", description="A test goal")
        assert goal.name == "Test Goal"
        assert goal.status == GoalStatus.PENDING
        assert goal.priority == GoalPriority.MEDIUM

    def test_is_leaf(self):
        goal = Goal(name="Leaf")
        assert goal.is_leaf() is True

        goal.subgoal_ids = ["sub1", "sub2"]
        assert goal.is_leaf() is False

    def test_is_actionable(self):
        goal = Goal(name="Test", depends_on=["dep1"])

        # Not actionable (dependency not met)
        assert goal.is_actionable(set()) is False

        # Actionable (dependency met)
        assert goal.is_actionable({"dep1"}) is True

    def test_is_actionable_when_blocked(self):
        goal = Goal(name="Test", blocked_by=["Some reason"])
        assert goal.is_actionable(set()) is False


# =============================================================================
# GoalGraph Tests
# =============================================================================

class TestGoalGraph:
    def test_add_and_get_goal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir))

            goal = Goal(name="Test Goal")
            graph.add_goal(goal)

            retrieved = graph.get_goal(goal.id)
            assert retrieved is not None
            assert retrieved.name == "Test Goal"

    def test_decompose_goal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir))

            parent = Goal(name="Parent Goal")
            graph.add_goal(parent)

            subgoals = [
                Goal(name="Subgoal 1"),
                Goal(name="Subgoal 2"),
            ]

            created = graph.decompose_goal(parent.id, subgoals)
            assert len(created) == 2

            # Check parent has subgoals
            parent = graph.get_goal(parent.id)
            assert len(parent.subgoal_ids) == 2

    def test_get_root_goals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir))

            root1 = Goal(name="Root 1")
            root2 = Goal(name="Root 2")
            graph.add_goal(root1)
            graph.add_goal(root2)

            # Add child to root1
            child = Goal(name="Child", parent_id=root1.id)
            graph.add_goal(child)

            roots = graph.get_root_goals()
            assert len(roots) == 2

    def test_get_actionable_goals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir))

            # Add goals
            actionable = Goal(name="Actionable", priority=GoalPriority.HIGH)
            blocked = Goal(name="Blocked", blocked_by=["reason"])
            with_dep = Goal(name="Has Dependency", depends_on=["nonexistent"])

            graph.add_goal(actionable)
            graph.add_goal(blocked)
            graph.add_goal(with_dep)

            result = graph.get_actionable_goals()
            assert len(result) == 1
            assert result[0].name == "Actionable"

    def test_update_status_propagates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir))

            parent = Goal(name="Parent")
            graph.add_goal(parent)

            subgoals = [Goal(name="Sub1"), Goal(name="Sub2")]
            created = graph.decompose_goal(parent.id, subgoals)

            # Complete both subgoals
            graph.update_status(created[0].id, GoalStatus.COMPLETED)
            graph.update_status(created[1].id, GoalStatus.COMPLETED)

            # Parent should be complete
            parent = graph.get_goal(parent.id)
            assert parent.status == GoalStatus.COMPLETED

    def test_compute_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir))

            parent = Goal(name="Parent")
            graph.add_goal(parent)

            subgoals = [
                Goal(name="Sub1", estimated_effort=1.0),
                Goal(name="Sub2", estimated_effort=1.0),
            ]
            created = graph.decompose_goal(parent.id, subgoals)

            # Complete one subgoal
            graph.update_status(created[0].id, GoalStatus.COMPLETED)

            completion = graph.compute_completion(parent.id)
            assert completion == 50.0

    def test_get_critical_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir))

            goal = Goal(name="Main Goal")
            dep = Goal(name="Dependency")
            graph.add_goal(dep)
            goal.depends_on = [dep.id]
            graph.add_goal(goal)

            path = graph.get_critical_path(goal.id)
            assert len(path) == 2
            assert path[0].name == "Dependency"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            graph1 = GoalGraph(Path(tmpdir))
            graph1.add_goal(Goal(id="g1", name="Persistent Goal"))

            # Load in new instance
            graph2 = GoalGraph(Path(tmpdir))
            goal = graph2.get_goal("g1")
            assert goal is not None
            assert goal.name == "Persistent Goal"


# =============================================================================
# Plan Tests
# =============================================================================

class TestPlan:
    def test_create_plan(self):
        plan = Plan(name="Test Plan", steps=[
            PlanStep(action="step1", description="First step"),
            PlanStep(action="step2", description="Second step"),
        ])
        assert len(plan.steps) == 2
        assert plan.status == PlanStatus.DRAFT

    def test_get_next_steps(self):
        step1 = PlanStep(action="step1")
        step2 = PlanStep(action="step2", dependencies=[step1.id])

        plan = Plan(name="Test", steps=[step1, step2])

        next_steps = plan.get_next_steps()
        assert len(next_steps) == 1
        assert next_steps[0].action == "step1"

    def test_mark_step_complete(self):
        step1 = PlanStep(action="step1")
        step2 = PlanStep(action="step2")

        plan = Plan(name="Test", steps=[step1, step2])

        plan.mark_step_complete(step1.id)
        assert plan.steps[0].completed is True

        plan.mark_step_complete(step2.id)
        assert plan.status == PlanStatus.COMPLETED


# =============================================================================
# Planner Tests
# =============================================================================

class TestPlanner:
    def test_create_plan_forward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir) / "goals")
            planner = Planner(Path(tmpdir) / "plans", graph)

            goal = Goal(name="Test Goal")
            graph.add_goal(goal)

            plan = planner.create_plan(goal.id, PlanningStrategy.FORWARD)
            assert plan is not None
            assert len(plan.steps) >= 1

    def test_create_plan_for_goal_with_subgoals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir) / "goals")
            planner = Planner(Path(tmpdir) / "plans", graph)

            parent = Goal(name="Parent")
            graph.add_goal(parent)
            graph.decompose_goal(parent.id, [Goal(name="Sub1"), Goal(name="Sub2")])

            plan = planner.create_plan(parent.id, PlanningStrategy.FORWARD)
            assert plan is not None
            assert len(plan.steps) == 2  # One for each subgoal

    def test_create_plan_backward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir) / "goals")
            planner = Planner(Path(tmpdir) / "plans", graph)

            dep = Goal(name="Dependency")
            graph.add_goal(dep)

            goal = Goal(name="Goal", depends_on=[dep.id])
            graph.add_goal(goal)

            plan = planner.create_plan(goal.id, PlanningStrategy.BACKWARD)
            assert plan is not None
            # Should have step for dependency + step for goal
            assert len(plan.steps) >= 2

    def test_get_plans_for_goal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir) / "goals")
            planner = Planner(Path(tmpdir) / "plans", graph)

            goal = Goal(name="Test")
            graph.add_goal(goal)

            planner.create_plan(goal.id)
            planner.create_plan(goal.id)

            plans = planner.get_plans_for_goal(goal.id)
            assert len(plans) == 2

    def test_refine_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir) / "goals")
            planner = Planner(Path(tmpdir) / "plans", graph)

            goal = Goal(name="Test")
            graph.add_goal(goal)

            plan = planner.create_plan(goal.id)
            original_steps = len(plan.steps)

            refined = planner.refine_plan(plan.id, {
                "add_steps": [{"action": "new_step", "description": "A new step"}]
            })

            assert len(refined.steps) == original_steps + 1


# =============================================================================
# PlanEvaluator Tests
# =============================================================================

class TestPlanEvaluator:
    def test_evaluate_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir))
            evaluator = PlanEvaluator(graph)

            plan = Plan(
                name="Test Plan",
                confidence=0.8,
                steps=[
                    PlanStep(action="step1"),
                    PlanStep(action="step2"),
                ],
            )

            evaluation = evaluator.evaluate_plan(plan)
            assert "scores" in evaluation
            assert "overall" in evaluation
            assert "recommendation" in evaluation

    def test_compare_plans(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir))
            evaluator = PlanEvaluator(graph)

            # Simple plan (should rank higher)
            simple = Plan(name="Simple", confidence=0.9, steps=[PlanStep(action="one_step")])

            # Complex plan
            complex_plan = Plan(
                name="Complex",
                confidence=0.5,
                steps=[PlanStep(action=f"step_{i}") for i in range(10)],
            )

            ranked = evaluator.compare_plans([complex_plan, simple])

            # Simple should be ranked first
            assert ranked[0][0].name == "Simple"


# =============================================================================
# StrategicReasoner Tests
# =============================================================================

class TestStrategicReasoner:
    def test_analyze_situation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir) / "goals")
            planner = Planner(Path(tmpdir) / "plans", graph)
            reasoner = StrategicReasoner(Path(tmpdir) / "strategy", graph, planner)

            # Add some goals
            graph.add_goal(Goal(name="Goal 1"))
            graph.add_goal(Goal(name="Goal 2", blocked_by=["blocker"]))

            analysis = reasoner.analyze_situation()
            assert "goal_overview" in analysis
            assert "actionable_goals" in analysis
            assert "blocked_goals" in analysis

    def test_identify_opportunities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir) / "goals")
            planner = Planner(Path(tmpdir) / "plans", graph)
            reasoner = StrategicReasoner(Path(tmpdir) / "strategy", graph, planner)

            # Add multiple actionable goals
            graph.add_goal(Goal(name="Goal 1"))
            graph.add_goal(Goal(name="Goal 2"))

            opportunities = reasoner.identify_opportunities()
            assert len(opportunities) >= 1

    def test_identify_risks_deadline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir) / "goals")
            planner = Planner(Path(tmpdir) / "plans", graph)
            reasoner = StrategicReasoner(Path(tmpdir) / "strategy", graph, planner)

            # Add goal with past deadline
            graph.add_goal(Goal(
                name="Overdue Goal",
                deadline=datetime.now(UTC) - timedelta(days=1),
            ))

            risks = reasoner.identify_risks()
            deadline_risks = [r for r in risks if "deadline" in r.description.lower()]
            assert len(deadline_risks) >= 1

    def test_recommend_focus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir) / "goals")
            planner = Planner(Path(tmpdir) / "plans", graph)
            reasoner = StrategicReasoner(Path(tmpdir) / "strategy", graph, planner)

            # Add actionable goal
            graph.add_goal(Goal(name="Focus Here", priority=GoalPriority.HIGH))

            recommendation = reasoner.recommend_focus()
            assert "Focus Here" in recommendation["recommendation"]

    def test_suggest_goal_decomposition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GoalGraph(Path(tmpdir) / "goals")
            planner = Planner(Path(tmpdir) / "plans", graph)
            reasoner = StrategicReasoner(Path(tmpdir) / "strategy", graph, planner)

            goal = Goal(name="Implement feature X")
            graph.add_goal(goal)

            suggestions = reasoner.suggest_goal_decomposition(goal.id)
            assert len(suggestions) >= 2


# =============================================================================
# StrategicPlanningSystem Integration Tests
# =============================================================================

class TestStrategicPlanningSystem:
    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = StrategicPlanningSystem(Path(tmpdir))

            assert system.goals is not None
            assert system.planner is not None
            assert system.evaluator is not None
            assert system.strategist is not None

    def test_add_and_decompose_goal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = StrategicPlanningSystem(Path(tmpdir))

            goal = system.add_goal("Main Goal", priority=GoalPriority.HIGH)
            subgoals = system.decompose_goal(goal.id, ["Step 1", "Step 2", "Step 3"])

            assert len(subgoals) == 3
            main = system.goals.get_goal(goal.id)
            assert len(main.subgoal_ids) == 3

    def test_create_and_evaluate_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = StrategicPlanningSystem(Path(tmpdir))

            goal = system.add_goal("Test Goal")
            plan = system.create_plan(goal.id)

            assert plan is not None

            evaluations = system.evaluate_plans(goal.id)
            assert len(evaluations) == 1

    def test_strategic_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = StrategicPlanningSystem(Path(tmpdir))

            system.add_goal("Goal 1")
            system.add_goal("Goal 2", priority=GoalPriority.HIGH)

            analysis = system.get_strategic_analysis()

            assert "situation" in analysis
            assert "opportunities" in analysis
            assert "risks" in analysis
            assert "recommended_focus" in analysis

    def test_complete_goal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = StrategicPlanningSystem(Path(tmpdir))

            goal = system.add_goal("Completable Goal")
            system.complete_goal(goal.id)

            updated = system.goals.get_goal(goal.id)
            assert updated.status == GoalStatus.COMPLETED

    def test_block_goal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = StrategicPlanningSystem(Path(tmpdir))

            goal = system.add_goal("Blockable Goal")
            system.block_goal(goal.id, "Waiting for input")

            updated = system.goals.get_goal(goal.id)
            assert updated.status == GoalStatus.BLOCKED
            assert "Waiting for input" in updated.blocked_by

    def test_get_next_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = StrategicPlanningSystem(Path(tmpdir))

            goal = system.add_goal("Action Goal")
            system.create_plan(goal.id)

            actions = system.get_next_actions()
            assert len(actions) >= 1
            assert actions[0]["goal"] == "Action Goal"

    def test_full_workflow(self):
        """Test complete strategic planning workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = StrategicPlanningSystem(Path(tmpdir))

            # 1. Add high-level goal
            main_goal = system.add_goal(
                "Implement new feature",
                description="Add user authentication",
                priority=GoalPriority.HIGH,
                success_criteria=["Users can log in", "Users can log out"],
            )

            # 2. Decompose into subgoals
            subgoals = system.decompose_goal(main_goal.id, [
                "Design authentication flow",
                "Implement login endpoint",
                "Implement logout endpoint",
                "Write tests",
            ])

            # 3. Create plan
            plan = system.create_plan(main_goal.id)
            assert plan is not None

            # 4. Get strategic analysis
            analysis = system.get_strategic_analysis()
            assert len(analysis["opportunities"]) >= 0

            # 5. Get next actions
            actions = system.get_next_actions()
            assert len(actions) >= 1

            # 6. Complete subgoals
            for subgoal in subgoals:
                system.complete_goal(subgoal.id)

            # 7. Main goal should be complete
            main = system.goals.get_goal(main_goal.id)
            assert main.status == GoalStatus.COMPLETED
