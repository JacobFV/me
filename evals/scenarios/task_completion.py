"""
Task completion eval scenarios.

Measures: Can the agent achieve specified goals?

Key metrics:
- completion: Did the agent achieve the goal? (binary)
- steps: How many steps did it take? (count)
- efficiency: Goal achieved / resources used (ratio)

Interpretation notes:
- Completion alone is insufficient - a brute force agent could complete
  tasks inefficiently. Always consider efficiency.
- Some tasks have multiple valid solutions. The eval should accept
  any correct solution, not just the expected one.
- Partial completion may be valuable - track progress, not just success.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from evals.framework import (
    Scenario,
    EvalTrace,
    Metric,
    MetricType,
)


class TaskCompletionScenario(Scenario):
    """
    Base class for task completion scenarios.

    Subclass this to create specific task evaluations.
    """

    def __init__(
        self,
        scenario_id: str,
        name: str,
        description: str,
        task_prompt: str,
        success_criteria: list[str],
        max_steps: int = 20,
        tags: list[str] | None = None,
    ):
        super().__init__(scenario_id, name, description, tags)
        self.task_prompt = task_prompt
        self.success_criteria = success_criteria
        self.max_steps = max_steps

    async def setup(self, agent: Any) -> dict[str, Any]:
        """Default setup - create temp directory for work."""
        work_dir = tempfile.mkdtemp(prefix="eval_")
        return {
            "work_dir": Path(work_dir),
            "steps": 0,
        }

    async def run(self, agent: Any, context: dict[str, Any]) -> EvalTrace:
        """Run agent on task."""
        trace = EvalTrace()

        # Send task prompt
        trace.add_message("user", self.task_prompt)

        try:
            async for message in agent.run(self.task_prompt):
                role = message.get("role", "unknown")
                content = message.get("content", "")

                if role == "assistant":
                    trace.add_message("assistant", content)
                    context["steps"] += 1

                elif role == "tool":
                    trace.add_tool_call(
                        message.get("name", "unknown"),
                        message.get("input", {}),
                        message.get("result", ""),
                    )

                # Enforce step limit
                if context["steps"] >= self.max_steps:
                    trace.errors.append(f"Exceeded max steps: {self.max_steps}")
                    break

        except Exception as e:
            trace.errors.append(str(e))

        return trace

    async def evaluate(
        self,
        agent: Any,
        context: dict[str, Any],
        trace: EvalTrace,
    ) -> tuple[bool, list[Metric]]:
        """
        Evaluate task completion.

        Override check_criteria() in subclasses for specific checks.
        """
        metrics = []

        # Count steps
        steps = context.get("steps", 0)
        metrics.append(Metric(
            name="steps",
            value=steps,
            metric_type=MetricType.COUNT,
        ))

        # Count tool calls
        tool_calls = len(trace.tool_calls)
        metrics.append(Metric(
            name="tool_calls",
            value=tool_calls,
            metric_type=MetricType.COUNT,
        ))

        # Check success criteria
        criteria_met = await self.check_criteria(agent, context, trace)
        success = all(criteria_met.values())

        # Track which criteria passed
        criteria_passed = sum(1 for v in criteria_met.values() if v)
        metrics.append(Metric(
            name="criteria_passed",
            value=criteria_passed,
            metric_type=MetricType.COUNT,
            metadata={"total": len(criteria_met), "details": criteria_met},
        ))

        # Efficiency: criteria passed per step (higher is better)
        if steps > 0:
            efficiency = criteria_passed / steps
        else:
            efficiency = 0.0
        metrics.append(Metric(
            name="efficiency",
            value=efficiency,
            metric_type=MetricType.RATIO,
        ))

        # Check for errors
        if trace.errors:
            metrics.append(Metric(
                name="error_count",
                value=len(trace.errors),
                metric_type=MetricType.COUNT,
            ))

        return success, metrics

    async def check_criteria(
        self,
        agent: Any,
        context: dict[str, Any],
        trace: EvalTrace,
    ) -> dict[str, bool]:
        """
        Check if success criteria are met.

        Override in subclasses for specific checks.
        Returns dict mapping criterion name to whether it passed.
        """
        # Default: just check if agent responded without errors
        return {
            "no_errors": len(trace.errors) == 0,
            "responded": len(trace.messages) > 1,
        }

    async def teardown(self, agent: Any, context: dict[str, Any]):
        """Clean up temp directory."""
        import shutil
        work_dir = context.get("work_dir")
        if work_dir and work_dir.exists():
            shutil.rmtree(work_dir)


class FileCreationScenario(TaskCompletionScenario):
    """Scenario: Create a file with specific content."""

    def __init__(
        self,
        scenario_id: str = "file_creation",
        filename: str = "test.txt",
        expected_content: str = "Hello, World!",
    ):
        self.filename = filename
        self.expected_content = expected_content

        super().__init__(
            scenario_id=scenario_id,
            name="File Creation",
            description=f"Create file '{filename}' with specific content",
            task_prompt=f"Create a file named '{filename}' in your working directory with the content: {expected_content}",
            success_criteria=["file_exists", "content_matches"],
            tags=["basic", "file_ops"],
        )

    async def check_criteria(
        self,
        agent: Any,
        context: dict[str, Any],
        trace: EvalTrace,
    ) -> dict[str, bool]:
        work_dir = context.get("work_dir")
        if not work_dir:
            return {"file_exists": False, "content_matches": False}

        file_path = work_dir / self.filename
        file_exists = file_path.exists()

        content_matches = False
        if file_exists:
            content = file_path.read_text()
            content_matches = self.expected_content in content

        return {
            "file_exists": file_exists,
            "content_matches": content_matches,
        }


class MultiStepTaskScenario(TaskCompletionScenario):
    """Scenario: Complete a task requiring multiple steps."""

    def __init__(
        self,
        scenario_id: str = "multi_step",
        steps_description: list[str] | None = None,
    ):
        self.steps_description = steps_description or [
            "Create a directory called 'project'",
            "Create a file 'project/README.md'",
            "Write a brief description in the README",
        ]

        super().__init__(
            scenario_id=scenario_id,
            name="Multi-Step Task",
            description="Complete a task requiring multiple sequential steps",
            task_prompt="Please complete these steps:\n" + "\n".join(
                f"{i+1}. {step}" for i, step in enumerate(self.steps_description)
            ),
            success_criteria=["all_steps_completed"],
            tags=["multi_step", "planning"],
        )

    async def check_criteria(
        self,
        agent: Any,
        context: dict[str, Any],
        trace: EvalTrace,
    ) -> dict[str, bool]:
        work_dir = context.get("work_dir")
        if not work_dir:
            return {"all_steps_completed": False}

        # Check expected artifacts
        project_dir = work_dir / "project"
        readme = project_dir / "README.md"

        return {
            "directory_created": project_dir.is_dir(),
            "readme_created": readme.exists(),
            "readme_has_content": readme.exists() and len(readme.read_text().strip()) > 0,
            "all_steps_completed": (
                project_dir.is_dir() and
                readme.exists() and
                len(readme.read_text().strip()) > 0
            ),
        }


class GoalDecompositionScenario(TaskCompletionScenario):
    """Scenario: Decompose a complex goal into subtasks."""

    def __init__(self, scenario_id: str = "goal_decomposition"):
        super().__init__(
            scenario_id=scenario_id,
            name="Goal Decomposition",
            description="Break down a complex goal and execute subtasks",
            task_prompt=(
                "Your goal is to set up a simple Python project. "
                "First, analyze what needs to be done, break it into steps, "
                "then execute each step. Create: 1) a project folder, "
                "2) a main.py with a hello world function, "
                "3) a requirements.txt (can be empty)."
            ),
            success_criteria=["planning", "execution", "artifacts"],
            max_steps=30,
            tags=["planning", "decomposition", "complex"],
        )

    async def check_criteria(
        self,
        agent: Any,
        context: dict[str, Any],
        trace: EvalTrace,
    ) -> dict[str, bool]:
        work_dir = context.get("work_dir")

        # Check for planning behavior (mentions of steps/plan in response)
        planning = False
        for msg in trace.messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                if any(word in content for word in ["step", "first", "then", "plan", "1.", "2."]):
                    planning = True
                    break

        # Check artifacts
        artifacts_ok = False
        if work_dir:
            # Look for main.py anywhere in work_dir
            main_files = list(work_dir.rglob("main.py"))
            req_files = list(work_dir.rglob("requirements.txt"))
            artifacts_ok = len(main_files) > 0 and len(req_files) > 0

        return {
            "planning": planning,
            "execution": len(trace.tool_calls) > 0,
            "artifacts": artifacts_ok,
        }
