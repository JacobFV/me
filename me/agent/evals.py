"""
Agent Evaluation Framework - Metrics-driven development.

This module provides infrastructure for:
1. Defining measurable metrics for agent capabilities
2. Running standardized evaluation tasks
3. Tracking metrics over time
4. Comparing before/after for changes
5. Detecting regressions

Philosophy: Without measurement, improvement is just wishful thinking.
Every capability should be testable, and every change should show
measurable impact. This framework enables metrics-driven development.

Usage:
    evaluator = AgentEvaluator(body_dir)

    # Run all evals
    report = evaluator.run_full_evaluation()

    # Run specific eval
    result = evaluator.run_eval("coding_simple")

    # Compare to baseline
    comparison = evaluator.compare_to_baseline()

    # Track over time
    history = evaluator.get_metrics_history()
"""

from __future__ import annotations

from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import json
import hashlib
import statistics

from pydantic import BaseModel, Field


class MetricCategory(str, Enum):
    """Categories of metrics we track."""
    TASK_COMPLETION = "task_completion"  # Did it finish?
    QUALITY = "quality"  # How good was the output?
    EFFICIENCY = "efficiency"  # Resources used
    LEARNING = "learning"  # Improvement over time
    CALIBRATION = "calibration"  # Accuracy of predictions
    ROBUSTNESS = "robustness"  # Handling edge cases
    MEMORY = "memory"  # Recall and retention


class EvalDifficulty(str, Enum):
    """Difficulty levels for evaluation tasks."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"


class MetricValue(BaseModel):
    """A single metric measurement."""
    name: str
    category: MetricCategory
    value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: dict[str, Any] = Field(default_factory=dict)


class EvalTask(BaseModel):
    """Definition of an evaluation task."""
    id: str
    name: str
    description: str
    category: MetricCategory
    difficulty: EvalDifficulty
    # Task specification
    input_data: dict[str, Any] = Field(default_factory=dict)
    expected_output: dict[str, Any] | None = None
    validation_fn: str | None = None  # Name of validation function
    # Scoring
    max_score: float = 1.0
    time_limit_seconds: float | None = None
    token_budget: int | None = None


class EvalResult(BaseModel):
    """Result from running an evaluation task."""
    task_id: str
    task_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Outcomes
    completed: bool
    score: float  # 0-1 normalized
    raw_score: float
    max_score: float
    # Resource usage
    time_taken_seconds: float
    tokens_used: int = 0
    # Details
    output: Any = None
    errors: list[str] = Field(default_factory=list)
    notes: str = ""


class EvalReport(BaseModel):
    """Report from a full evaluation run."""
    id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Aggregate metrics
    overall_score: float
    completion_rate: float
    average_quality: float
    average_efficiency: float
    # Per-category scores
    category_scores: dict[MetricCategory, float] = Field(default_factory=dict)
    # Per-difficulty scores
    difficulty_scores: dict[EvalDifficulty, float] = Field(default_factory=dict)
    # Individual results
    results: list[EvalResult] = Field(default_factory=list)
    # Comparison to baseline
    baseline_comparison: dict[str, Any] | None = None
    # Metadata
    agent_version: str = ""
    config_hash: str = ""


class BaselineMetrics(BaseModel):
    """Baseline metrics for comparison."""
    id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    overall_score: float
    category_scores: dict[str, float] = Field(default_factory=dict)
    difficulty_scores: dict[str, float] = Field(default_factory=dict)
    task_scores: dict[str, float] = Field(default_factory=dict)


class MetricsHistory(BaseModel):
    """Historical metrics over time."""
    metric_name: str
    data_points: list[tuple[datetime, float]] = Field(default_factory=list)
    trend: str = "stable"  # improving, declining, stable
    trend_strength: float = 0.0


# =============================================================================
# Evaluation Tasks
# =============================================================================

# These are the standardized evaluation tasks
EVAL_TASKS: list[EvalTask] = [
    # Task Completion - Coding
    EvalTask(
        id="coding_trivial_hello",
        name="Hello World",
        description="Write a function that returns 'Hello, World!'",
        category=MetricCategory.TASK_COMPLETION,
        difficulty=EvalDifficulty.TRIVIAL,
        input_data={"language": "python"},
        expected_output={"contains": "Hello, World!"},
    ),
    EvalTask(
        id="coding_simple_fizzbuzz",
        name="FizzBuzz",
        description="Implement FizzBuzz for numbers 1-100",
        category=MetricCategory.TASK_COMPLETION,
        difficulty=EvalDifficulty.SIMPLE,
        input_data={"language": "python"},
        validation_fn="validate_fizzbuzz",
    ),
    EvalTask(
        id="coding_medium_parser",
        name="JSON Parser",
        description="Implement a simple JSON parser for objects and arrays",
        category=MetricCategory.TASK_COMPLETION,
        difficulty=EvalDifficulty.MEDIUM,
        input_data={"language": "python"},
        validation_fn="validate_json_parser",
        time_limit_seconds=300,
    ),
    EvalTask(
        id="coding_complex_scheduler",
        name="Task Scheduler",
        description="Implement a priority task scheduler with dependencies",
        category=MetricCategory.TASK_COMPLETION,
        difficulty=EvalDifficulty.COMPLEX,
        input_data={"language": "python"},
        validation_fn="validate_scheduler",
        time_limit_seconds=600,
    ),

    # Quality - Code Review
    EvalTask(
        id="quality_review_bugs",
        name="Bug Detection",
        description="Find bugs in provided code samples",
        category=MetricCategory.QUALITY,
        difficulty=EvalDifficulty.MEDIUM,
        input_data={
            "code_samples": [
                {"code": "def divide(a, b): return a / b", "bugs": ["no zero check"]},
                {"code": "users = []; users[0]", "bugs": ["index error"]},
            ]
        },
        validation_fn="validate_bug_detection",
    ),
    EvalTask(
        id="quality_explain_code",
        name="Code Explanation",
        description="Explain what complex code does",
        category=MetricCategory.QUALITY,
        difficulty=EvalDifficulty.MEDIUM,
        input_data={
            "code": "def f(n): return n if n < 2 else f(n-1) + f(n-2)",
        },
        expected_output={"contains_concepts": ["fibonacci", "recursive"]},
    ),

    # Efficiency
    EvalTask(
        id="efficiency_minimal_solution",
        name="Minimal Solution",
        description="Solve problem with minimal code",
        category=MetricCategory.EFFICIENCY,
        difficulty=EvalDifficulty.SIMPLE,
        input_data={"problem": "reverse a string"},
        validation_fn="validate_minimal_solution",
        token_budget=500,
    ),

    # Memory / Recall
    EvalTask(
        id="memory_context_recall",
        name="Context Recall",
        description="Recall information from earlier in conversation",
        category=MetricCategory.MEMORY,
        difficulty=EvalDifficulty.SIMPLE,
        input_data={
            "context": "The user's name is Alice and they prefer tabs over spaces.",
            "questions": ["What is the user's name?", "Tabs or spaces?"],
        },
        expected_output={"answers": ["Alice", "tabs"]},
    ),
    EvalTask(
        id="memory_procedure_recall",
        name="Procedure Recall",
        description="Recall a previously learned procedure",
        category=MetricCategory.MEMORY,
        difficulty=EvalDifficulty.MEDIUM,
        input_data={
            "procedure": "To deploy: 1) run tests, 2) build, 3) push to registry, 4) update k8s",
        },
        validation_fn="validate_procedure_recall",
    ),

    # Calibration
    EvalTask(
        id="calibration_confidence",
        name="Confidence Calibration",
        description="Assess confidence accuracy",
        category=MetricCategory.CALIBRATION,
        difficulty=EvalDifficulty.MEDIUM,
        input_data={
            "questions": [
                {"q": "Is Python dynamically typed?", "a": True},
                {"q": "Is Rust garbage collected?", "a": False},
                {"q": "Does JavaScript have classes?", "a": True},
            ]
        },
        validation_fn="validate_calibration",
    ),

    # Robustness
    EvalTask(
        id="robustness_malformed_input",
        name="Malformed Input Handling",
        description="Handle malformed or adversarial inputs gracefully",
        category=MetricCategory.ROBUSTNESS,
        difficulty=EvalDifficulty.MEDIUM,
        input_data={
            "inputs": [
                "",  # Empty
                "x" * 10000,  # Very long
                "\x00\x01\x02",  # Binary
                "'; DROP TABLE users; --",  # SQL injection attempt
            ]
        },
        validation_fn="validate_robustness",
    ),

    # Learning
    EvalTask(
        id="learning_from_feedback",
        name="Learning from Feedback",
        description="Improve based on correction",
        category=MetricCategory.LEARNING,
        difficulty=EvalDifficulty.MEDIUM,
        input_data={
            "initial_task": "Write a greeting function",
            "feedback": "Use f-strings instead of concatenation",
            "retry_task": "Write another greeting function",
        },
        validation_fn="validate_learning",
    ),
]


# =============================================================================
# Validators
# =============================================================================

class EvalValidators:
    """Validation functions for evaluation tasks."""

    @staticmethod
    def validate_fizzbuzz(output: str) -> tuple[float, str]:
        """Validate FizzBuzz implementation."""
        # Check for key outputs
        checks = [
            ("Fizz" in output, "Contains Fizz"),
            ("Buzz" in output, "Contains Buzz"),
            ("FizzBuzz" in output, "Contains FizzBuzz"),
            ("1" in output, "Contains numbers"),
        ]
        passed = sum(1 for check, _ in checks if check)
        notes = ", ".join(note for check, note in checks if check)
        return passed / len(checks), notes

    @staticmethod
    def validate_json_parser(output: str) -> tuple[float, str]:
        """Validate JSON parser implementation."""
        # Check for key components
        checks = [
            ("def" in output.lower() or "function" in output.lower(), "Has function"),
            ("parse" in output.lower(), "Has parse logic"),
            ("{" in output and "}" in output, "Handles objects"),
            ("[" in output and "]" in output, "Handles arrays"),
        ]
        passed = sum(1 for check, _ in checks if check)
        return passed / len(checks), f"Passed {passed}/{len(checks)} checks"

    @staticmethod
    def validate_scheduler(output: str) -> tuple[float, str]:
        """Validate task scheduler implementation."""
        checks = [
            ("priority" in output.lower(), "Has priority handling"),
            ("depend" in output.lower(), "Has dependency handling"),
            ("queue" in output.lower() or "heap" in output.lower(), "Has queue structure"),
            ("def" in output.lower() or "class" in output.lower(), "Has implementation"),
        ]
        passed = sum(1 for check, _ in checks if check)
        return passed / len(checks), f"Passed {passed}/{len(checks)} checks"

    @staticmethod
    def validate_bug_detection(output: str, expected: dict) -> tuple[float, str]:
        """Validate bug detection results."""
        bugs_found = 0
        total_bugs = len(expected.get("code_samples", []))

        for sample in expected.get("code_samples", []):
            for bug in sample.get("bugs", []):
                if bug.lower() in output.lower():
                    bugs_found += 1
                    break

        return bugs_found / max(1, total_bugs), f"Found {bugs_found}/{total_bugs} bugs"

    @staticmethod
    def validate_minimal_solution(output: str) -> tuple[float, str]:
        """Validate solution minimality."""
        # Count lines and characters
        lines = len([l for l in output.split("\n") if l.strip()])
        chars = len(output)

        # Score inversely proportional to length
        # Optimal: 1 line, ~30 chars
        line_score = max(0, 1 - (lines - 1) * 0.2)
        char_score = max(0, 1 - (chars - 30) / 200)

        score = (line_score + char_score) / 2
        return score, f"{lines} lines, {chars} chars"

    @staticmethod
    def validate_procedure_recall(output: str, expected: dict) -> tuple[float, str]:
        """Validate procedure recall accuracy."""
        procedure = expected.get("procedure", "")
        steps = ["test", "build", "push", "k8s"]

        found = sum(1 for step in steps if step.lower() in output.lower())
        return found / len(steps), f"Recalled {found}/{len(steps)} steps"

    @staticmethod
    def validate_calibration(output: str, expected: dict) -> tuple[float, str]:
        """Validate confidence calibration."""
        # This is a simplified check - real calibration needs many samples
        questions = expected.get("questions", [])
        correct = 0

        for q in questions:
            answer = q.get("a", False)
            # Check if output contains correct answer
            if answer and ("yes" in output.lower() or "true" in output.lower()):
                correct += 1
            elif not answer and ("no" in output.lower() or "false" in output.lower()):
                correct += 1

        return correct / max(1, len(questions)), f"{correct}/{len(questions)} correct"

    @staticmethod
    def validate_robustness(output: str, expected: dict) -> tuple[float, str]:
        """Validate robustness to malformed input."""
        # Check that agent didn't crash or produce garbage
        checks = [
            (len(output) > 0, "Produced output"),
            ("error" not in output.lower() or "handled" in output.lower(), "Handled gracefully"),
            (len(output) < 10000, "Reasonable length"),
        ]
        passed = sum(1 for check, _ in checks if check)
        return passed / len(checks), f"Passed {passed}/{len(checks)} robustness checks"

    @staticmethod
    def validate_learning(output: str, expected: dict) -> tuple[float, str]:
        """Validate learning from feedback."""
        feedback = expected.get("feedback", "")

        # Check if feedback was incorporated
        if "f-string" in feedback.lower() or "f'" in feedback:
            learned = "f'" in output or 'f"' in output
        else:
            learned = True  # Can't verify

        return 1.0 if learned else 0.0, "Incorporated feedback" if learned else "Did not learn"


# =============================================================================
# Evaluator
# =============================================================================

class TaskRunner:
    """
    Runs individual evaluation tasks.

    In a real implementation, this would invoke the actual agent.
    For now, it provides the infrastructure for plugging in agent execution.
    """

    def __init__(self):
        self.validators = EvalValidators()

    def run_task(
        self,
        task: EvalTask,
        agent_fn: Callable[[dict], tuple[str, int]] | None = None,
    ) -> EvalResult:
        """
        Run an evaluation task.

        Args:
            task: The evaluation task to run
            agent_fn: Function that takes input_data and returns (output, tokens_used)
                     If None, returns a placeholder result
        """
        import time
        start_time = time.time()

        try:
            if agent_fn:
                output, tokens_used = agent_fn(task.input_data)
            else:
                # Placeholder for when agent isn't connected
                output = "[Agent not connected - placeholder result]"
                tokens_used = 0

            time_taken = time.time() - start_time

            # Validate output
            score, notes = self._validate(task, output)

            # Check time limit
            if task.time_limit_seconds and time_taken > task.time_limit_seconds:
                score *= 0.5  # Penalty for exceeding time
                notes += f" [TIMEOUT: {time_taken:.1f}s > {task.time_limit_seconds}s]"

            # Check token budget
            if task.token_budget and tokens_used > task.token_budget:
                score *= 0.8  # Penalty for exceeding budget
                notes += f" [OVER BUDGET: {tokens_used} > {task.token_budget}]"

            return EvalResult(
                task_id=task.id,
                task_name=task.name,
                completed=True,
                score=score,
                raw_score=score * task.max_score,
                max_score=task.max_score,
                time_taken_seconds=time_taken,
                tokens_used=tokens_used,
                output=output,
                notes=notes,
            )

        except Exception as e:
            time_taken = time.time() - start_time
            return EvalResult(
                task_id=task.id,
                task_name=task.name,
                completed=False,
                score=0.0,
                raw_score=0.0,
                max_score=task.max_score,
                time_taken_seconds=time_taken,
                errors=[str(e)],
            )

    def _validate(self, task: EvalTask, output: str) -> tuple[float, str]:
        """Validate task output."""
        # Check expected_output first
        if task.expected_output:
            if "contains" in task.expected_output:
                expected = task.expected_output["contains"]
                if expected.lower() in output.lower():
                    return 1.0, f"Contains '{expected}'"
                return 0.0, f"Missing '{expected}'"

            if "contains_concepts" in task.expected_output:
                concepts = task.expected_output["contains_concepts"]
                found = sum(1 for c in concepts if c.lower() in output.lower())
                return found / len(concepts), f"Found {found}/{len(concepts)} concepts"

            if "answers" in task.expected_output:
                answers = task.expected_output["answers"]
                found = sum(1 for a in answers if a.lower() in output.lower())
                return found / len(answers), f"Correct {found}/{len(answers)}"

        # Use validation function
        if task.validation_fn:
            validator = getattr(self.validators, task.validation_fn, None)
            if validator:
                if task.input_data:
                    return validator(output, task.input_data)
                return validator(output)

        # Default: check if non-empty
        return 1.0 if output.strip() else 0.0, "Default validation"


class AgentEvaluator:
    """
    Main evaluator for running agent evaluations.

    Provides:
    - Running individual or full evaluation suites
    - Baseline comparison
    - Metrics history tracking
    - Report generation
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.evals_dir = body_dir / "evals"
        self.evals_dir.mkdir(parents=True, exist_ok=True)

        self.runner = TaskRunner()
        self.tasks = {t.id: t for t in EVAL_TASKS}

    def run_eval(
        self,
        task_id: str,
        agent_fn: Callable[[dict], tuple[str, int]] | None = None,
    ) -> EvalResult:
        """Run a single evaluation task."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")

        result = self.runner.run_task(task, agent_fn)
        self._save_result(result)
        return result

    def run_category(
        self,
        category: MetricCategory,
        agent_fn: Callable[[dict], tuple[str, int]] | None = None,
    ) -> list[EvalResult]:
        """Run all tasks in a category."""
        results = []
        for task in EVAL_TASKS:
            if task.category == category:
                result = self.runner.run_task(task, agent_fn)
                results.append(result)
                self._save_result(result)
        return results

    def run_difficulty(
        self,
        difficulty: EvalDifficulty,
        agent_fn: Callable[[dict], tuple[str, int]] | None = None,
    ) -> list[EvalResult]:
        """Run all tasks at a difficulty level."""
        results = []
        for task in EVAL_TASKS:
            if task.difficulty == difficulty:
                result = self.runner.run_task(task, agent_fn)
                results.append(result)
                self._save_result(result)
        return results

    def run_full_evaluation(
        self,
        agent_fn: Callable[[dict], tuple[str, int]] | None = None,
        agent_version: str = "",
    ) -> EvalReport:
        """Run all evaluation tasks."""
        results = []
        for task in EVAL_TASKS:
            result = self.runner.run_task(task, agent_fn)
            results.append(result)
            self._save_result(result)

        report = self._generate_report(results, agent_version)
        self._save_report(report)

        return report

    def _generate_report(
        self,
        results: list[EvalResult],
        agent_version: str = "",
    ) -> EvalReport:
        """Generate evaluation report from results."""
        report_id = hashlib.md5(
            f"{datetime.now(UTC).isoformat()}".encode()
        ).hexdigest()[:12]

        # Aggregate metrics
        scores = [r.score for r in results]
        overall_score = statistics.mean(scores) if scores else 0.0
        completion_rate = sum(1 for r in results if r.completed) / len(results) if results else 0.0

        # Per-category
        category_scores: dict[MetricCategory, float] = {}
        for cat in MetricCategory:
            cat_results = [r for r in results if self.tasks[r.task_id].category == cat]
            if cat_results:
                category_scores[cat] = statistics.mean(r.score for r in cat_results)

        # Per-difficulty
        difficulty_scores: dict[EvalDifficulty, float] = {}
        for diff in EvalDifficulty:
            diff_results = [r for r in results if self.tasks[r.task_id].difficulty == diff]
            if diff_results:
                difficulty_scores[diff] = statistics.mean(r.score for r in diff_results)

        # Efficiency (inverse of time/tokens)
        times = [r.time_taken_seconds for r in results if r.completed]
        avg_efficiency = 1.0 / (1.0 + statistics.mean(times)) if times else 0.0

        # Quality (from quality category)
        quality_results = [r for r in results if self.tasks[r.task_id].category == MetricCategory.QUALITY]
        avg_quality = statistics.mean(r.score for r in quality_results) if quality_results else 0.0

        # Compare to baseline
        baseline = self._load_baseline()
        comparison = None
        if baseline:
            comparison = {
                "overall_delta": overall_score - baseline.overall_score,
                "improved_categories": [],
                "declined_categories": [],
            }
            for cat, score in category_scores.items():
                baseline_score = baseline.category_scores.get(cat.value, 0.0)
                if score > baseline_score + 0.05:
                    comparison["improved_categories"].append(cat.value)
                elif score < baseline_score - 0.05:
                    comparison["declined_categories"].append(cat.value)

        return EvalReport(
            id=report_id,
            overall_score=overall_score,
            completion_rate=completion_rate,
            average_quality=avg_quality,
            average_efficiency=avg_efficiency,
            category_scores=category_scores,
            difficulty_scores=difficulty_scores,
            results=results,
            baseline_comparison=comparison,
            agent_version=agent_version,
        )

    def set_baseline(self, report: EvalReport | None = None) -> BaselineMetrics:
        """Set current metrics as baseline."""
        if report is None:
            report = self.run_full_evaluation()

        baseline = BaselineMetrics(
            id=hashlib.md5(f"{datetime.now(UTC).isoformat()}".encode()).hexdigest()[:12],
            overall_score=report.overall_score,
            category_scores={k.value: v for k, v in report.category_scores.items()},
            difficulty_scores={k.value: v for k, v in report.difficulty_scores.items()},
            task_scores={r.task_id: r.score for r in report.results},
        )

        baseline_path = self.evals_dir / "baseline.json"
        baseline_path.write_text(json.dumps(baseline.model_dump(mode='json'), indent=2, default=str))

        return baseline

    def _load_baseline(self) -> BaselineMetrics | None:
        """Load baseline metrics."""
        baseline_path = self.evals_dir / "baseline.json"
        if baseline_path.exists():
            try:
                data = json.loads(baseline_path.read_text())
                return BaselineMetrics.model_validate(data)
            except Exception:
                return None
        return None

    def compare_to_baseline(self) -> dict[str, Any]:
        """Compare current performance to baseline."""
        baseline = self._load_baseline()
        if not baseline:
            return {"error": "No baseline set. Use set_baseline() first."}

        current = self.run_full_evaluation()

        return {
            "baseline_overall": baseline.overall_score,
            "current_overall": current.overall_score,
            "delta": current.overall_score - baseline.overall_score,
            "improved": current.overall_score > baseline.overall_score,
            "significant": abs(current.overall_score - baseline.overall_score) > 0.05,
            "category_deltas": {
                cat.value: current.category_scores.get(cat, 0) - baseline.category_scores.get(cat.value, 0)
                for cat in MetricCategory
            },
            "regressions": [
                cat.value for cat in MetricCategory
                if current.category_scores.get(cat, 0) < baseline.category_scores.get(cat.value, 0) - 0.05
            ],
        }

    def get_metrics_history(self, metric: str = "overall_score") -> MetricsHistory:
        """Get history of a metric over time."""
        reports = self._load_reports()

        data_points = []
        for report in sorted(reports, key=lambda r: r.timestamp):
            if metric == "overall_score":
                value = report.overall_score
            elif metric == "completion_rate":
                value = report.completion_rate
            elif metric == "average_quality":
                value = report.average_quality
            elif metric in [cat.value for cat in MetricCategory]:
                cat = MetricCategory(metric)
                value = report.category_scores.get(cat, 0.0)
            else:
                continue

            data_points.append((report.timestamp, value))

        # Compute trend
        trend = "stable"
        trend_strength = 0.0
        if len(data_points) >= 3:
            recent = [v for _, v in data_points[-5:]]
            older = [v for _, v in data_points[-10:-5]] or [v for _, v in data_points[:len(data_points)//2]]

            if older:
                recent_avg = statistics.mean(recent)
                older_avg = statistics.mean(older)
                diff = recent_avg - older_avg

                if diff > 0.05:
                    trend = "improving"
                    trend_strength = min(1.0, diff * 5)
                elif diff < -0.05:
                    trend = "declining"
                    trend_strength = min(1.0, -diff * 5)

        return MetricsHistory(
            metric_name=metric,
            data_points=data_points,
            trend=trend,
            trend_strength=trend_strength,
        )

    def _save_result(self, result: EvalResult) -> None:
        """Save individual result."""
        results_dir = self.evals_dir / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        path = results_dir / f"{result.task_id}_{timestamp}.json"
        path.write_text(json.dumps(result.model_dump(mode='json'), indent=2, default=str))

    def _save_report(self, report: EvalReport) -> None:
        """Save evaluation report."""
        reports_dir = self.evals_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        path = reports_dir / f"report_{report.id}.json"
        path.write_text(json.dumps(report.model_dump(mode='json'), indent=2, default=str))

    def _load_reports(self) -> list[EvalReport]:
        """Load all saved reports."""
        reports = []
        reports_dir = self.evals_dir / "reports"
        if reports_dir.exists():
            for path in reports_dir.glob("report_*.json"):
                try:
                    data = json.loads(path.read_text())
                    reports.append(EvalReport.model_validate(data))
                except Exception:
                    continue
        return reports

    def get_summary(self) -> dict[str, Any]:
        """Get summary of evaluation status."""
        baseline = self._load_baseline()
        reports = self._load_reports()

        latest_report = max(reports, key=lambda r: r.timestamp) if reports else None

        return {
            "total_tasks": len(EVAL_TASKS),
            "categories": list(MetricCategory),
            "difficulties": list(EvalDifficulty),
            "has_baseline": baseline is not None,
            "baseline_score": baseline.overall_score if baseline else None,
            "total_reports": len(reports),
            "latest_score": latest_report.overall_score if latest_report else None,
            "latest_timestamp": latest_report.timestamp.isoformat() if latest_report else None,
        }


# =============================================================================
# CLI Interface
# =============================================================================

def print_report(report: EvalReport) -> str:
    """Format report for display."""
    lines = [
        f"=== Evaluation Report {report.id} ===",
        f"Timestamp: {report.timestamp.isoformat()}",
        f"",
        f"Overall Score: {report.overall_score:.2%}",
        f"Completion Rate: {report.completion_rate:.2%}",
        f"Average Quality: {report.average_quality:.2%}",
        f"Average Efficiency: {report.average_efficiency:.2%}",
        f"",
        "Category Scores:",
    ]

    for cat, score in sorted(report.category_scores.items(), key=lambda x: -x[1]):
        lines.append(f"  {cat.value}: {score:.2%}")

    lines.append("")
    lines.append("Difficulty Scores:")
    for diff, score in sorted(report.difficulty_scores.items(), key=lambda x: -x[1]):
        lines.append(f"  {diff.value}: {score:.2%}")

    if report.baseline_comparison:
        lines.append("")
        lines.append("Baseline Comparison:")
        delta = report.baseline_comparison.get("overall_delta", 0)
        lines.append(f"  Overall Delta: {delta:+.2%}")
        if report.baseline_comparison.get("improved_categories"):
            lines.append(f"  Improved: {', '.join(report.baseline_comparison['improved_categories'])}")
        if report.baseline_comparison.get("declined_categories"):
            lines.append(f"  Declined: {', '.join(report.baseline_comparison['declined_categories'])}")

    lines.append("")
    lines.append("Individual Results:")
    for result in sorted(report.results, key=lambda r: -r.score):
        status = "✓" if result.completed else "✗"
        lines.append(f"  {status} {result.task_name}: {result.score:.2%} ({result.notes})")

    return "\n".join(lines)
