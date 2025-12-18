"""
Core evaluation framework.

Design principles:
1. Scenarios define WHAT to evaluate (task, context, expected outcomes)
2. Metrics define HOW to score (quantitative measures)
3. Results capture the full picture (scores, traces, artifacts)
4. Reports synthesize insights (not just numbers)

Careful interpretation:
- High scores don't always mean "good" (Goodhart's Law)
- Context matters - a slow but correct agent may be preferable
- Distributions matter more than means
- Failure modes are often more informative than successes
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class MetricType(str, Enum):
    """Types of metrics for different interpretations."""
    BINARY = "binary"  # 0 or 1 (task completed or not)
    SCORE = "score"  # 0.0 to 1.0 (quality measure)
    COUNT = "count"  # Integer count (steps, tokens, etc.)
    DURATION = "duration"  # Time in seconds
    RATIO = "ratio"  # Proportion (correct/total)
    CATEGORICAL = "categorical"  # Discrete categories


@dataclass
class Metric:
    """A single measurement from an eval."""
    name: str
    value: float | int | str
    metric_type: MetricType
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "metadata": self.metadata,
        }


@dataclass
class EvalTrace:
    """Trace of agent execution during eval."""
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def add_tool_call(self, name: str, args: dict, result: Any):
        self.tool_calls.append({
            "name": name,
            "args": args,
            "result": str(result)[:500],  # Truncate
            "timestamp": datetime.now().isoformat(),
        })


@dataclass
class EvalResult:
    """Result of a single eval run."""
    scenario_id: str
    scenario_name: str
    success: bool
    metrics: list[Metric]
    trace: EvalTrace
    started_at: datetime
    completed_at: datetime
    error: str | None = None
    notes: str = ""

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    def get_metric(self, name: str) -> Metric | None:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "success": self.success,
            "metrics": [m.to_dict() for m in self.metrics],
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "notes": self.notes,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
        }


class Scenario(ABC):
    """
    Base class for eval scenarios.

    A scenario defines:
    - What task the agent should perform
    - What context/setup is needed
    - How to judge success
    - What metrics to collect

    Subclasses implement specific evaluation types.
    """

    def __init__(
        self,
        scenario_id: str,
        name: str,
        description: str,
        tags: list[str] | None = None,
    ):
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.tags = tags or []

    @abstractmethod
    async def setup(self, agent: Any) -> dict[str, Any]:
        """
        Set up the scenario before running.

        Returns context dict that will be passed to run() and evaluate().
        """
        pass

    @abstractmethod
    async def run(self, agent: Any, context: dict[str, Any]) -> EvalTrace:
        """
        Run the agent on this scenario.

        Returns trace of execution.
        """
        pass

    @abstractmethod
    async def evaluate(
        self,
        agent: Any,
        context: dict[str, Any],
        trace: EvalTrace,
    ) -> tuple[bool, list[Metric]]:
        """
        Evaluate the agent's performance.

        Returns (success, metrics).
        """
        pass

    async def teardown(self, agent: Any, context: dict[str, Any]):
        """Clean up after scenario. Override if needed."""
        pass


class Eval:
    """
    A collection of scenarios forming a complete evaluation.

    Evals group related scenarios and provide aggregate analysis.
    """

    def __init__(
        self,
        eval_id: str,
        name: str,
        description: str,
        scenarios: list[Scenario],
    ):
        self.eval_id = eval_id
        self.name = name
        self.description = description
        self.scenarios = scenarios

    def filter_scenarios(self, tags: list[str] | None = None) -> list[Scenario]:
        """Filter scenarios by tags."""
        if not tags:
            return self.scenarios
        return [s for s in self.scenarios if any(t in s.tags for t in tags)]


class EvalRunner:
    """
    Runs evals and collects results.

    Handles:
    - Sequential or parallel execution
    - Error recovery
    - Progress reporting
    - Result aggregation
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        parallel: bool = False,
        max_workers: int = 4,
        on_result: Callable[[EvalResult], None] | None = None,
    ):
        self.output_dir = output_dir or Path.home() / ".me" / "evals"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parallel = parallel
        self.max_workers = max_workers
        self.on_result = on_result

    async def run_scenario(
        self,
        scenario: Scenario,
        agent: Any,
    ) -> EvalResult:
        """Run a single scenario."""
        started_at = datetime.now()
        trace = EvalTrace()

        try:
            # Setup
            context = await scenario.setup(agent)

            # Run
            trace = await scenario.run(agent, context)

            # Evaluate
            success, metrics = await scenario.evaluate(agent, context, trace)

            # Teardown
            await scenario.teardown(agent, context)

            result = EvalResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                success=success,
                metrics=metrics,
                trace=trace,
                started_at=started_at,
                completed_at=datetime.now(),
            )

        except Exception as e:
            result = EvalResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                success=False,
                metrics=[],
                trace=trace,
                started_at=started_at,
                completed_at=datetime.now(),
                error=str(e),
            )

        if self.on_result:
            self.on_result(result)

        return result

    async def run_eval(
        self,
        eval_: Eval,
        agent: Any,
        tags: list[str] | None = None,
    ) -> list[EvalResult]:
        """Run all scenarios in an eval."""
        scenarios = eval_.filter_scenarios(tags)
        results = []

        if self.parallel:
            # Run in parallel with semaphore
            sem = asyncio.Semaphore(self.max_workers)

            async def run_with_sem(s):
                async with sem:
                    return await self.run_scenario(s, agent)

            results = await asyncio.gather(
                *[run_with_sem(s) for s in scenarios],
                return_exceptions=True,
            )
            # Convert exceptions to failed results
            results = [
                r if isinstance(r, EvalResult)
                else EvalResult(
                    scenario_id="unknown",
                    scenario_name="unknown",
                    success=False,
                    metrics=[],
                    trace=EvalTrace(),
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    error=str(r),
                )
                for r in results
            ]
        else:
            for scenario in scenarios:
                result = await self.run_scenario(scenario, agent)
                results.append(result)

        return results

    def save_results(
        self,
        eval_: Eval,
        results: list[EvalResult],
        run_id: str | None = None,
    ) -> Path:
        """Save results to file."""
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{eval_.eval_id}_{run_id}.json"

        data = {
            "eval_id": eval_.eval_id,
            "eval_name": eval_.name,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
            "summary": self._compute_summary(results),
        }

        output_file.write_text(json.dumps(data, indent=2))
        return output_file

    def _compute_summary(self, results: list[EvalResult]) -> dict:
        """Compute summary statistics."""
        if not results:
            return {}

        successes = sum(1 for r in results if r.success)
        total = len(results)

        # Aggregate metrics
        metric_values: dict[str, list[float]] = {}
        for r in results:
            for m in r.metrics:
                if m.metric_type in (MetricType.SCORE, MetricType.RATIO, MetricType.COUNT, MetricType.DURATION):
                    if m.name not in metric_values:
                        metric_values[m.name] = []
                    metric_values[m.name].append(float(m.value))

        metric_stats = {}
        for name, values in metric_values.items():
            if values:
                metric_stats[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return {
            "total_scenarios": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": successes / total if total > 0 else 0,
            "total_duration_seconds": sum(r.duration_seconds for r in results),
            "metric_stats": metric_stats,
        }


@dataclass
class EvalReport:
    """
    Human-readable report from eval results.

    Focus on insights, not just numbers:
    - What patterns emerge?
    - Where does the agent struggle?
    - What improvements would help most?
    """

    eval_name: str
    run_id: str
    results: list[EvalResult]
    summary: dict

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Eval Report: {self.eval_name}",
            f"",
            f"**Run ID:** {self.run_id}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            "## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Scenarios | {self.summary.get('total_scenarios', 0)} |",
            f"| Successes | {self.summary.get('successes', 0)} |",
            f"| Failures | {self.summary.get('failures', 0)} |",
            f"| Success Rate | {self.summary.get('success_rate', 0):.1%} |",
            f"| Total Duration | {self.summary.get('total_duration_seconds', 0):.1f}s |",
            f"",
        ]

        # Metric stats
        metric_stats = self.summary.get("metric_stats", {})
        if metric_stats:
            lines.extend([
                "## Metrics",
                "",
                "| Metric | Mean | Min | Max |",
                "|--------|------|-----|-----|",
            ])
            for name, stats in metric_stats.items():
                lines.append(
                    f"| {name} | {stats['mean']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |"
                )
            lines.append("")

        # Failures analysis
        failures = [r for r in self.results if not r.success]
        if failures:
            lines.extend([
                "## Failures",
                "",
            ])
            for f in failures:
                lines.append(f"### {f.scenario_name}")
                if f.error:
                    lines.append(f"**Error:** {f.error}")
                lines.append("")

        # Success patterns
        successes = [r for r in self.results if r.success]
        if successes:
            lines.extend([
                "## Insights",
                "",
                f"- {len(successes)} scenarios completed successfully",
            ])
            # Find fastest/slowest
            if len(successes) > 1:
                fastest = min(successes, key=lambda r: r.duration_seconds)
                slowest = max(successes, key=lambda r: r.duration_seconds)
                lines.append(f"- Fastest: {fastest.scenario_name} ({fastest.duration_seconds:.1f}s)")
                lines.append(f"- Slowest: {slowest.scenario_name} ({slowest.duration_seconds:.1f}s)")

        return "\n".join(lines)

    @classmethod
    def from_results(
        cls,
        eval_name: str,
        run_id: str,
        results: list[EvalResult],
    ) -> "EvalReport":
        """Create report from results."""
        runner = EvalRunner()
        summary = runner._compute_summary(results)
        return cls(
            eval_name=eval_name,
            run_id=run_id,
            results=results,
            summary=summary,
        )
