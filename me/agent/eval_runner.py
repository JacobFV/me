"""
Agent Evaluation Runner - Connects real agent to eval framework.

This module runs the actual agent against evaluation tasks and
generates detailed reports for CI and development.

Usage:
    # Run from CLI
    python -m me.agent.eval_runner --output report.md

    # Run in CI
    ANTHROPIC_API_KEY=xxx python -m me.agent.eval_runner --ci

    # Run specific categories
    python -m me.agent.eval_runner --category task_completion
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from me.agent.core import Agent, AgentConfig
from me.agent.evals import (
    AgentEvaluator,
    EvalReport,
    EvalResult,
    EvalTask,
    MetricCategory,
    EvalDifficulty,
    EVAL_TASKS,
)


class AgentEvalRunner:
    """
    Runs the actual agent against evaluation tasks.

    This bridges the eval framework with the real agent implementation.
    """

    def __init__(
        self,
        body_dir: Path | None = None,
        timeout_per_task: float = 60.0,
        max_tokens_per_task: int = 4000,
    ):
        self.body_dir = body_dir or Path(tempfile.mkdtemp(prefix="me_eval_"))
        self.timeout_per_task = timeout_per_task
        self.max_tokens_per_task = max_tokens_per_task
        self.evaluator = AgentEvaluator(self.body_dir)

        # Initialize agent
        config = AgentConfig(base_dir=self.body_dir)
        self.agent = Agent(config)

        # Track token usage
        self.total_tokens = 0

    async def run_task_with_agent(
        self,
        task: EvalTask,
    ) -> tuple[str, int]:
        """
        Run a single task with the actual agent.

        Returns:
            Tuple of (output_text, tokens_used)
        """
        # Build prompt from task
        prompt = self._build_prompt(task)

        try:
            # Run agent with timeout
            output_parts = []
            tokens_used = 0

            async def run_with_timeout():
                nonlocal tokens_used
                async for msg in self.agent.run(prompt):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            output_parts.append(content)
                        elif isinstance(content, list):
                            for part in content:
                                if hasattr(part, "text"):
                                    output_parts.append(part.text)
                    # Track tokens if available
                    if "usage" in msg:
                        tokens_used += msg["usage"].get("output_tokens", 0)

            await asyncio.wait_for(
                run_with_timeout(),
                timeout=self.timeout_per_task,
            )

            output = "\n".join(output_parts)
            self.total_tokens += tokens_used
            return output, tokens_used

        except asyncio.TimeoutError:
            return f"[TIMEOUT after {self.timeout_per_task}s]", 0
        except Exception as e:
            return f"[ERROR: {str(e)}]", 0

    def _build_prompt(self, task: EvalTask) -> str:
        """Build a prompt for the agent from a task."""
        prompt_parts = [
            f"# Evaluation Task: {task.name}",
            "",
            task.description,
            "",
        ]

        if task.input_data:
            prompt_parts.append("## Input Data")
            prompt_parts.append("```json")
            prompt_parts.append(json.dumps(task.input_data, indent=2))
            prompt_parts.append("```")
            prompt_parts.append("")

        if task.expected_output:
            prompt_parts.append("## Expected Output Format")
            prompt_parts.append(str(task.expected_output))
            prompt_parts.append("")

        prompt_parts.append("Please complete this task. Provide your response directly.")

        return "\n".join(prompt_parts)

    def create_agent_fn(self) -> callable:
        """Create an agent function for the evaluator."""
        async def agent_fn_async(input_data: dict) -> tuple[str, int]:
            # Build a simple task from input_data
            task = EvalTask(
                id="dynamic",
                name="Dynamic Task",
                description=str(input_data),
                category=MetricCategory.TASK_COMPLETION,
                difficulty=EvalDifficulty.SIMPLE,
                input_data=input_data,
            )
            return await self.run_task_with_agent(task)

        def agent_fn(input_data: dict) -> tuple[str, int]:
            return asyncio.run(agent_fn_async(input_data))

        return agent_fn

    async def run_all_evals(self) -> EvalReport:
        """Run all evaluation tasks with the agent."""
        results = []

        for task in EVAL_TASKS:
            print(f"Running: {task.name} ({task.difficulty.value})...", end=" ", flush=True)
            start_time = time.time()

            try:
                output, tokens = await self.run_task_with_agent(task)
                time_taken = time.time() - start_time

                # Validate
                result = self.evaluator.runner.run_task(
                    task,
                    agent_fn=lambda _: (output, tokens),
                )

                print(f"Score: {result.score:.0%} ({time_taken:.1f}s)")
                results.append(result)

            except Exception as e:
                print(f"ERROR: {e}")
                results.append(EvalResult(
                    task_id=task.id,
                    task_name=task.name,
                    completed=False,
                    score=0.0,
                    raw_score=0.0,
                    max_score=task.max_score,
                    time_taken_seconds=time.time() - start_time,
                    errors=[str(e)],
                ))

        # Generate report
        report = self.evaluator._generate_report(results, agent_version=self._get_version())
        self.evaluator._save_report(report)

        return report

    async def run_category(self, category: MetricCategory) -> EvalReport:
        """Run tasks in a specific category."""
        results = []
        tasks = [t for t in EVAL_TASKS if t.category == category]

        for task in tasks:
            print(f"Running: {task.name}...", end=" ", flush=True)
            start_time = time.time()

            output, tokens = await self.run_task_with_agent(task)

            result = self.evaluator.runner.run_task(
                task,
                agent_fn=lambda _: (output, tokens),
            )

            print(f"Score: {result.score:.0%}")
            results.append(result)

        return self.evaluator._generate_report(results, agent_version=self._get_version())

    def _get_version(self) -> str:
        """Get agent version from git or environment."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return os.environ.get("AGENT_VERSION", "unknown")


def generate_markdown_report(report: EvalReport) -> str:
    """Generate a detailed Markdown report."""
    lines = [
        "# Agent Evaluation Report",
        "",
        f"**Date:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Version:** {report.agent_version}",
        f"**Report ID:** {report.id}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall Score | {report.overall_score:.1%} |",
        f"| Completion Rate | {report.completion_rate:.1%} |",
        f"| Average Quality | {report.average_quality:.1%} |",
        f"| Average Efficiency | {report.average_efficiency:.1%} |",
        "",
    ]

    # Category breakdown
    lines.extend([
        "## Scores by Category",
        "",
        "| Category | Score | Status |",
        "|----------|-------|--------|",
    ])

    for cat, score in sorted(report.category_scores.items(), key=lambda x: -x[1]):
        status = "✅" if score >= 0.7 else "⚠️" if score >= 0.4 else "❌"
        lines.append(f"| {cat.value} | {score:.1%} | {status} |")

    lines.append("")

    # Difficulty breakdown
    lines.extend([
        "## Scores by Difficulty",
        "",
        "| Difficulty | Score | Status |",
        "|------------|-------|--------|",
    ])

    for diff, score in sorted(report.difficulty_scores.items(), key=lambda x: list(EvalDifficulty).index(x[0])):
        status = "✅" if score >= 0.7 else "⚠️" if score >= 0.4 else "❌"
        lines.append(f"| {diff.value} | {score:.1%} | {status} |")

    lines.append("")

    # Baseline comparison
    if report.baseline_comparison:
        lines.extend([
            "## Baseline Comparison",
            "",
        ])
        delta = report.baseline_comparison.get("overall_delta", 0)
        direction = "📈 Improved" if delta > 0 else "📉 Declined" if delta < 0 else "➡️ Unchanged"
        lines.append(f"**Overall Change:** {delta:+.1%} ({direction})")
        lines.append("")

        if report.baseline_comparison.get("improved_categories"):
            lines.append(f"**Improved Categories:** {', '.join(report.baseline_comparison['improved_categories'])}")
        if report.baseline_comparison.get("declined_categories"):
            lines.append(f"**Declined Categories:** {', '.join(report.baseline_comparison['declined_categories'])}")
        if report.baseline_comparison.get("regressions"):
            lines.append(f"**⚠️ Regressions:** {', '.join(report.baseline_comparison['regressions'])}")

        lines.append("")

    # Individual results
    lines.extend([
        "## Individual Task Results",
        "",
        "| Task | Category | Difficulty | Score | Time | Status |",
        "|------|----------|------------|-------|------|--------|",
    ])

    for result in sorted(report.results, key=lambda r: (-r.score, r.task_name)):
        task = next((t for t in EVAL_TASKS if t.id == result.task_id), None)
        if task:
            status = "✅" if result.completed and result.score >= 0.7 else "⚠️" if result.completed else "❌"
            lines.append(
                f"| {result.task_name} | {task.category.value} | {task.difficulty.value} | "
                f"{result.score:.1%} | {result.time_taken_seconds:.1f}s | {status} |"
            )

    lines.append("")

    # Errors if any
    errors = [r for r in report.results if r.errors]
    if errors:
        lines.extend([
            "## Errors",
            "",
        ])
        for result in errors:
            lines.append(f"### {result.task_name}")
            for error in result.errors:
                lines.append(f"- {error}")
            lines.append("")

    # Footer
    lines.extend([
        "---",
        "",
        f"*Generated at {datetime.now(UTC).isoformat()}*",
    ])

    return "\n".join(lines)


def generate_json_report(report: EvalReport) -> str:
    """Generate a JSON report for programmatic consumption."""
    return json.dumps(report.model_dump(mode='json'), indent=2, default=str)


async def main():
    parser = argparse.ArgumentParser(description="Run agent evaluations")
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--category", "-c",
        choices=[c.value for c in MetricCategory],
        help="Run only specific category",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode - fail on regression",
    )
    parser.add_argument(
        "--set-baseline",
        action="store_true",
        help="Set results as new baseline",
    )
    parser.add_argument(
        "--body-dir",
        type=Path,
        help="Body directory for agent",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout per task in seconds",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip actual API calls (placeholder mode)",
    )

    args = parser.parse_args()

    # Check for API key
    if not args.skip_api and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set. Use --skip-api for placeholder mode.", file=sys.stderr)
        if args.ci:
            print("CI mode requires API key. Exiting.", file=sys.stderr)
            sys.exit(1)
        args.skip_api = True

    print("=" * 60)
    print("Agent Evaluation Runner")
    print("=" * 60)
    print()

    if args.skip_api:
        print("Running in placeholder mode (no API calls)")
        body_dir = args.body_dir or Path(tempfile.mkdtemp(prefix="me_eval_"))
        evaluator = AgentEvaluator(body_dir)
        report = evaluator.run_full_evaluation(agent_version="placeholder")
    else:
        print("Running with actual agent")
        runner = AgentEvalRunner(
            body_dir=args.body_dir,
            timeout_per_task=args.timeout,
        )

        if args.category:
            category = MetricCategory(args.category)
            report = await runner.run_category(category)
        else:
            report = await runner.run_all_evals()

        print()
        print(f"Total tokens used: {runner.total_tokens}")

    # Generate output
    if args.format == "markdown":
        output = generate_markdown_report(report)
    elif args.format == "json":
        output = generate_json_report(report)
    else:
        from me.agent.evals import print_report
        output = print_report(report)

    # Write output
    if args.output:
        Path(args.output).write_text(output)
        print(f"\nReport written to: {args.output}")
    else:
        print()
        print(output)

    # Set baseline if requested
    if args.set_baseline:
        if args.body_dir:
            evaluator = AgentEvaluator(args.body_dir)
        evaluator.set_baseline(report)
        print("\nBaseline updated!")

    # CI mode - check for regressions
    if args.ci:
        if args.body_dir:
            evaluator = AgentEvaluator(args.body_dir)

        baseline = evaluator._load_baseline()
        if baseline:
            if report.overall_score < baseline.overall_score - 0.05:
                print(f"\n❌ REGRESSION DETECTED: Score dropped from {baseline.overall_score:.1%} to {report.overall_score:.1%}")
                sys.exit(1)
            elif report.overall_score > baseline.overall_score + 0.05:
                print(f"\n✅ IMPROVEMENT: Score improved from {baseline.overall_score:.1%} to {report.overall_score:.1%}")
            else:
                print(f"\n➡️ STABLE: Score is {report.overall_score:.1%} (baseline: {baseline.overall_score:.1%})")
        else:
            print("\nNo baseline set. Consider running with --set-baseline to establish one.")

    print()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
