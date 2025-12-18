"""
CLI for running evaluations.

Usage:
    python -m evals.cli run --suite basic
    python -m evals.cli run --suite full --output ./results
    python -m evals.cli list
    python -m evals.cli report ./results/full_20240101_120000.json
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import click

from evals.framework import EvalRunner, EvalReport
from evals.suites import list_suites, get_suite


@click.group()
def cli():
    """Evaluation CLI for the me agent."""
    pass


@cli.command()
def list():
    """List available eval suites."""
    suites = list_suites()

    click.echo("\nAvailable Eval Suites:")
    click.echo("-" * 60)

    for suite in suites:
        click.echo(f"\n  {suite['id']}")
        click.echo(f"    Name: {suite['name']}")
        click.echo(f"    Scenarios: {suite['scenario_count']}")
        click.echo(f"    {suite['description']}")

    click.echo("\n")


@cli.command()
@click.option("--suite", "-s", default="basic", help="Eval suite to run")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--parallel", "-p", is_flag=True, help="Run scenarios in parallel")
@click.option("--tags", "-t", multiple=True, help="Filter scenarios by tag")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(suite: str, output: str | None, parallel: bool, tags: tuple, verbose: bool):
    """Run an eval suite."""
    eval_suite = get_suite(suite)
    if not eval_suite:
        click.echo(f"Unknown suite: {suite}")
        click.echo("Use 'python -m evals.cli list' to see available suites")
        sys.exit(1)

    output_dir = Path(output) if output else Path.home() / ".me" / "evals"
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\nRunning eval: {eval_suite.name}")
    click.echo(f"Scenarios: {len(eval_suite.scenarios)}")
    click.echo(f"Output: {output_dir}")
    click.echo("-" * 60)

    # Create agent
    from me.agent.core import Agent, AgentConfig
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(base_dir=Path(tmpdir))
        agent = Agent(config)

        # Create runner with progress callback
        def on_result(result):
            status = "PASS" if result.success else "FAIL"
            click.echo(f"  [{status}] {result.scenario_name} ({result.duration_seconds:.1f}s)")
            if verbose and result.error:
                click.echo(f"       Error: {result.error}")

        runner = EvalRunner(
            output_dir=output_dir,
            parallel=parallel,
            on_result=on_result,
        )

        # Run eval
        tag_list = list(tags) if tags else None
        results = asyncio.run(runner.run_eval(eval_suite, agent, tags=tag_list))

        # Save results
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = runner.save_results(eval_suite, results, run_id)

        # Generate report
        report = EvalReport.from_results(eval_suite.name, run_id, results)

        click.echo("\n" + "=" * 60)
        click.echo("SUMMARY")
        click.echo("=" * 60)
        click.echo(f"Total: {report.summary['total_scenarios']}")
        click.echo(f"Passed: {report.summary['successes']}")
        click.echo(f"Failed: {report.summary['failures']}")
        click.echo(f"Success Rate: {report.summary['success_rate']:.1%}")
        click.echo(f"Duration: {report.summary['total_duration_seconds']:.1f}s")
        click.echo(f"\nResults saved to: {output_file}")

        # Save markdown report
        report_file = output_dir / f"{suite}_{run_id}_report.md"
        report_file.write_text(report.to_markdown())
        click.echo(f"Report saved to: {report_file}")


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(["text", "markdown", "json"]), default="text")
def report(results_file: str, format: str):
    """Generate report from results file."""
    results_path = Path(results_file)
    data = json.loads(results_path.read_text())

    if format == "json":
        click.echo(json.dumps(data, indent=2))
    elif format == "markdown":
        # Reconstruct results for report
        from evals.framework import EvalResult, EvalTrace, Metric, MetricType

        results = []
        for r in data.get("results", []):
            metrics = [
                Metric(
                    name=m["name"],
                    value=m["value"],
                    metric_type=MetricType(m["type"]),
                    metadata=m.get("metadata", {}),
                )
                for m in r.get("metrics", [])
            ]
            result = EvalResult(
                scenario_id=r["scenario_id"],
                scenario_name=r["scenario_name"],
                success=r["success"],
                metrics=metrics,
                trace=EvalTrace(),
                started_at=datetime.fromisoformat(r["started_at"]),
                completed_at=datetime.fromisoformat(r["completed_at"]),
                error=r.get("error"),
            )
            results.append(result)

        report = EvalReport.from_results(
            data.get("eval_name", "Unknown"),
            data.get("run_id", "Unknown"),
            results,
        )
        click.echo(report.to_markdown())
    else:
        # Text format
        click.echo(f"\nEval: {data.get('eval_name', 'Unknown')}")
        click.echo(f"Run ID: {data.get('run_id', 'Unknown')}")
        click.echo(f"Timestamp: {data.get('timestamp', 'Unknown')}")
        click.echo("-" * 40)

        summary = data.get("summary", {})
        click.echo(f"Total: {summary.get('total_scenarios', 0)}")
        click.echo(f"Success Rate: {summary.get('success_rate', 0):.1%}")

        click.echo("\nResults:")
        for r in data.get("results", []):
            status = "PASS" if r["success"] else "FAIL"
            click.echo(f"  [{status}] {r['scenario_name']}")


if __name__ == "__main__":
    cli()
