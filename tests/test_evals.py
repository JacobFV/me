"""Tests for the evaluation framework."""

import pytest
from pathlib import Path
from datetime import datetime, UTC

from me.agent.evals import (
    MetricCategory,
    EvalDifficulty,
    EvalTask,
    EvalResult,
    EvalReport,
    EvalValidators,
    TaskRunner,
    AgentEvaluator,
    EVAL_TASKS,
    print_report,
)


class TestEvalValidators:
    """Test evaluation validators."""

    def test_validate_fizzbuzz(self):
        validators = EvalValidators()

        # Good output
        output = "1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz"
        score, notes = validators.validate_fizzbuzz(output)
        assert score == 1.0

        # Missing FizzBuzz
        output = "1 2 Fizz 4 Buzz"
        score, notes = validators.validate_fizzbuzz(output)
        assert score == 0.75

        # Empty
        score, notes = validators.validate_fizzbuzz("")
        assert score == 0.0

    def test_validate_minimal_solution(self):
        validators = EvalValidators()

        # Minimal solution (1 line, ~20 chars)
        output = "s[::-1]"
        score, _ = validators.validate_minimal_solution(output)
        assert score > 0.8

        # Verbose solution
        output = """
def reverse_string(s):
    result = ''
    for char in s:
        result = char + result
    return result
"""
        score, _ = validators.validate_minimal_solution(output)
        assert score < 0.5

    def test_validate_robustness(self):
        validators = EvalValidators()

        # Good handling
        output = "Input was handled gracefully"
        score, _ = validators.validate_robustness(output, {})
        assert score == 1.0

        # Error but handled
        output = "Error encountered but handled gracefully"
        score, _ = validators.validate_robustness(output, {})
        assert score > 0.5


class TestTaskRunner:
    """Test task runner."""

    def test_run_task_without_agent(self):
        runner = TaskRunner()
        task = EvalTask(
            id="test_task",
            name="Test Task",
            description="A test task",
            category=MetricCategory.TASK_COMPLETION,
            difficulty=EvalDifficulty.SIMPLE,
        )

        result = runner.run_task(task, agent_fn=None)
        assert result.task_id == "test_task"
        assert result.completed == True
        # Score should be 1.0 because default validation passes for non-empty

    def test_run_task_with_mock_agent(self):
        runner = TaskRunner()
        task = EvalTask(
            id="test_hello",
            name="Hello Test",
            description="Return hello",
            category=MetricCategory.TASK_COMPLETION,
            difficulty=EvalDifficulty.TRIVIAL,
            expected_output={"contains": "hello"},
        )

        def mock_agent(input_data):
            return "hello world", 100

        result = runner.run_task(task, agent_fn=mock_agent)
        assert result.completed == True
        assert result.score == 1.0
        assert result.tokens_used == 100

    def test_run_task_with_time_limit(self):
        runner = TaskRunner()
        task = EvalTask(
            id="test_slow",
            name="Slow Test",
            description="Test time limit",
            category=MetricCategory.EFFICIENCY,
            difficulty=EvalDifficulty.SIMPLE,
            time_limit_seconds=0.01,  # Very short limit
            expected_output={"contains": "done"},
        )

        import time
        def slow_agent(input_data):
            time.sleep(0.1)  # Sleep longer than limit
            return "done", 100

        result = runner.run_task(task, agent_fn=slow_agent)
        assert result.completed == True
        assert result.score == 0.5  # Penalty for timeout
        assert "TIMEOUT" in result.notes

    def test_run_task_with_token_budget(self):
        runner = TaskRunner()
        task = EvalTask(
            id="test_budget",
            name="Budget Test",
            description="Test token budget",
            category=MetricCategory.EFFICIENCY,
            difficulty=EvalDifficulty.SIMPLE,
            token_budget=50,
            expected_output={"contains": "done"},
        )

        def expensive_agent(input_data):
            return "done", 100  # Over budget

        result = runner.run_task(task, agent_fn=expensive_agent)
        assert result.completed == True
        assert result.score == 0.8  # Penalty for over budget
        assert "OVER BUDGET" in result.notes


class TestAgentEvaluator:
    """Test agent evaluator."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        body_dir = tmp_path / "body"
        body_dir.mkdir()
        return AgentEvaluator(body_dir)

    def test_run_eval(self, evaluator):
        result = evaluator.run_eval("coding_trivial_hello")
        assert result.task_id == "coding_trivial_hello"
        assert result.task_name == "Hello World"

    def test_run_category(self, evaluator):
        results = evaluator.run_category(MetricCategory.TASK_COMPLETION)
        assert len(results) > 0
        for result in results:
            task = evaluator.tasks[result.task_id]
            assert task.category == MetricCategory.TASK_COMPLETION

    def test_run_difficulty(self, evaluator):
        results = evaluator.run_difficulty(EvalDifficulty.SIMPLE)
        assert len(results) > 0
        for result in results:
            task = evaluator.tasks[result.task_id]
            assert task.difficulty == EvalDifficulty.SIMPLE

    def test_run_full_evaluation(self, evaluator):
        report = evaluator.run_full_evaluation(agent_version="test-1.0")
        assert report.agent_version == "test-1.0"
        assert len(report.results) == len(EVAL_TASKS)
        assert 0 <= report.overall_score <= 1.0
        assert 0 <= report.completion_rate <= 1.0

    def test_set_and_compare_baseline(self, evaluator):
        # Set initial baseline
        report = evaluator.run_full_evaluation()
        baseline = evaluator.set_baseline(report)
        assert baseline.overall_score == report.overall_score

        # Compare to baseline
        comparison = evaluator.compare_to_baseline()
        assert "baseline_overall" in comparison
        assert "current_overall" in comparison
        assert "delta" in comparison

    def test_get_metrics_history(self, evaluator):
        # Run a few evaluations
        evaluator.run_full_evaluation()
        evaluator.run_full_evaluation()

        history = evaluator.get_metrics_history("overall_score")
        assert history.metric_name == "overall_score"
        assert len(history.data_points) >= 2

    def test_get_summary(self, evaluator):
        summary = evaluator.get_summary()
        assert summary["total_tasks"] == len(EVAL_TASKS)
        assert "categories" in summary
        assert "difficulties" in summary


class TestEvalTasks:
    """Test the predefined evaluation tasks."""

    def test_all_tasks_have_required_fields(self):
        for task in EVAL_TASKS:
            assert task.id
            assert task.name
            assert task.description
            assert task.category in MetricCategory
            assert task.difficulty in EvalDifficulty

    def test_task_ids_are_unique(self):
        ids = [task.id for task in EVAL_TASKS]
        assert len(ids) == len(set(ids))

    def test_coverage_of_categories(self):
        categories_covered = set(task.category for task in EVAL_TASKS)
        # Should have at least some category coverage
        assert len(categories_covered) >= 5

    def test_coverage_of_difficulties(self):
        difficulties_covered = set(task.difficulty for task in EVAL_TASKS)
        # Should have multiple difficulty levels
        assert len(difficulties_covered) >= 3


class TestPrintReport:
    """Test report formatting."""

    def test_print_report_basic(self):
        report = EvalReport(
            id="test123",
            overall_score=0.75,
            completion_rate=0.9,
            average_quality=0.8,
            average_efficiency=0.7,
            category_scores={MetricCategory.TASK_COMPLETION: 0.8},
            difficulty_scores={EvalDifficulty.SIMPLE: 0.85},
            results=[
                EvalResult(
                    task_id="test",
                    task_name="Test Task",
                    completed=True,
                    score=0.8,
                    raw_score=0.8,
                    max_score=1.0,
                    time_taken_seconds=1.5,
                    notes="Good job",
                )
            ],
        )

        output = print_report(report)
        assert "test123" in output
        assert "75.00%" in output
        assert "Test Task" in output

    def test_print_report_with_baseline_comparison(self):
        report = EvalReport(
            id="test456",
            overall_score=0.8,
            completion_rate=0.9,
            average_quality=0.8,
            average_efficiency=0.7,
            baseline_comparison={
                "overall_delta": 0.1,
                "improved_categories": ["task_completion"],
                "declined_categories": [],
            },
            results=[],
        )

        output = print_report(report)
        assert "Baseline Comparison" in output
        assert "improved_categories" in output.lower() or "Improved" in output
