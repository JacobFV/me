"""
Reasoning eval scenarios.

Measures: Can the agent reason through problems?

Key metrics:
- logical_validity: Are conclusions logically sound? (binary)
- evidence_use: Does agent cite evidence for claims? (ratio)
- uncertainty_calibration: Does agent know what it doesn't know? (score)
- multi_step: Can agent chain reasoning steps? (binary)

Interpretation notes:
- Correct answers from wrong reasoning are concerning
- Expressing uncertainty is often better than false confidence
- Reasoning should be traceable in agent's output
- Different reasoning types (deductive, inductive, abductive) have different uses
"""

from __future__ import annotations

from typing import Any

from evals.framework import (
    Scenario,
    EvalTrace,
    Metric,
    MetricType,
)


class ReasoningScenario(Scenario):
    """Base class for reasoning scenarios."""

    def __init__(
        self,
        scenario_id: str,
        name: str,
        description: str,
        problem: str,
        expected_answer: str | None = None,
        reasoning_keywords: list[str] | None = None,
        tags: list[str] | None = None,
    ):
        super().__init__(scenario_id, name, description, tags)
        self.problem = problem
        self.expected_answer = expected_answer
        self.reasoning_keywords = reasoning_keywords or [
            "because", "therefore", "since", "if", "then",
            "follows", "implies", "conclude", "reason"
        ]

    async def setup(self, agent: Any) -> dict[str, Any]:
        return {}

    async def run(self, agent: Any, context: dict[str, Any]) -> EvalTrace:
        trace = EvalTrace()
        trace.add_message("user", self.problem)

        try:
            async for message in agent.run(self.problem):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                if role == "assistant":
                    trace.add_message("assistant", content)
                    # Track reasoning indicators
                    for keyword in self.reasoning_keywords:
                        if keyword.lower() in content.lower():
                            trace.thoughts.append(f"Used reasoning word: {keyword}")
                elif role == "tool":
                    trace.add_tool_call(
                        message.get("name", "unknown"),
                        message.get("input", {}),
                        message.get("result", ""),
                    )
        except Exception as e:
            trace.errors.append(str(e))

        return trace

    async def evaluate(
        self,
        agent: Any,
        context: dict[str, Any],
        trace: EvalTrace,
    ) -> tuple[bool, list[Metric]]:
        metrics = []

        response = ""
        for msg in trace.messages:
            if msg.get("role") == "assistant":
                response += msg.get("content", "")

        response_lower = response.lower()

        # Count reasoning keywords used
        reasoning_count = sum(
            1 for kw in self.reasoning_keywords
            if kw.lower() in response_lower
        )
        metrics.append(Metric(
            name="reasoning_keywords",
            value=reasoning_count,
            metric_type=MetricType.COUNT,
        ))

        # Check answer correctness if expected answer provided
        answer_correct = False
        if self.expected_answer:
            answer_correct = self.expected_answer.lower() in response_lower
            metrics.append(Metric(
                name="answer_correct",
                value=1 if answer_correct else 0,
                metric_type=MetricType.BINARY,
                metadata={"expected": self.expected_answer},
            ))

        # Check for uncertainty acknowledgment
        uncertainty_words = ["might", "could", "uncertain", "possibly", "maybe", "not sure", "unclear"]
        uncertainty_count = sum(1 for w in uncertainty_words if w in response_lower)
        metrics.append(Metric(
            name="uncertainty_expressed",
            value=uncertainty_count,
            metric_type=MetricType.COUNT,
        ))

        success = answer_correct if self.expected_answer else reasoning_count >= 2
        return success, metrics


class DeductiveReasoningScenario(ReasoningScenario):
    """Scenario: Test deductive reasoning (if A then B, A, therefore B)."""

    def __init__(self, scenario_id: str = "deductive_reasoning"):
        super().__init__(
            scenario_id=scenario_id,
            name="Deductive Reasoning",
            description="Test logical deduction from premises",
            problem=(
                "Consider these facts:\n"
                "1. All files in the 'config' directory are JSON files.\n"
                "2. 'settings.json' is in the 'config' directory.\n"
                "3. JSON files can be parsed by Python's json module.\n\n"
                "Question: Can settings.json be parsed by Python's json module? "
                "Explain your reasoning step by step."
            ),
            expected_answer="yes",
            tags=["deductive", "logic", "reasoning"],
        )


class InductiveReasoningScenario(ReasoningScenario):
    """Scenario: Test inductive reasoning (pattern recognition)."""

    def __init__(self, scenario_id: str = "inductive_reasoning"):
        super().__init__(
            scenario_id=scenario_id,
            name="Inductive Reasoning",
            description="Test pattern recognition and generalization",
            problem=(
                "I've observed the following pattern in my deployments:\n"
                "- Monday deploy: 2 errors\n"
                "- Tuesday deploy: 1 error\n"
                "- Wednesday deploy: 0 errors\n"
                "- Thursday deploy: 2 errors\n"
                "- Friday deploy: 5 errors\n\n"
                "Based on this pattern, which day seems safest for deployment? "
                "What pattern might explain Friday's high error count? "
                "Reason through this carefully."
            ),
            expected_answer="wednesday",
            tags=["inductive", "pattern", "reasoning"],
        )


class AbductiveReasoningScenario(ReasoningScenario):
    """Scenario: Test abductive reasoning (best explanation)."""

    def __init__(self, scenario_id: str = "abductive_reasoning"):
        super().__init__(
            scenario_id=scenario_id,
            name="Abductive Reasoning",
            description="Test inference to the best explanation",
            problem=(
                "Observations:\n"
                "- The server response time suddenly increased from 50ms to 500ms\n"
                "- This happened at 2:00 PM\n"
                "- CPU usage is normal\n"
                "- Memory usage is normal\n"
                "- A new batch job was scheduled to start at 2:00 PM\n"
                "- Network latency to the database has increased\n\n"
                "What is the most likely explanation for the slow response time? "
                "Why is this explanation more likely than alternatives?"
            ),
            expected_answer=None,  # Multiple valid answers
            tags=["abductive", "diagnosis", "reasoning"],
        )

    async def evaluate(
        self,
        agent: Any,
        context: dict[str, Any],
        trace: EvalTrace,
    ) -> tuple[bool, list[Metric]]:
        metrics = []

        response = ""
        for msg in trace.messages:
            if msg.get("role") == "assistant":
                response += msg.get("content", "")

        response_lower = response.lower()

        # Check if agent considers multiple hypotheses
        hypothesis_words = ["could be", "might be", "possibly", "one explanation", "another explanation", "alternatively"]
        considers_alternatives = sum(1 for w in hypothesis_words if w in response_lower)
        metrics.append(Metric(
            name="considers_alternatives",
            value=considers_alternatives,
            metric_type=MetricType.COUNT,
        ))

        # Check if agent mentions the key evidence (database, batch job)
        mentions_evidence = "database" in response_lower or "batch" in response_lower
        metrics.append(Metric(
            name="mentions_key_evidence",
            value=1 if mentions_evidence else 0,
            metric_type=MetricType.BINARY,
        ))

        # Check for explanation quality
        explanation_words = ["because", "explains", "caused by", "due to", "reason"]
        provides_explanation = any(w in response_lower for w in explanation_words)
        metrics.append(Metric(
            name="provides_explanation",
            value=1 if provides_explanation else 0,
            metric_type=MetricType.BINARY,
        ))

        success = mentions_evidence and provides_explanation
        return success, metrics


class MetaReasoningScenario(ReasoningScenario):
    """Scenario: Test reasoning about own reasoning process."""

    def __init__(self, scenario_id: str = "meta_reasoning"):
        super().__init__(
            scenario_id=scenario_id,
            name="Meta-Reasoning",
            description="Test ability to reason about own reasoning",
            problem=(
                "I want you to solve this problem, but also think about HOW you're thinking:\n\n"
                "Problem: A function is returning None when it should return a list. "
                "The function uses a for loop to build the list.\n\n"
                "1. What's your initial hypothesis?\n"
                "2. What assumptions are you making?\n"
                "3. How confident are you in your diagnosis?\n"
                "4. What would change your mind?"
            ),
            expected_answer=None,
            tags=["meta", "self-reflection", "reasoning"],
        )

    async def evaluate(
        self,
        agent: Any,
        context: dict[str, Any],
        trace: EvalTrace,
    ) -> tuple[bool, list[Metric]]:
        metrics = []

        response = ""
        for msg in trace.messages:
            if msg.get("role") == "assistant":
                response += msg.get("content", "")

        response_lower = response.lower()

        # Check for hypothesis
        has_hypothesis = any(w in response_lower for w in ["hypothesis", "think", "suspect", "guess"])
        metrics.append(Metric(
            name="has_hypothesis",
            value=1 if has_hypothesis else 0,
            metric_type=MetricType.BINARY,
        ))

        # Check for assumption acknowledgment
        has_assumptions = "assum" in response_lower
        metrics.append(Metric(
            name="acknowledges_assumptions",
            value=1 if has_assumptions else 0,
            metric_type=MetricType.BINARY,
        ))

        # Check for confidence expression
        confidence_words = ["confident", "certain", "sure", "likely", "probably"]
        expresses_confidence = any(w in response_lower for w in confidence_words)
        metrics.append(Metric(
            name="expresses_confidence",
            value=1 if expresses_confidence else 0,
            metric_type=MetricType.BINARY,
        ))

        # Check for falsifiability
        falsifiability_words = ["if", "would change", "wrong if", "incorrect if", "evidence"]
        considers_falsifiability = any(w in response_lower for w in falsifiability_words)
        metrics.append(Metric(
            name="considers_falsifiability",
            value=1 if considers_falsifiability else 0,
            metric_type=MetricType.BINARY,
        ))

        # Calculate meta-reasoning score
        meta_score = sum(m.value for m in metrics if m.metric_type == MetricType.BINARY) / 4
        metrics.append(Metric(
            name="meta_reasoning_score",
            value=meta_score,
            metric_type=MetricType.SCORE,
        ))

        success = meta_score >= 0.75
        return success, metrics
