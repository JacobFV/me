"""
Evaluation framework for the me agent.

Evals measure agent capabilities, not code correctness:
- Task completion: Can the agent achieve goals?
- Tool use: Does the agent use tools appropriately?
- Self-model: Does the agent accurately know its own state?
- Reasoning: Can the agent solve problems requiring inference?
- Efficiency: How many tokens/steps/time does the agent use?

Key principle: Evals measure what matters for agent usefulness,
not just what's easy to measure. Interpret results carefully.
"""

from evals.framework import (
    Eval,
    EvalResult,
    EvalRunner,
    EvalReport,
    Scenario,
    Metric,
)
from evals.scenarios import (
    TaskCompletionScenario,
    ToolUseScenario,
    SelfModelScenario,
    ReasoningScenario,
)

__all__ = [
    "Eval",
    "EvalResult",
    "EvalRunner",
    "EvalReport",
    "Scenario",
    "Metric",
    "TaskCompletionScenario",
    "ToolUseScenario",
    "SelfModelScenario",
    "ReasoningScenario",
]
