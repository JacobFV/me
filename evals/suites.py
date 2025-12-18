"""
Pre-built eval suites.

These combine scenarios into meaningful evaluation sets.
"""

from evals.framework import Eval
from evals.scenarios.task_completion import (
    FileCreationScenario,
    MultiStepTaskScenario,
    GoalDecompositionScenario,
)
from evals.scenarios.tool_use import (
    MemoryToolScenario,
    FileToolScenario,
    ToolSelectionScenario,
    ToolArgumentScenario,
)
from evals.scenarios.self_model import (
    IdentityAwarenessScenario,
    ModelAwarenessScenario,
    CapabilityAwarenessScenario,
    StateConsistencyScenario,
)
from evals.scenarios.reasoning import (
    DeductiveReasoningScenario,
    InductiveReasoningScenario,
    AbductiveReasoningScenario,
    MetaReasoningScenario,
)


def get_basic_suite() -> Eval:
    """Basic eval suite for quick sanity checks."""
    return Eval(
        eval_id="basic",
        name="Basic Capabilities",
        description="Quick sanity check of core agent capabilities",
        scenarios=[
            FileCreationScenario(),
            IdentityAwarenessScenario(),
            DeductiveReasoningScenario(),
        ],
    )


def get_task_completion_suite() -> Eval:
    """Full task completion eval suite."""
    return Eval(
        eval_id="task_completion",
        name="Task Completion",
        description="Evaluate agent's ability to complete various tasks",
        scenarios=[
            FileCreationScenario(),
            MultiStepTaskScenario(),
            GoalDecompositionScenario(),
        ],
    )


def get_tool_use_suite() -> Eval:
    """Full tool use eval suite."""
    return Eval(
        eval_id="tool_use",
        name="Tool Usage",
        description="Evaluate agent's tool selection and usage",
        scenarios=[
            MemoryToolScenario(),
            FileToolScenario(),
            ToolSelectionScenario(),
            ToolArgumentScenario(),
        ],
    )


def get_self_model_suite() -> Eval:
    """Full self-model eval suite."""
    return Eval(
        eval_id="self_model",
        name="Self-Model Accuracy",
        description="Evaluate agent's knowledge of its own state",
        scenarios=[
            IdentityAwarenessScenario(),
            ModelAwarenessScenario(),
            CapabilityAwarenessScenario(),
            StateConsistencyScenario(),
        ],
    )


def get_reasoning_suite() -> Eval:
    """Full reasoning eval suite."""
    return Eval(
        eval_id="reasoning",
        name="Reasoning Capabilities",
        description="Evaluate agent's reasoning abilities",
        scenarios=[
            DeductiveReasoningScenario(),
            InductiveReasoningScenario(),
            AbductiveReasoningScenario(),
            MetaReasoningScenario(),
        ],
    )


def get_full_suite() -> Eval:
    """Comprehensive eval covering all capabilities."""
    return Eval(
        eval_id="full",
        name="Full Evaluation",
        description="Comprehensive evaluation of all agent capabilities",
        scenarios=[
            # Task completion
            FileCreationScenario(),
            MultiStepTaskScenario(),
            GoalDecompositionScenario(),
            # Tool use
            MemoryToolScenario(),
            ToolArgumentScenario(),
            # Self-model
            IdentityAwarenessScenario(),
            ModelAwarenessScenario(),
            CapabilityAwarenessScenario(),
            # Reasoning
            DeductiveReasoningScenario(),
            InductiveReasoningScenario(),
            MetaReasoningScenario(),
        ],
    )


# Registry of available suites
SUITES = {
    "basic": get_basic_suite,
    "task_completion": get_task_completion_suite,
    "tool_use": get_tool_use_suite,
    "self_model": get_self_model_suite,
    "reasoning": get_reasoning_suite,
    "full": get_full_suite,
}


def list_suites() -> list[dict]:
    """List available eval suites."""
    return [
        {
            "id": suite_id,
            "name": getter().name,
            "description": getter().description,
            "scenario_count": len(getter().scenarios),
        }
        for suite_id, getter in SUITES.items()
    ]


def get_suite(suite_id: str) -> Eval | None:
    """Get eval suite by ID."""
    getter = SUITES.get(suite_id)
    return getter() if getter else None
