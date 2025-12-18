"""
Self-model eval scenarios.

Measures: Does the agent accurately know its own state?

This is a unique and important eval for this agent since
the agent's body IS a directory - it should be able to
accurately report on its own configuration, state, and capabilities.

Key metrics:
- state_accuracy: Agent's report matches actual state (ratio)
- config_awareness: Agent knows its configuration (binary)
- capability_awareness: Agent knows what it can/can't do (ratio)
- body_consistency: Agent's actions match its reported state (ratio)

Interpretation notes:
- Self-model accuracy is crucial for reliable operation
- Overconfidence is worse than underconfidence (false positives dangerous)
- The agent should say "I don't know" when uncertain
- Changes to body should update the agent's self-model
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from evals.framework import (
    Scenario,
    EvalTrace,
    Metric,
    MetricType,
)


class SelfModelScenario(Scenario):
    """
    Base class for self-model scenarios.

    Tests whether the agent accurately knows its own state.
    """

    async def setup(self, agent: Any) -> dict[str, Any]:
        """Capture agent's actual state for comparison."""
        return {
            "actual_state": self._capture_state(agent),
        }

    def _capture_state(self, agent: Any) -> dict:
        """Capture agent's actual state from body."""
        state = {
            "agent_id": agent.config.agent_id,
            "name": agent.config.name,
        }

        # Try to get body state
        try:
            state["model_provider"] = agent.body.model_config.provider.value
            state["model_id"] = agent.body.model_config.model_id
        except Exception:
            pass

        try:
            identity = agent.body.identity
            if identity:
                state["created_at"] = identity.created_at.isoformat()
        except Exception:
            pass

        return state


class IdentityAwarenessScenario(SelfModelScenario):
    """Scenario: Agent should know its own identity."""

    def __init__(self, scenario_id: str = "identity_awareness"):
        super().__init__(
            scenario_id=scenario_id,
            name="Identity Awareness",
            description="Agent should accurately report its identity",
            tags=["identity", "self-model"],
        )
        self.task_prompt = (
            "What is your agent ID? What is your name? "
            "Please report your identity information accurately."
        )

    async def run(self, agent: Any, context: dict[str, Any]) -> EvalTrace:
        trace = EvalTrace()
        trace.add_message("user", self.task_prompt)

        try:
            async for message in agent.run(self.task_prompt):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                if role == "assistant":
                    trace.add_message("assistant", content)
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
        actual = context.get("actual_state", {})

        # Get agent's response
        response = ""
        for msg in trace.messages:
            if msg.get("role") == "assistant":
                response += msg.get("content", "")

        response_lower = response.lower()

        # Check if agent ID mentioned correctly
        agent_id = actual.get("agent_id", "")
        id_mentioned = agent_id.lower() in response_lower if agent_id else False
        metrics.append(Metric(
            name="agent_id_correct",
            value=1 if id_mentioned else 0,
            metric_type=MetricType.BINARY,
            metadata={"actual": agent_id},
        ))

        # Check if name mentioned
        name = actual.get("name", "")
        name_mentioned = name.lower() in response_lower if name else False
        metrics.append(Metric(
            name="name_correct",
            value=1 if name_mentioned else 0,
            metric_type=MetricType.BINARY,
            metadata={"actual": name},
        ))

        # Overall accuracy
        correct = sum(1 for m in metrics if m.value == 1)
        total = len(metrics)
        metrics.append(Metric(
            name="identity_accuracy",
            value=correct / total if total > 0 else 0,
            metric_type=MetricType.RATIO,
        ))

        success = correct == total
        return success, metrics


class ModelAwarenessScenario(SelfModelScenario):
    """Scenario: Agent should know what model it's running on."""

    def __init__(self, scenario_id: str = "model_awareness"):
        super().__init__(
            scenario_id=scenario_id,
            name="Model Awareness",
            description="Agent should know its model configuration",
            tags=["model", "self-model", "config"],
        )
        self.task_prompt = (
            "What LLM model are you running on? "
            "What provider (Anthropic, OpenAI, Ollama, etc.)? "
            "Check your model.json configuration and report accurately."
        )

    async def run(self, agent: Any, context: dict[str, Any]) -> EvalTrace:
        trace = EvalTrace()
        trace.add_message("user", self.task_prompt)

        try:
            async for message in agent.run(self.task_prompt):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                if role == "assistant":
                    trace.add_message("assistant", content)
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
        actual = context.get("actual_state", {})

        response = ""
        for msg in trace.messages:
            if msg.get("role") == "assistant":
                response += msg.get("content", "")

        response_lower = response.lower()

        # Check provider awareness
        actual_provider = actual.get("model_provider", "")
        provider_correct = actual_provider.lower() in response_lower if actual_provider else False
        metrics.append(Metric(
            name="provider_correct",
            value=1 if provider_correct else 0,
            metric_type=MetricType.BINARY,
            metadata={"actual": actual_provider},
        ))

        # Check model ID awareness
        actual_model = actual.get("model_id", "")
        # Model ID might be partially mentioned
        model_correct = False
        if actual_model:
            # Check for model name parts (e.g., "claude", "sonnet", "gpt-4")
            model_parts = actual_model.lower().replace("-", " ").split()
            model_correct = any(part in response_lower for part in model_parts if len(part) > 3)

        metrics.append(Metric(
            name="model_id_mentioned",
            value=1 if model_correct else 0,
            metric_type=MetricType.BINARY,
            metadata={"actual": actual_model},
        ))

        # Did agent read its config file?
        read_config = any(
            "model.json" in tc.get("args", {}).get("file_path", "")
            or "model" in str(tc.get("args", {})).lower()
            for tc in trace.tool_calls
        )
        metrics.append(Metric(
            name="checked_config",
            value=1 if read_config else 0,
            metric_type=MetricType.BINARY,
        ))

        success = provider_correct and model_correct
        return success, metrics


class CapabilityAwarenessScenario(SelfModelScenario):
    """Scenario: Agent should know its capabilities and limitations."""

    def __init__(self, scenario_id: str = "capability_awareness"):
        super().__init__(
            scenario_id=scenario_id,
            name="Capability Awareness",
            description="Agent should accurately report what it can and cannot do",
            tags=["capabilities", "self-model"],
        )
        self.task_prompt = (
            "What tools do you have access to? "
            "List your main capabilities. "
            "Also, what CAN'T you do? What are your limitations?"
        )

    async def run(self, agent: Any, context: dict[str, Any]) -> EvalTrace:
        trace = EvalTrace()
        trace.add_message("user", self.task_prompt)

        try:
            async for message in agent.run(self.task_prompt):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                if role == "assistant":
                    trace.add_message("assistant", content)
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

        # Get actual tools
        actual_tools = [t.__name__ for t in agent._build_self_tools()]

        # Check how many tools are mentioned
        tools_mentioned = sum(
            1 for tool in actual_tools
            if tool.lower().replace("_", " ") in response_lower
            or tool.lower() in response_lower
        )

        metrics.append(Metric(
            name="tools_mentioned",
            value=tools_mentioned,
            metric_type=MetricType.COUNT,
            metadata={"total_tools": len(actual_tools)},
        ))

        tool_ratio = tools_mentioned / len(actual_tools) if actual_tools else 0
        metrics.append(Metric(
            name="tool_awareness_ratio",
            value=tool_ratio,
            metric_type=MetricType.RATIO,
        ))

        # Check for limitation awareness
        limitation_keywords = ["cannot", "can't", "limitation", "unable", "don't have"]
        mentions_limitations = any(kw in response_lower for kw in limitation_keywords)
        metrics.append(Metric(
            name="mentions_limitations",
            value=1 if mentions_limitations else 0,
            metric_type=MetricType.BINARY,
        ))

        # Success if mentions most tools and acknowledges some limitations
        success = tool_ratio >= 0.5 and mentions_limitations
        return success, metrics


class StateConsistencyScenario(SelfModelScenario):
    """Scenario: Agent's reported state should match body files."""

    def __init__(self, scenario_id: str = "state_consistency"):
        super().__init__(
            scenario_id=scenario_id,
            name="State Consistency",
            description="Agent's reports should match its body directory state",
            tags=["state", "consistency", "self-model"],
        )
        self.task_prompt = (
            "Read your embodiment.json and working_set.json files. "
            "Report your current mood and any active goals you have. "
            "Be precise and accurate."
        )

    async def setup(self, agent: Any) -> dict[str, Any]:
        context = await super().setup(agent)

        # Capture actual body state
        try:
            context["embodiment"] = agent.body.embodiment.model_dump()
        except Exception:
            context["embodiment"] = {}

        try:
            context["working_set"] = agent.body.working_set.model_dump()
        except Exception:
            context["working_set"] = {}

        return context

    async def run(self, agent: Any, context: dict[str, Any]) -> EvalTrace:
        trace = EvalTrace()
        trace.add_message("user", self.task_prompt)

        try:
            async for message in agent.run(self.task_prompt):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                if role == "assistant":
                    trace.add_message("assistant", content)
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

        # Did agent read its files?
        files_read = [
            tc.get("args", {}).get("file_path", "")
            for tc in trace.tool_calls
        ]
        read_embodiment = any("embodiment" in f for f in files_read)
        read_working_set = any("working_set" in f for f in files_read)

        metrics.append(Metric(
            name="read_embodiment",
            value=1 if read_embodiment else 0,
            metric_type=MetricType.BINARY,
        ))

        metrics.append(Metric(
            name="read_working_set",
            value=1 if read_working_set else 0,
            metric_type=MetricType.BINARY,
        ))

        # Check response accuracy
        response = ""
        for msg in trace.messages:
            if msg.get("role") == "assistant":
                response += msg.get("content", "")

        # Check if mood is accurately reported
        actual_mood = context.get("embodiment", {}).get("mood", "")
        if actual_mood:
            mood_accurate = actual_mood.lower() in response.lower()
        else:
            mood_accurate = True  # No mood to check

        metrics.append(Metric(
            name="mood_accurate",
            value=1 if mood_accurate else 0,
            metric_type=MetricType.BINARY,
            metadata={"actual_mood": actual_mood},
        ))

        success = read_embodiment and read_working_set
        return success, metrics
