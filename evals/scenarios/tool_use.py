"""
Tool use eval scenarios.

Measures: Does the agent use tools appropriately?

Key metrics:
- tool_selection: Did the agent choose the right tool? (binary)
- tool_efficiency: Minimal tools used for task (ratio)
- tool_correctness: Were tool arguments valid? (ratio)
- unnecessary_calls: Tools called that weren't needed (count)

Interpretation notes:
- Using MORE tools isn't better - efficiency matters
- Wrong tool with right result is still concerning
- Tool call order can indicate reasoning quality
- Failed tool calls may be exploration vs mistakes
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


class ToolUseScenario(Scenario):
    """
    Base class for tool use scenarios.

    Evaluates whether agents select and use tools appropriately.
    """

    def __init__(
        self,
        scenario_id: str,
        name: str,
        description: str,
        task_prompt: str,
        expected_tools: list[str],
        forbidden_tools: list[str] | None = None,
        tags: list[str] | None = None,
    ):
        super().__init__(scenario_id, name, description, tags)
        self.task_prompt = task_prompt
        self.expected_tools = expected_tools
        self.forbidden_tools = forbidden_tools or []

    async def setup(self, agent: Any) -> dict[str, Any]:
        work_dir = tempfile.mkdtemp(prefix="eval_tool_")
        return {"work_dir": Path(work_dir)}

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

        # Analyze tool usage
        tools_used = [tc["name"] for tc in trace.tool_calls]
        unique_tools = set(tools_used)

        # Expected tools used
        expected_used = [t for t in self.expected_tools if t in unique_tools]
        expected_ratio = len(expected_used) / len(self.expected_tools) if self.expected_tools else 1.0
        metrics.append(Metric(
            name="expected_tools_ratio",
            value=expected_ratio,
            metric_type=MetricType.RATIO,
            metadata={"expected": self.expected_tools, "used": list(unique_tools)},
        ))

        # Forbidden tools used (should be 0)
        forbidden_used = [t for t in self.forbidden_tools if t in unique_tools]
        metrics.append(Metric(
            name="forbidden_tools_used",
            value=len(forbidden_used),
            metric_type=MetricType.COUNT,
            metadata={"tools": forbidden_used},
        ))

        # Total tool calls (efficiency)
        metrics.append(Metric(
            name="total_tool_calls",
            value=len(tools_used),
            metric_type=MetricType.COUNT,
        ))

        # Unique tools (diversity)
        metrics.append(Metric(
            name="unique_tools",
            value=len(unique_tools),
            metric_type=MetricType.COUNT,
        ))

        # Redundant calls (same tool called multiple times)
        redundant = len(tools_used) - len(unique_tools)
        metrics.append(Metric(
            name="redundant_calls",
            value=redundant,
            metric_type=MetricType.COUNT,
        ))

        # Success: used expected tools, avoided forbidden ones
        success = expected_ratio >= 0.8 and len(forbidden_used) == 0

        return success, metrics

    async def teardown(self, agent: Any, context: dict[str, Any]):
        import shutil
        work_dir = context.get("work_dir")
        if work_dir and work_dir.exists():
            shutil.rmtree(work_dir)


class MemoryToolScenario(ToolUseScenario):
    """Scenario: Agent should use memory tools appropriately."""

    def __init__(self, scenario_id: str = "memory_tool_use"):
        super().__init__(
            scenario_id=scenario_id,
            name="Memory Tool Usage",
            description="Agent should store and recall information using memory tools",
            task_prompt=(
                "I want you to remember that my favorite color is blue. "
                "Store this information so you can recall it later. "
                "Then, demonstrate that you can recall this information."
            ),
            expected_tools=["store_memory", "recall_memories"],
            tags=["memory", "tools"],
        )


class FileToolScenario(ToolUseScenario):
    """Scenario: Agent should use file tools correctly."""

    def __init__(self, scenario_id: str = "file_tool_use"):
        super().__init__(
            scenario_id=scenario_id,
            name="File Tool Usage",
            description="Agent should use file operations correctly",
            task_prompt=(
                "Create a file called 'notes.txt' with the content 'Important notes'. "
                "Then read the file back to verify its contents."
            ),
            expected_tools=["run_command"],  # For file ops via bash or similar
            tags=["files", "tools"],
        )


class ToolSelectionScenario(ToolUseScenario):
    """Scenario: Agent must choose the right tool for the task."""

    def __init__(self, scenario_id: str = "tool_selection"):
        super().__init__(
            scenario_id=scenario_id,
            name="Tool Selection",
            description="Agent must select appropriate tools for task",
            task_prompt=(
                "I need you to find all Python files in the current directory. "
                "Use the most appropriate tool for this task."
            ),
            expected_tools=["run_command"],  # Should use ls/find
            forbidden_tools=[],  # Could list tools that would be wrong
            tags=["selection", "tools"],
        )


class ToolArgumentScenario(Scenario):
    """Scenario: Evaluate quality of tool arguments."""

    def __init__(self, scenario_id: str = "tool_arguments"):
        super().__init__(
            scenario_id=scenario_id,
            name="Tool Argument Quality",
            description="Evaluate if agent provides correct tool arguments",
            tags=["arguments", "tools"],
        )
        self.task_prompt = (
            "Store the following in memory: 'Meeting at 3pm tomorrow'. "
            "Use appropriate tags like 'meeting', 'schedule'."
        )

    async def setup(self, agent: Any) -> dict[str, Any]:
        return {}

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

        # Find store_memory calls
        memory_calls = [
            tc for tc in trace.tool_calls
            if tc["name"] == "store_memory"
        ]

        if not memory_calls:
            metrics.append(Metric(
                name="memory_tool_used",
                value=0,
                metric_type=MetricType.BINARY,
            ))
            return False, metrics

        metrics.append(Metric(
            name="memory_tool_used",
            value=1,
            metric_type=MetricType.BINARY,
        ))

        # Check argument quality
        for call in memory_calls:
            args = call.get("args", {})

            # Check content provided
            has_content = bool(args.get("content"))
            metrics.append(Metric(
                name="has_content_arg",
                value=1 if has_content else 0,
                metric_type=MetricType.BINARY,
            ))

            # Check tags provided
            tags = args.get("tags", "")
            has_tags = bool(tags and len(tags.strip()) > 0)
            metrics.append(Metric(
                name="has_tags_arg",
                value=1 if has_tags else 0,
                metric_type=MetricType.BINARY,
            ))

            # Check if tags are relevant
            if has_tags:
                tag_list = [t.strip().lower() for t in tags.split(",")]
                relevant_tags = [t for t in tag_list if t in ["meeting", "schedule", "reminder", "calendar"]]
                metrics.append(Metric(
                    name="relevant_tags",
                    value=len(relevant_tags),
                    metric_type=MetricType.COUNT,
                ))

        success = any(
            m.name == "has_content_arg" and m.value == 1
            for m in metrics
        )

        return success, metrics
