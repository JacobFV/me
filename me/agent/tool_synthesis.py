"""
Tool Synthesis - Create new tools from detected patterns.

The agent can extend its own capabilities by:
1. Detecting repetitive action patterns
2. Generating reusable scripts/functions
3. Creating tool definitions
4. Testing and validating tools
5. Adding working tools to its toolkit

Philosophy: A truly capable agent doesn't just use tools - it makes them.
The ability to recognize patterns and codify them into reusable tools
is a fundamental form of intelligence. Tools are crystallized procedures.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ToolType(str, Enum):
    """Types of synthesized tools."""
    BASH_SCRIPT = "bash_script"
    PYTHON_FUNCTION = "python_function"
    COMPOSITE = "composite"  # Chains other tools
    ALIAS = "alias"  # Simple command alias
    TEMPLATE = "template"  # Parameterized template


class ToolStatus(str, Enum):
    """Status of a synthesized tool."""
    DRAFT = "draft"
    TESTING = "testing"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ToolParameter(BaseModel):
    """Parameter for a synthesized tool."""
    name: str
    description: str
    param_type: str = "string"
    required: bool = True
    default: Any = None
    choices: list[str] | None = None


class ToolTestCase(BaseModel):
    """Test case for validating a tool."""
    name: str
    inputs: dict[str, Any]
    expected_output: str | None = None
    expected_exit_code: int = 0
    timeout_seconds: int = 30


class ToolTestResult(BaseModel):
    """Result of running a tool test."""
    test_name: str
    passed: bool
    actual_output: str = ""
    actual_exit_code: int = 0
    error: str | None = None
    duration_seconds: float = 0.0


class SynthesizedTool(BaseModel):
    """A tool created by the agent."""
    id: str
    name: str
    description: str
    tool_type: ToolType

    # The tool implementation
    source_code: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    returns: str = "string"

    # Provenance
    source_pattern: str | None = None  # ID of pattern that generated this
    source_episodes: list[str] = Field(default_factory=list)
    created_from: str = ""  # Description of how it was created

    # Status and validation
    status: ToolStatus = ToolStatus.DRAFT
    test_cases: list[ToolTestCase] = Field(default_factory=list)
    test_results: list[ToolTestResult] = Field(default_factory=list)
    validation_score: float = 0.0

    # Usage tracking
    times_used: int = 0
    times_succeeded: int = 0
    times_failed: int = 0
    success_rate: float = 0.0
    avg_duration: float = 0.0

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = 1
    tags: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


class PatternToToolSpec(BaseModel):
    """Specification for converting a pattern to a tool."""
    pattern_description: str
    actions: list[str]
    parameters: list[ToolParameter]
    tool_type: ToolType
    suggested_name: str
    suggested_description: str


class ToolSynthesizer:
    """
    Synthesizes tools from patterns and specifications.

    The synthesizer can:
    1. Convert action sequences to bash scripts
    2. Generate Python functions from patterns
    3. Create composite tools that chain others
    4. Generate test cases for validation
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.tools_dir = body_dir / "tools" / "synthesized"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir = self.tools_dir / "scripts"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_bash_script(
        self,
        name: str,
        description: str,
        commands: list[str],
        parameters: list[ToolParameter] | None = None,
        source_pattern: str | None = None,
    ) -> SynthesizedTool:
        """Synthesize a bash script from commands."""
        parameters = parameters or []

        # Generate script
        lines = ["#!/bin/bash", f"# {description}", "set -e", ""]

        # Add parameter handling
        if parameters:
            lines.append("# Parameters")
            for i, param in enumerate(parameters):
                if param.required:
                    lines.append(f'{param.name}="${{{i+1}:?{param.name} is required}}"')
                else:
                    default = param.default or ""
                    lines.append(f'{param.name}="${{{i+1}:-{default}}}"')
            lines.append("")

        # Add commands
        lines.append("# Main")
        for cmd in commands:
            # Substitute parameter references
            for param in parameters:
                cmd = cmd.replace(f"${{{param.name}}}", f"${param.name}")
                cmd = cmd.replace(f"<{param.name}>", f"${param.name}")
            lines.append(cmd)

        source_code = "\n".join(lines)

        tool = SynthesizedTool(
            id=hashlib.md5(f"{name}:{source_code}".encode()).hexdigest()[:12],
            name=name,
            description=description,
            tool_type=ToolType.BASH_SCRIPT,
            source_code=source_code,
            parameters=parameters,
            source_pattern=source_pattern,
            created_from="synthesized from command sequence",
        )

        self._save_tool(tool)
        return tool

    def synthesize_python_function(
        self,
        name: str,
        description: str,
        body: str,
        parameters: list[ToolParameter] | None = None,
        imports: list[str] | None = None,
        source_pattern: str | None = None,
    ) -> SynthesizedTool:
        """Synthesize a Python function."""
        parameters = parameters or []
        imports = imports or []

        # Generate function
        lines = []

        # Imports
        if imports:
            for imp in imports:
                lines.append(f"import {imp}")
            lines.append("")

        # Function signature
        params_str = ", ".join(
            f"{p.name}: {p.param_type}" + (f" = {repr(p.default)}" if not p.required else "")
            for p in parameters
        )
        lines.append(f"def {name}({params_str}):")
        lines.append(f'    """{description}"""')

        # Body (indent)
        for line in body.split("\n"):
            lines.append(f"    {line}" if line.strip() else "")

        source_code = "\n".join(lines)

        tool = SynthesizedTool(
            id=hashlib.md5(f"{name}:{source_code}".encode()).hexdigest()[:12],
            name=name,
            description=description,
            tool_type=ToolType.PYTHON_FUNCTION,
            source_code=source_code,
            parameters=parameters,
            source_pattern=source_pattern,
            created_from="synthesized Python function",
        )

        self._save_tool(tool)
        return tool

    def synthesize_composite_tool(
        self,
        name: str,
        description: str,
        tool_chain: list[str],
        data_flow: dict[str, str] | None = None,
    ) -> SynthesizedTool:
        """Synthesize a tool that chains other tools."""
        data_flow = data_flow or {}

        # Generate composite definition
        definition = {
            "chain": tool_chain,
            "data_flow": data_flow,
        }

        source_code = json.dumps(definition, indent=2)

        tool = SynthesizedTool(
            id=hashlib.md5(f"{name}:{source_code}".encode()).hexdigest()[:12],
            name=name,
            description=description,
            tool_type=ToolType.COMPOSITE,
            source_code=source_code,
            created_from="composite tool chain",
        )

        self._save_tool(tool)
        return tool

    def generate_test_cases(self, tool: SynthesizedTool) -> list[ToolTestCase]:
        """Generate test cases for a tool."""
        test_cases = []

        # Basic execution test
        basic_inputs = {}
        for param in tool.parameters:
            if param.required:
                if param.choices:
                    basic_inputs[param.name] = param.choices[0]
                elif param.param_type == "string":
                    basic_inputs[param.name] = "test_value"
                elif param.param_type == "int":
                    basic_inputs[param.name] = 1
                elif param.param_type == "bool":
                    basic_inputs[param.name] = True

        test_cases.append(ToolTestCase(
            name="basic_execution",
            inputs=basic_inputs,
            expected_exit_code=0,
        ))

        # Missing required parameter test
        if any(p.required for p in tool.parameters):
            test_cases.append(ToolTestCase(
                name="missing_required",
                inputs={},
                expected_exit_code=1,  # Should fail
            ))

        return test_cases

    def _save_tool(self, tool: SynthesizedTool) -> None:
        """Save tool to disk."""
        # Save metadata
        path = self.tools_dir / f"{tool.id}.json"
        path.write_text(json.dumps(tool.model_dump(mode='json'), indent=2, default=str))

        # Save script if applicable
        if tool.tool_type == ToolType.BASH_SCRIPT:
            script_path = self.scripts_dir / f"{tool.name}.sh"
            script_path.write_text(tool.source_code)
            script_path.chmod(0o755)
        elif tool.tool_type == ToolType.PYTHON_FUNCTION:
            script_path = self.scripts_dir / f"{tool.name}.py"
            script_path.write_text(tool.source_code)

    def get_tool(self, tool_id: str) -> SynthesizedTool | None:
        """Get a tool by ID."""
        path = self.tools_dir / f"{tool_id}.json"
        if not path.exists():
            return None
        try:
            return SynthesizedTool.model_validate(json.loads(path.read_text()))
        except Exception:
            return None

    def list_tools(self, status: ToolStatus | None = None) -> list[SynthesizedTool]:
        """List synthesized tools."""
        tools = []
        for path in self.tools_dir.glob("*.json"):
            try:
                tool = SynthesizedTool.model_validate(json.loads(path.read_text()))
                if status is None or tool.status == status:
                    tools.append(tool)
            except Exception:
                continue
        return tools


class ToolValidator:
    """
    Validates synthesized tools through testing.

    Runs test cases, checks outputs, and determines
    if a tool is safe and working.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.synthesizer = ToolSynthesizer(body_dir)

    def validate_tool(self, tool_id: str) -> tuple[bool, list[ToolTestResult]]:
        """
        Validate a tool by running its test cases.

        Returns (passed, results).
        """
        tool = self.synthesizer.get_tool(tool_id)
        if not tool:
            return False, []

        # Generate test cases if none exist
        if not tool.test_cases:
            tool.test_cases = self.synthesizer.generate_test_cases(tool)

        results = []
        all_passed = True

        for test_case in tool.test_cases:
            result = self._run_test(tool, test_case)
            results.append(result)
            if not result.passed:
                all_passed = False

        # Update tool status and results
        tool.test_results = results
        tool.validation_score = sum(1 for r in results if r.passed) / len(results) if results else 0
        tool.status = ToolStatus.VALIDATED if all_passed else ToolStatus.FAILED
        tool.updated_at = datetime.now(UTC)

        self.synthesizer._save_tool(tool)

        return all_passed, results

    def _run_test(self, tool: SynthesizedTool, test_case: ToolTestCase) -> ToolTestResult:
        """Run a single test case."""
        import time
        start_time = time.time()

        try:
            if tool.tool_type == ToolType.BASH_SCRIPT:
                return self._run_bash_test(tool, test_case, start_time)
            elif tool.tool_type == ToolType.PYTHON_FUNCTION:
                return self._run_python_test(tool, test_case, start_time)
            else:
                return ToolTestResult(
                    test_name=test_case.name,
                    passed=False,
                    error=f"Unsupported tool type: {tool.tool_type}",
                )
        except Exception as e:
            return ToolTestResult(
                test_name=test_case.name,
                passed=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _run_bash_test(
        self,
        tool: SynthesizedTool,
        test_case: ToolTestCase,
        start_time: float,
    ) -> ToolTestResult:
        """Run a bash script test."""
        import time

        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(tool.source_code)
            script_path = f.name

        try:
            # Build command with arguments
            args = [script_path]
            for param in tool.parameters:
                if param.name in test_case.inputs:
                    args.append(str(test_case.inputs[param.name]))

            # Run script
            result = subprocess.run(
                ["bash"] + args,
                capture_output=True,
                text=True,
                timeout=test_case.timeout_seconds,
            )

            passed = result.returncode == test_case.expected_exit_code
            if test_case.expected_output and passed:
                passed = test_case.expected_output in result.stdout

            return ToolTestResult(
                test_name=test_case.name,
                passed=passed,
                actual_output=result.stdout[:1000],
                actual_exit_code=result.returncode,
                error=result.stderr[:500] if result.returncode != 0 else None,
                duration_seconds=time.time() - start_time,
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

    def _run_python_test(
        self,
        tool: SynthesizedTool,
        test_case: ToolTestCase,
        start_time: float,
    ) -> ToolTestResult:
        """Run a Python function test."""
        import time

        # Create test script
        test_script = f"""
{tool.source_code}

# Run test
import json
import sys

try:
    result = {tool.name}(**{test_case.inputs})
    print(json.dumps({{"result": str(result), "success": True}}))
except Exception as e:
    print(json.dumps({{"error": str(e), "success": False}}))
    sys.exit(1)
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name

        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=test_case.timeout_seconds,
            )

            passed = result.returncode == test_case.expected_exit_code

            return ToolTestResult(
                test_name=test_case.name,
                passed=passed,
                actual_output=result.stdout[:1000],
                actual_exit_code=result.returncode,
                error=result.stderr[:500] if result.returncode != 0 else None,
                duration_seconds=time.time() - start_time,
            )
        finally:
            Path(script_path).unlink(missing_ok=True)


class ToolDeployer:
    """
    Deploys validated tools for agent use.

    Makes tools available in the agent's toolkit and
    generates appropriate tool definitions.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.synthesizer = ToolSynthesizer(body_dir)
        self.deployed_dir = body_dir / "tools" / "deployed"
        self.deployed_dir.mkdir(parents=True, exist_ok=True)

    def deploy_tool(self, tool_id: str) -> dict[str, Any] | None:
        """
        Deploy a validated tool.

        Returns tool definition for adding to agent tools.
        """
        tool = self.synthesizer.get_tool(tool_id)
        if not tool or tool.status != ToolStatus.VALIDATED:
            return None

        # Generate tool definition
        tool_def = {
            "name": f"synth_{tool.name}",
            "description": f"[Synthesized] {tool.description}",
            "input_schema": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.param_type,
                        "description": param.description,
                    }
                    for param in tool.parameters
                },
                "required": [p.name for p in tool.parameters if p.required],
            },
            "synthesized_tool_id": tool.id,
            "tool_type": tool.tool_type.value,
        }

        # Update tool status
        tool.status = ToolStatus.DEPLOYED
        tool.updated_at = datetime.now(UTC)
        self.synthesizer._save_tool(tool)

        # Save deployment info
        deploy_path = self.deployed_dir / f"{tool.id}.json"
        deploy_path.write_text(json.dumps(tool_def, indent=2))

        return tool_def

    def get_deployed_tools(self) -> list[dict[str, Any]]:
        """Get all deployed tool definitions."""
        tools = []
        for path in self.deployed_dir.glob("*.json"):
            try:
                tools.append(json.loads(path.read_text()))
            except Exception:
                continue
        return tools

    def record_usage(
        self,
        tool_id: str,
        success: bool,
        duration_seconds: float,
    ) -> None:
        """Record usage of a deployed tool."""
        tool = self.synthesizer.get_tool(tool_id)
        if not tool:
            return

        tool.times_used += 1
        if success:
            tool.times_succeeded += 1
        else:
            tool.times_failed += 1

        tool.success_rate = tool.times_succeeded / tool.times_used
        n = tool.times_used
        tool.avg_duration = (tool.avg_duration * (n-1) + duration_seconds) / n

        self.synthesizer._save_tool(tool)


class ToolSynthesisSystem:
    """
    Unified system for tool synthesis and management.

    Coordinates pattern detection, synthesis, validation, and deployment.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.synthesizer = ToolSynthesizer(body_dir)
        self.validator = ToolValidator(body_dir)
        self.deployer = ToolDeployer(body_dir)

    def synthesize_from_commands(
        self,
        name: str,
        description: str,
        commands: list[str],
        parameters: list[dict[str, Any]] | None = None,
        auto_validate: bool = True,
        auto_deploy: bool = False,
    ) -> SynthesizedTool:
        """
        Full pipeline: synthesize, validate, optionally deploy.
        """
        # Convert parameter dicts to ToolParameter
        params = []
        if parameters:
            for p in parameters:
                params.append(ToolParameter(**p))

        # Synthesize
        tool = self.synthesizer.synthesize_bash_script(
            name=name,
            description=description,
            commands=commands,
            parameters=params,
        )

        # Validate
        if auto_validate:
            passed, _ = self.validator.validate_tool(tool.id)

            # Deploy if validation passed
            if auto_deploy and passed:
                self.deployer.deploy_tool(tool.id)

        return tool

    def synthesize_from_pattern(
        self,
        spec: PatternToToolSpec,
        auto_validate: bool = True,
    ) -> SynthesizedTool:
        """Synthesize a tool from a pattern specification."""
        if spec.tool_type == ToolType.BASH_SCRIPT:
            tool = self.synthesizer.synthesize_bash_script(
                name=spec.suggested_name,
                description=spec.suggested_description,
                commands=spec.actions,
                parameters=spec.parameters,
                source_pattern=spec.pattern_description,
            )
        elif spec.tool_type == ToolType.PYTHON_FUNCTION:
            # Convert actions to Python body
            body = "\n".join(spec.actions)
            tool = self.synthesizer.synthesize_python_function(
                name=spec.suggested_name,
                description=spec.suggested_description,
                body=body,
                parameters=spec.parameters,
                source_pattern=spec.pattern_description,
            )
        else:
            raise ValueError(f"Unsupported tool type for synthesis: {spec.tool_type}")

        if auto_validate:
            self.validator.validate_tool(tool.id)

        return tool

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Get all tools available for use."""
        return self.deployer.get_deployed_tools()

    def get_tool_summary(self) -> str:
        """Get a summary of synthesized tools."""
        lines = ["# Synthesized Tools\n"]

        tools = self.synthesizer.list_tools()

        by_status = {}
        for tool in tools:
            status = tool.status.value
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(tool)

        for status, status_tools in by_status.items():
            lines.append(f"## {status.title()} ({len(status_tools)})")
            for tool in status_tools[:5]:
                success_info = ""
                if tool.times_used > 0:
                    success_info = f" [{tool.success_rate:.0%} success]"
                lines.append(f"- **{tool.name}**: {tool.description[:50]}...{success_info}")
            if len(status_tools) > 5:
                lines.append(f"  *+{len(status_tools) - 5} more*")
            lines.append("")

        return "\n".join(lines)
