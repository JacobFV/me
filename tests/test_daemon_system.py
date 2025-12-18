"""Tests for the Agentic OS daemon system."""

import asyncio
import json
import tempfile
from datetime import datetime, UTC
from pathlib import Path

import pytest

from me.agent.unconscious import (
    Daemon,
    DaemonState,
    Runlevel,
    ProcessGroup,
    TokenBudget,
    JournalEntry,
    LogLevel,
    Pipeline,
    PipelineTrigger,
    PipelineSource,
    TriggerMode,
    UnconsciousDirectory,
    UnconsciousRunner,
    DEFAULT_PIPELINES,
    DEFAULT_GROUPS,
    install_default_pipelines,
)


# =============================================================================
# TokenBudget Tests
# =============================================================================

class TestTokenBudget:
    def test_can_run_within_limits(self):
        budget = TokenBudget(
            max_tokens_per_run=1000,
            max_tokens_per_step=5000,
            max_tokens_per_hour=50000,
        )
        assert budget.can_run(500) is True
        assert budget.can_run(1000) is True
        assert budget.can_run(1001) is False  # Exceeds per-run

    def test_deduct_tokens(self):
        budget = TokenBudget()
        budget.deduct(500)
        assert budget.used_this_step == 500
        assert budget.used_this_hour == 500

    def test_reset_step(self):
        budget = TokenBudget()
        budget.deduct(500)
        budget.reset_step()
        assert budget.used_this_step == 0
        assert budget.used_this_hour == 500  # Hour not reset

    def test_budget_exhaustion(self):
        budget = TokenBudget(max_tokens_per_step=1000)
        budget.deduct(800)
        assert budget.can_run(500) is False  # Would exceed step limit


# =============================================================================
# ProcessGroup Tests
# =============================================================================

class TestProcessGroup:
    def test_create_group(self):
        group = ProcessGroup(
            name="test-group",
            description="Test group",
            priority=3,
            max_concurrent=2,
        )
        assert group.name == "test-group"
        assert group.priority == 3
        assert group.max_concurrent == 2
        assert group.enabled is True

    def test_default_groups_exist(self):
        assert "perception" in DEFAULT_GROUPS
        assert "abstraction" in DEFAULT_GROUPS
        assert "maintenance" in DEFAULT_GROUPS
        assert "default" in DEFAULT_GROUPS


# =============================================================================
# Daemon Tests
# =============================================================================

class TestDaemon:
    def test_create_daemon(self):
        daemon = Daemon(name="test-daemon")
        assert daemon.name == "test-daemon"
        assert daemon.state == DaemonState.STOPPED
        assert daemon.pid == 0
        assert daemon.restarts == 0

    def test_daemon_states(self):
        assert DaemonState.STOPPED.value == "stopped"
        assert DaemonState.RUNNING.value == "running"
        assert DaemonState.FAILED.value == "failed"
        assert DaemonState.DISABLED.value == "disabled"


# =============================================================================
# Runlevel Tests
# =============================================================================

class TestRunlevel:
    def test_runlevel_ordering(self):
        levels = [Runlevel.HALT, Runlevel.MINIMAL, Runlevel.NORMAL, Runlevel.FULL]
        assert levels[0] == Runlevel.HALT
        assert levels[-1] == Runlevel.FULL


# =============================================================================
# JournalEntry Tests
# =============================================================================

class TestJournalEntry:
    def test_create_entry(self):
        entry = JournalEntry(
            daemon="test-daemon",
            level=LogLevel.INFO,
            message="Test message",
            metadata={"key": "value"},
        )
        assert entry.daemon == "test-daemon"
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"

    def test_jsonl_serialization(self):
        entry = JournalEntry(
            daemon="test",
            level=LogLevel.WARN,
            message="Warning",
        )
        jsonl = entry.to_jsonl()
        parsed = JournalEntry.from_jsonl(jsonl)
        assert parsed.daemon == entry.daemon
        assert parsed.level == entry.level
        assert parsed.message == entry.message


# =============================================================================
# Pipeline Tests (with new fields)
# =============================================================================

class TestPipelineWithDaemonFields:
    def test_pipeline_with_group(self):
        pipeline = Pipeline(
            name="test-pipeline",
            prompt="Test prompt",
            output="test.md",
            group="perception",
            runlevel=Runlevel.MINIMAL,
            boot_order=10,
        )
        assert pipeline.group == "perception"
        assert pipeline.runlevel == Runlevel.MINIMAL
        assert pipeline.boot_order == 10

    def test_pipeline_dependencies(self):
        pipeline = Pipeline(
            name="dependent",
            prompt="Depends on other",
            output="dep.md",
            depends_on=["situation-summary"],
            feeds_into=["next-step"],
        )
        assert "situation-summary" in pipeline.depends_on
        assert "next-step" in pipeline.feeds_into

    def test_default_pipelines_have_groups(self):
        for pipeline in DEFAULT_PIPELINES:
            assert pipeline.group is not None
            assert pipeline.runlevel is not None


# =============================================================================
# UnconsciousDirectory Tests
# =============================================================================

class TestUnconsciousDirectory:
    def test_daemon_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)

            # Save daemon
            daemon = Daemon(name="test", state=DaemonState.RUNNING, pid=42)
            udir.save_daemon(daemon)

            # Load and verify
            loaded = udir.get_daemon("test")
            assert loaded is not None
            assert loaded.state == DaemonState.RUNNING
            assert loaded.pid == 42

    def test_group_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)

            # Save group
            group = ProcessGroup(name="test-group", priority=1)
            udir.save_group(group)

            # Load and verify
            loaded = udir.get_group("test-group")
            assert loaded is not None
            assert loaded.priority == 1

    def test_journal_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)

            # Write entries
            udir.log("daemon1", LogLevel.INFO, "Started")
            udir.log("daemon2", LogLevel.WARN, "Warning message")

            # Query all
            entries = udir.query_journal()
            assert len(entries) == 2

            # Query by daemon
            entries = udir.query_journal(daemon="daemon1")
            assert len(entries) == 1
            assert entries[0].message == "Started"

    def test_budget_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)

            # Save budget
            budget = TokenBudget(max_tokens_per_step=1000)
            budget.deduct(500)
            udir.update_budget("test", budget)

            # Load and verify
            loaded = udir.get_budget("test")
            assert loaded is not None
            assert loaded.used_this_step == 500


# =============================================================================
# UnconsciousRunner Tests
# =============================================================================

class TestUnconsciousRunner:
    def test_runner_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            runner = UnconsciousRunner(udir, body_root, Path.cwd())

            assert runner._runlevel == Runlevel.HALT
            assert runner._booted is False

    def test_load_pipelines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)

            # Install defaults
            install_default_pipelines(udir)

            # Create runner and load
            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            assert len(runner._pipelines) == 5  # Including focus-updater
            assert "danger-assessment" in runner._pipelines
            assert "focus-updater" in runner._pipelines
            assert len(runner._daemons) == 5
            assert len(runner._groups) > 0

    @pytest.mark.asyncio
    async def test_daemon_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            # Start daemon
            success = await runner.start_daemon("danger-assessment")
            assert success is True

            daemon = runner._daemons.get("danger-assessment")
            assert daemon is not None
            assert daemon.state == DaemonState.RUNNING
            assert daemon.pid > 0

            # Stop daemon
            success = await runner.stop_daemon("danger-assessment")
            assert success is True
            assert daemon.state == DaemonState.STOPPED

    @pytest.mark.asyncio
    async def test_boot_and_shutdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            # Boot to MINIMAL - only danger-assessment should start
            await runner.boot(Runlevel.MINIMAL)
            assert runner._runlevel == Runlevel.MINIMAL
            assert runner._booted is True

            # danger-assessment has runlevel=MINIMAL
            daemon = runner._daemons.get("danger-assessment")
            assert daemon.state == DaemonState.RUNNING

            # situation-summary has runlevel=NORMAL, should not be started
            daemon = runner._daemons.get("situation-summary")
            assert daemon.state != DaemonState.RUNNING

            # Shutdown
            await runner.shutdown()
            assert runner._runlevel == Runlevel.HALT
            assert runner._booted is False

    def test_dependency_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            # Test topological sort
            pipelines = ["next-step-prediction", "situation-summary"]
            ordered = runner._resolve_run_order(pipelines)

            # situation-summary should come before next-step-prediction (dependency)
            idx_situation = ordered.index("situation-summary")
            idx_next = ordered.index("next-step-prediction")
            assert idx_situation < idx_next

    def test_list_daemons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            daemons = runner.list_daemons()
            assert len(daemons) == 5  # Including focus-updater
            assert all("name" in d for d in daemons)
            assert all("state" in d for d in daemons)
            assert all("group" in d for d in daemons)


# =============================================================================
# Integration Tests
# =============================================================================

class TestDaemonSystemIntegration:
    @pytest.mark.asyncio
    async def test_full_boot_cycle(self):
        """Test complete boot -> run -> shutdown cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            # Boot to NORMAL
            await runner.boot(Runlevel.NORMAL)

            # Check running daemons
            running = [
                name for name, d in runner._daemons.items()
                if d.state == DaemonState.RUNNING
            ]
            # At NORMAL, danger-assessment and situation-summary should run
            assert "danger-assessment" in running
            assert "situation-summary" in running

            # Change to FULL
            await runner.change_runlevel(Runlevel.FULL)

            # Now self-reflection should also be running
            running = [
                name for name, d in runner._daemons.items()
                if d.state == DaemonState.RUNNING
            ]
            assert "self-reflection" in running

            # Shutdown
            await runner.shutdown()

            # All should be stopped
            stopped = [
                name for name, d in runner._daemons.items()
                if d.state == DaemonState.STOPPED
            ]
            assert len(stopped) == 5  # Including focus-updater

    def test_budget_enforcement(self):
        """Test that budget limits are enforced."""
        budget = TokenBudget(
            max_tokens_per_run=100,
            max_tokens_per_step=500,
            max_tokens_per_hour=10000,
        )

        # Can run initially
        assert budget.can_run(100) is True

        # Deduct tokens
        budget.deduct(400)

        # Now limited by step budget (500 - 400 = 100 remaining)
        assert budget.can_run(100) is True
        assert budget.can_run(150) is False

        # Reset step
        budget.reset_step()
        assert budget.can_run(100) is True

    def test_journal_rotation(self):
        """Test that journal entries are rotated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)

            # Write many entries
            for i in range(50):
                udir.log("test", LogLevel.INFO, f"Message {i}")

            # Query should return most recent
            entries = udir.query_journal(limit=10)
            assert len(entries) == 10
            assert "Message 49" in entries[0].message
