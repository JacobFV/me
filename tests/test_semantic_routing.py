"""Tests for the semantic routing system (focus-based daemon energy allocation)."""

import tempfile
from datetime import datetime, UTC
from pathlib import Path

import pytest

from me.agent.unconscious import (
    UnconsciousDirectory,
    UnconsciousRunner,
    DaemonProfile,
    Focus,
    Pipeline,
    PipelineTrigger,
    TriggerMode,
    Runlevel,
    cosine_similarity,
    install_default_pipelines,
)


# =============================================================================
# Cosine Similarity Tests
# =============================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], []) == 0.0

    def test_different_length_vectors(self):
        a = [1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0


# =============================================================================
# DaemonProfile Tests
# =============================================================================

class TestDaemonProfile:
    def test_create_profile(self):
        profile = DaemonProfile(daemon_name="test-daemon")
        assert profile.daemon_name == "test-daemon"
        assert profile.embedding == []
        assert profile.total_reward == 0.0

    def test_add_activity(self):
        profile = DaemonProfile(daemon_name="test")
        profile.add_activity("prompt 1", "output 1")
        profile.add_activity("prompt 2", "output 2")

        assert len(profile.recent_prompts) == 2
        assert len(profile.recent_outputs) == 2
        assert profile.recent_prompts[-1] == "prompt 2"

    def test_activity_history_bounded(self):
        profile = DaemonProfile(daemon_name="test")
        for i in range(15):
            profile.add_activity(f"prompt {i}", f"output {i}")

        assert len(profile.recent_prompts) == 10
        assert len(profile.recent_outputs) == 10
        assert profile.recent_prompts[0] == "prompt 5"

    def test_get_embedding_text(self):
        profile = DaemonProfile(daemon_name="test-daemon")
        # No activity - returns daemon name
        assert profile.get_embedding_text() == "test-daemon"

        profile.add_activity("test prompt", "test output")
        text = profile.get_embedding_text()
        assert "Prompt:" in text
        assert "Output:" in text


# =============================================================================
# Focus Tests
# =============================================================================

class TestFocus:
    def test_focus_with_embedding_fields(self):
        focus = Focus(
            description="Working on authentication feature",
            embedding=[0.1, 0.2, 0.3],
        )
        assert focus.description == "Working on authentication feature"
        assert len(focus.embedding) == 3
        assert focus.embedding_updated is None

    def test_focus_defaults(self):
        focus = Focus()
        assert focus.description == ""
        assert focus.embedding == []
        assert focus.budget == 1.0


# =============================================================================
# Profile Persistence Tests
# =============================================================================

class TestProfilePersistence:
    def test_save_and_load_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)

            profile = DaemonProfile(
                daemon_name="test-daemon",
                embedding=[0.1, 0.2, 0.3],
                total_reward=1.5,
                weighted_reward=0.75,
            )
            profile.add_activity("prompt", "output")
            udir.save_profile(profile)

            loaded = udir.get_profile("test-daemon")
            assert loaded.daemon_name == "test-daemon"
            assert loaded.embedding == [0.1, 0.2, 0.3]
            assert loaded.total_reward == 1.5
            assert loaded.weighted_reward == 0.75
            assert len(loaded.recent_prompts) == 1

    def test_list_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)

            udir.save_profile(DaemonProfile(daemon_name="daemon1"))
            udir.save_profile(DaemonProfile(daemon_name="daemon2"))

            profiles = udir.list_profiles()
            assert "daemon1" in profiles
            assert "daemon2" in profiles


# =============================================================================
# Template Variable Rendering Tests
# =============================================================================

class TestTemplateRendering:
    def test_render_focus_variable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            # Set focus
            focus = Focus(description="Testing the system")
            udir.save_focus(focus)

            # Test rendering
            template = "Current focus: {focus.description}"
            rendered = runner.render_prompt(template)
            assert rendered == "Current focus: Testing the system"

    def test_render_stream_variable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            # Write a stream
            udir.write_stream("test.md", "Test stream content")

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            template = "Stream says: {streams.test}"
            rendered = runner.render_prompt(template)
            assert rendered == "Stream says: Test stream content"

    def test_render_daemon_variable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            # Create profile with reward history
            profile = DaemonProfile(daemon_name="test-daemon")
            profile.reward_weight_history.append(0.75)
            udir.save_profile(profile)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            template = "My weight: {daemon.reward_weight}"
            rendered = runner.render_prompt(template, daemon_name="test-daemon")
            assert rendered == "My weight: 0.75"

    def test_unresolved_variable_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            # Unknown namespace
            template = "Unknown: {unknown.path}"
            rendered = runner.render_prompt(template)
            assert rendered == "Unknown: {unknown.path}"


# =============================================================================
# Dynamic Budget Allocation Tests
# =============================================================================

class TestDynamicBudgetAllocation:
    def test_budget_scales_with_similarity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            # Set focus with embedding
            runner.update_focus("Working on security analysis")

            # Get a pipeline and its budget
            pipeline = runner._pipelines.get("danger-assessment")
            if pipeline:
                # With no daemon embedding, should get neutral similarity
                similarity = runner._compute_focus_similarity("danger-assessment")
                assert 0 <= similarity <= 1

    def test_no_focus_embedding_returns_full_similarity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            # No focus embedding set
            similarity = runner._compute_focus_similarity("danger-assessment")
            assert similarity == 1.0

    def test_no_daemon_embedding_returns_neutral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            # Set focus with embedding (using the update_focus method)
            runner.update_focus("Test focus")

            # Daemon has no embedding yet
            similarity = runner._compute_focus_similarity("danger-assessment")
            assert similarity == 0.5  # Neutral


# =============================================================================
# Focus Update Tests
# =============================================================================

class TestFocusUpdate:
    def test_update_focus_sets_description(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            focus = runner.update_focus("Implementing new feature")
            assert focus.description == "Implementing new feature"
            assert focus.changed_at is not None

    def test_auto_apply_focus_respects_stability_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            runner = UnconsciousRunner(udir, body_root, Path.cwd())
            runner.load_pipelines()

            # Set focus manually
            runner.update_focus("Manual focus")

            # Try to auto-update immediately (should be skipped)
            result = runner._maybe_apply_focus_update(
                "focus-updater",
                "Auto focus",
                stability_window_minutes=5.0,
            )
            assert result is False

            # Focus should still be manual
            focus = udir.load_focus()
            assert focus.description == "Manual focus"


# =============================================================================
# Pipeline with Template Variables Tests
# =============================================================================

class TestPipelineWithTemplates:
    def test_pipeline_auto_apply_focus_field(self):
        pipeline = Pipeline(
            name="test",
            prompt="Test prompt",
            output="test.md",
            auto_apply_focus=True,
        )
        assert pipeline.auto_apply_focus is True

    def test_default_pipeline_has_auto_apply_focus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body_root = Path(tmpdir)
            udir = UnconsciousDirectory(body_root)
            install_default_pipelines(udir)

            # Check focus-updater has auto_apply_focus=True
            focus_updater = udir.get_pipeline("focus-updater")
            assert focus_updater is not None
            assert focus_updater.auto_apply_focus is True
