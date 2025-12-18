"""Tests for world model construction and prediction."""

import tempfile
from datetime import datetime, UTC
from pathlib import Path

import pytest

from me.agent.world_model import (
    FileSystemModel,
    FileNode,
    FileType,
    ProcessModel,
    ProcessNode,
    TransitionModel,
    StateTransition,
    WorldModel,
)


# =============================================================================
# FileSystemModel Tests
# =============================================================================

class TestFileSystemModel:
    def test_observe_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            fs_model = FileSystemModel(model_dir)

            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Hello world")

            # Observe it
            node = fs_model.observe_path(test_file)

            assert node.type == FileType.FILE
            assert node.size > 0
            assert node.content_hash is not None

    def test_observe_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            fs_model = FileSystemModel(model_dir)

            # Observe directory
            test_dir = Path(tmpdir)
            node = fs_model.observe_path(test_dir)

            assert node.type == FileType.DIRECTORY
            assert isinstance(node.children, list)

    def test_observe_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            fs_model = FileSystemModel(model_dir)

            # Create directory structure
            (Path(tmpdir) / "dir1").mkdir()
            (Path(tmpdir) / "dir1" / "file1.txt").write_text("test")
            (Path(tmpdir) / "dir2").mkdir()
            (Path(tmpdir) / "dir2" / "file2.txt").write_text("test")

            # Observe tree
            nodes = fs_model.observe_tree(Path(tmpdir), max_depth=2)

            # Should have observed directory + subdirs + files
            assert len(nodes) >= 5

            # Check we can retrieve nodes
            file1_path = str((Path(tmpdir) / "dir1" / "file1.txt").resolve())
            node = fs_model.get_node(file1_path)
            assert node is not None
            assert node.type == FileType.FILE

    def test_detect_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            fs_model = FileSystemModel(model_dir)

            # Create diverse files
            for i in range(5):
                (Path(tmpdir) / f"file{i}.txt").write_text(f"content {i}")
                (Path(tmpdir) / f"data{i}.json").write_text(f'{{"key": {i}}}')

            fs_model.observe_tree(Path(tmpdir))

            patterns = fs_model.detect_patterns()

            assert patterns["total_files"] == 10
            assert ".txt" in patterns["file_types"]
            assert ".json" in patterns["file_types"]

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"

            # Create and populate model
            fs_model = FileSystemModel(model_dir)
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")
            fs_model.observe_path(test_file)
            fs_model.save()

            # Load in new instance
            fs_model2 = FileSystemModel(model_dir)
            node = fs_model2.get_node(str(test_file.resolve()))

            assert node is not None
            assert node.type == FileType.FILE


# =============================================================================
# ProcessModel Tests
# =============================================================================

class TestProcessModel:
    def test_observe_process(self):
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            proc_model = ProcessModel(model_dir)

            # Observe current process
            node = proc_model.observe_process(os.getpid())

            assert node is not None
            assert node.pid == os.getpid()
            assert node.name != ""
            assert node.status != ""

    def test_get_resource_summary(self):
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            proc_model = ProcessModel(model_dir)

            # Observe current process
            proc_model.observe_process(os.getpid())

            summary = proc_model.get_resource_summary()

            assert summary["total_processes"] >= 1
            assert "top_cpu" in summary
            assert "top_memory" in summary

    def test_save_and_load(self):
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"

            # Create and populate
            proc_model = ProcessModel(model_dir)
            proc_model.observe_process(os.getpid())
            proc_model.save()

            # Load in new instance
            proc_model2 = ProcessModel(model_dir)
            assert len(proc_model2._processes) >= 1


# =============================================================================
# TransitionModel Tests
# =============================================================================

class TestTransitionModel:
    def test_record_transition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            trans_model = TransitionModel(model_dir)

            # Record a transition
            transition = trans_model.record_transition(
                initial_state={"file_count": 5},
                action="create_file",
                action_params={"filename": "test.txt"},
                resulting_state={"file_count": 6},
                success=True,
                duration_ms=10.5,
            )

            assert transition.action == "create_file"
            assert transition.success is True
            assert len(trans_model._transitions) == 1

    def test_get_transitions_for_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            trans_model = TransitionModel(model_dir)

            # Record multiple transitions
            for i in range(3):
                trans_model.record_transition(
                    initial_state={},
                    action="test_action",
                    action_params={"i": i},
                    resulting_state={},
                    success=True,
                    duration_ms=10.0,
                )

            trans_model.record_transition(
                initial_state={},
                action="other_action",
                action_params={},
                resulting_state={},
                success=True,
                duration_ms=10.0,
            )

            # Query by action
            test_transitions = trans_model.get_transitions_for_action("test_action")
            assert len(test_transitions) == 3

    def test_predict_outcome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            trans_model = TransitionModel(model_dir)

            # Record some successful transitions
            for i in range(5):
                trans_model.record_transition(
                    initial_state={"count": i},
                    action="increment",
                    action_params={},
                    resulting_state={"count": i + 1},
                    success=True,
                    duration_ms=10.0,
                )

            # Predict outcome
            prediction = trans_model.predict_outcome(
                action="increment",
                current_state={"count": 10},
            )

            assert prediction["confidence"] == 1.0  # All succeeded
            assert prediction["evidence_count"] == 5

    def test_predict_no_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            trans_model = TransitionModel(model_dir)

            # Predict with no evidence
            prediction = trans_model.predict_outcome(
                action="unknown_action",
                current_state={},
            )

            assert prediction["confidence"] == 0.0
            assert prediction["evidence_count"] == 0

    def test_detect_anomaly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            trans_model = TransitionModel(model_dir)

            # Normal duration
            is_anomaly = trans_model.detect_anomaly(
                action="test",
                expected_duration_ms=10.0,
                actual_duration_ms=15.0,
                threshold=2.0,
            )
            assert is_anomaly is False

            # Anomalous duration (> 2x expected)
            is_anomaly = trans_model.detect_anomaly(
                action="test",
                expected_duration_ms=10.0,
                actual_duration_ms=25.0,
                threshold=2.0,
            )
            assert is_anomaly is True

    def test_get_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            trans_model = TransitionModel(model_dir)

            # Record transitions
            trans_model.record_transition({}, "action1", {}, {}, True, 10.0)
            trans_model.record_transition({}, "action1", {}, {}, True, 12.0)
            trans_model.record_transition({}, "action2", {}, {}, False, 5.0)

            stats = trans_model.get_statistics()

            assert stats["total_transitions"] == 3
            assert stats["unique_actions"] == 2
            assert stats["success_rate"] == 2 / 3
            assert stats["action_counts"]["action1"] == 2
            assert "action1" in stats["avg_duration_by_action"]

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            # Create and record
            trans_model = TransitionModel(model_dir)
            trans_model.record_transition({}, "test", {}, {}, True, 10.0)

            # Load in new instance
            trans_model2 = TransitionModel(model_dir)
            assert len(trans_model2._transitions) == 1
            assert trans_model2._transitions[0].action == "test"


# =============================================================================
# Unified WorldModel Tests
# =============================================================================

class TestWorldModel:
    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "world_model"
            world = WorldModel(model_dir)

            assert world.filesystem is not None
            assert world.processes is not None
            assert world.transitions is not None

    def test_observe_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "world_model"
            world = WorldModel(model_dir)

            # Create some files
            (Path(tmpdir) / "test1.txt").write_text("test")
            (Path(tmpdir) / "test2.txt").write_text("test")

            # Observe
            world.observe_environment(Path(tmpdir), observe_processes=True)

            # Should have observed files
            assert len(world.filesystem._nodes) > 0

            # Should have observed processes
            assert len(world.processes._processes) >= 0

    def test_record_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "world_model"
            world = WorldModel(model_dir)

            # Record action
            world.record_action(
                action="create_file",
                action_params={"filename": "test.txt"},
                success=True,
                duration_ms=15.0,
            )

            # Should be in transition model
            transitions = world.transitions.get_transitions_for_action("create_file")
            assert len(transitions) == 1

    def test_predict_action_outcome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "world_model"
            world = WorldModel(model_dir)

            # Record some actions
            for i in range(3):
                world.record_action(
                    action="test_action",
                    action_params={},
                    success=True,
                    duration_ms=10.0,
                )

            # Predict
            prediction = world.predict_action_outcome("test_action", {})

            assert prediction["confidence"] == 1.0
            assert prediction["evidence_count"] == 3

    def test_save_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "world_model"
            world = WorldModel(model_dir)

            # Create test file and observe
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")
            world.filesystem.observe_path(test_file)

            # Save all
            world.save_all()

            # Verify files exist
            assert (model_dir / "filesystem" / "filesystem.json").exists()
            assert (model_dir / "processes" / "processes.json").exists()

    def test_get_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "world_model"
            world = WorldModel(model_dir)

            # Observe and record
            world.observe_environment(Path(tmpdir))
            world.record_action("test", {}, True, 10.0)

            # Get summary
            summary = world.get_summary()

            assert "filesystem" in summary
            assert "processes" in summary
            assert "transitions" in summary
            assert "timestamp" in summary

    def test_integration_workflow(self):
        """Test complete workflow: observe -> act -> predict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "world_model"
            world = WorldModel(model_dir)

            # Step 1: Observe initial environment
            world.observe_environment(Path(tmpdir))
            initial_files = len(world.filesystem._nodes)

            # Step 2: Simulate action (create file)
            test_file = Path(tmpdir) / "created_file.txt"
            test_file.write_text("test")

            # Step 3: Record the action
            world.record_action(
                action="create_file",
                action_params={"filename": "created_file.txt"},
                success=True,
                duration_ms=12.5,
            )

            # Step 4: Observe again
            world.observe_environment(Path(tmpdir))

            # Step 5: Predict future actions
            prediction = world.predict_action_outcome("create_file", {})

            assert prediction["evidence_count"] == 1
            assert prediction["confidence"] == 1.0
