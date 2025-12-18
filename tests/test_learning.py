"""Tests for experience replay and learning system."""

import tempfile
import uuid
from datetime import datetime, UTC
from pathlib import Path

import pytest

from me.agent.learning import (
    ExperienceBuffer,
    Experience,
    ExperienceType,
    PredictionTracker,
    Prediction,
    ProcedureUpdater,
    ProcedureStats,
    PatternExtractor,
    Pattern,
    TheoryRefiner,
    Evidence,
    ExperienceReplay,
)


# =============================================================================
# ExperienceBuffer Tests
# =============================================================================

class TestExperienceBuffer:
    def test_add_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = ExperienceBuffer(Path(tmpdir), max_size=100)

            exp = Experience(
                id=str(uuid.uuid4()),
                type=ExperienceType.ACTION,
                content="test action",
                reward=1.0,
            )
            buffer.add(exp)

            recent = buffer.get_recent(10)
            assert len(recent) == 1
            assert recent[0].content == "test action"

    def test_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = ExperienceBuffer(Path(tmpdir), max_size=100)

            for i in range(20):
                exp = Experience(
                    id=str(uuid.uuid4()),
                    type=ExperienceType.ACTION,
                    content=f"action_{i}",
                    reward=float(i) / 10,
                )
                buffer.add(exp)

            sampled = buffer.sample(5)
            assert len(sampled) == 5

    def test_prioritized_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = ExperienceBuffer(Path(tmpdir), max_size=100)

            # Add low reward experiences
            for i in range(10):
                exp = Experience(
                    id=str(uuid.uuid4()),
                    type=ExperienceType.ACTION,
                    content=f"low_{i}",
                    reward=0.1,
                )
                buffer.add(exp)

            # Add high reward experiences
            for i in range(5):
                exp = Experience(
                    id=str(uuid.uuid4()),
                    type=ExperienceType.SUCCESS,
                    content=f"high_{i}",
                    reward=1.0,
                )
                buffer.add(exp)

            # Prioritized sampling should favor high reward
            sampled = buffer.sample(10, prioritized=True)
            high_count = sum(1 for e in sampled if "high" in e.content)
            # High reward should appear more often than 1/3 of samples
            assert high_count >= 2

    def test_get_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = ExperienceBuffer(Path(tmpdir))

            buffer.add(Experience(id="1", type=ExperienceType.ACTION, content="action"))
            buffer.add(Experience(id="2", type=ExperienceType.ERROR, content="error"))
            buffer.add(Experience(id="3", type=ExperienceType.SUCCESS, content="success"))

            actions = buffer.get_by_type(ExperienceType.ACTION)
            assert len(actions) == 1
            assert actions[0].type == ExperienceType.ACTION

    def test_get_by_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = ExperienceBuffer(Path(tmpdir))

            buffer.add(Experience(id="1", type=ExperienceType.ACTION, content="a1", episode_id="ep1"))
            buffer.add(Experience(id="2", type=ExperienceType.ACTION, content="a2", episode_id="ep1"))
            buffer.add(Experience(id="3", type=ExperienceType.ACTION, content="a3", episode_id="ep2"))

            ep1_exps = buffer.get_by_episode("ep1")
            assert len(ep1_exps) == 2

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = ExperienceBuffer(Path(tmpdir))

            buffer.add(Experience(id="1", type=ExperienceType.ACTION, content="test"))
            assert len(buffer.get_recent(10)) == 1

            buffer.clear()
            assert len(buffer.get_recent(10)) == 0


# =============================================================================
# PredictionTracker Tests
# =============================================================================

class TestPredictionTracker:
    def test_record_and_resolve_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PredictionTracker(Path(tmpdir))

            pred = Prediction(
                id="pred1",
                action="run_tests",
                predicted_outcome="tests_pass",
                confidence=0.8,
            )
            tracker.record_prediction(pred)

            unresolved = tracker.get_unresolved()
            assert len(unresolved) == 1

            tracker.record_outcome("pred1", "tests_pass", error_magnitude=0.1)

            unresolved = tracker.get_unresolved()
            assert len(unresolved) == 0

    def test_accuracy_by_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PredictionTracker(Path(tmpdir))

            # Good predictions
            for i in range(3):
                pred = Prediction(
                    id=f"good_{i}",
                    action="action_a",
                    predicted_outcome="success",
                    confidence=0.9,
                )
                tracker.record_prediction(pred)
                tracker.record_outcome(f"good_{i}", "success", error_magnitude=0.1)

            # Bad predictions
            for i in range(2):
                pred = Prediction(
                    id=f"bad_{i}",
                    action="action_b",
                    predicted_outcome="success",
                    confidence=0.9,
                )
                tracker.record_prediction(pred)
                tracker.record_outcome(f"bad_{i}", "failure", error_magnitude=0.8)

            accuracy = tracker.get_accuracy_by_action()
            assert accuracy["action_a"] == 1.0
            assert accuracy["action_b"] == 0.0

    def test_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PredictionTracker(Path(tmpdir))

            # High confidence, correct
            for i in range(5):
                pred = Prediction(
                    id=f"high_{i}",
                    action="test",
                    predicted_outcome="x",
                    confidence=0.9,
                )
                tracker.record_prediction(pred)
                tracker.record_outcome(f"high_{i}", "x", error_magnitude=0.1)

            calibration = tracker.get_calibration()
            assert "90-100" in calibration


# =============================================================================
# ProcedureUpdater Tests
# =============================================================================

class TestProcedureUpdater:
    def test_record_use(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            updater = ProcedureUpdater(Path(tmpdir))

            updater.record_use("proc1", True, 100.0, 0.5, "context1")
            updater.record_use("proc1", True, 120.0, 0.6, "context1")
            updater.record_use("proc1", False, 80.0, -0.1, "context2", "error occurred")

            success_rate = updater.get_success_rate("proc1")
            assert success_rate == 2 / 3

    def test_recommended_procedures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            updater = ProcedureUpdater(Path(tmpdir))

            # Good procedure
            for i in range(5):
                updater.record_use("good_proc", True, 100.0, 0.5, "my_context")

            # Bad procedure
            for i in range(5):
                updater.record_use("bad_proc", False, 100.0, -0.1, "my_context")

            recommendations = updater.get_recommended_procedures("my_context")
            assert len(recommendations) > 0
            assert recommendations[0][0] == "good_proc"

    def test_procedures_to_deprecate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            updater = ProcedureUpdater(Path(tmpdir))

            # Failing procedure
            for i in range(10):
                updater.record_use("failing_proc", False, 100.0, -0.1, "context")

            to_deprecate = updater.get_procedures_to_deprecate(min_uses=5, max_success_rate=0.2)
            assert "failing_proc" in to_deprecate


# =============================================================================
# PatternExtractor Tests
# =============================================================================

class TestPatternExtractor:
    def test_extract_success_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = PatternExtractor(Path(tmpdir))

            experiences = [
                Experience(id="1", type=ExperienceType.SUCCESS, content="achieved goal", step_number=5),
                Experience(id="2", type=ExperienceType.SUCCESS, content="achieved goal", step_number=10),
                Experience(id="3", type=ExperienceType.ACTION, content="took action", step_number=4),
                Experience(id="4", type=ExperienceType.ACTION, content="took action", step_number=9),
            ]

            patterns = extractor.extract_from_experiences(experiences)

            success_patterns = [p for p in patterns if p.pattern_type == "success_pattern"]
            assert len(success_patterns) == 1

    def test_extract_failure_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = PatternExtractor(Path(tmpdir))

            experiences = [
                Experience(id="1", type=ExperienceType.ERROR, content="error 1"),
                Experience(id="2", type=ExperienceType.ERROR, content="error 2"),
            ]

            patterns = extractor.extract_from_experiences(experiences)

            failure_patterns = [p for p in patterns if p.pattern_type == "failure_pattern"]
            assert len(failure_patterns) == 1

    def test_get_patterns_for_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = PatternExtractor(Path(tmpdir))

            # Create pattern with conditions
            pattern = Pattern(
                id="p1",
                name="test_pattern",
                description="test",
                pattern_type="success_pattern",
                conditions=["key1", "key2"],
            )
            extractor._merge_and_save([pattern])

            relevant = extractor.get_patterns_for_context({"key1": "value1", "other": "x"})
            assert len(relevant) == 1


# =============================================================================
# TheoryRefiner Tests
# =============================================================================

class TestTheoryRefiner:
    def test_add_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            refiner = TheoryRefiner(Path(tmpdir))

            evidence = Evidence(
                id="e1",
                theory_id="theory1",
                supports=True,
                description="supports the theory",
                strength=0.8,
            )
            refiner.add_evidence(evidence)

            theory_evidence = refiner.get_evidence_for_theory("theory1")
            assert len(theory_evidence) == 1

    def test_theory_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            refiner = TheoryRefiner(Path(tmpdir))

            # Add supporting evidence
            for i in range(3):
                refiner.add_evidence(Evidence(
                    id=f"support_{i}",
                    theory_id="theory1",
                    supports=True,
                    description="supports",
                    strength=0.8,
                ))

            # Add contradicting evidence
            refiner.add_evidence(Evidence(
                id="contradict_1",
                theory_id="theory1",
                supports=False,
                description="contradicts",
                strength=0.5,
            ))

            confidence = refiner.get_theory_confidence("theory1")
            assert confidence > 0.5  # More support than contradiction

    def test_contradicted_theories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            refiner = TheoryRefiner(Path(tmpdir))

            # Add more contradicting than supporting
            refiner.add_evidence(Evidence(id="s1", theory_id="bad_theory", supports=True, description="", strength=0.5))
            refiner.add_evidence(Evidence(id="c1", theory_id="bad_theory", supports=False, description="", strength=0.5))
            refiner.add_evidence(Evidence(id="c2", theory_id="bad_theory", supports=False, description="", strength=0.5))

            contradicted = refiner.get_contradicted_theories()
            assert "bad_theory" in contradicted


# =============================================================================
# ExperienceReplay Integration Tests
# =============================================================================

class TestExperienceReplay:
    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            replay = ExperienceReplay(Path(tmpdir))

            assert replay.buffer is not None
            assert replay.predictions is not None
            assert replay.procedures is not None
            assert replay.patterns is not None
            assert replay.theories is not None

    def test_record_experience(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            replay = ExperienceReplay(Path(tmpdir))

            exp = Experience(
                id="test1",
                type=ExperienceType.ACTION,
                content="test action",
            )
            replay.record_experience(exp)

            recent = replay.buffer.get_recent(10)
            assert len(recent) == 1

    def test_replay_and_learn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            replay = ExperienceReplay(Path(tmpdir))

            # Add some experiences
            for i in range(10):
                exp = Experience(
                    id=f"exp_{i}",
                    type=ExperienceType.SUCCESS if i % 2 == 0 else ExperienceType.ACTION,
                    content=f"experience {i}",
                    reward=0.5 if i % 2 == 0 else 0.1,
                )
                replay.record_experience(exp)

            result = replay.replay_and_learn(batch_size=10)

            assert result["learned"] is True
            assert result["experiences_replayed"] <= 10

    def test_suggest_improvements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            replay = ExperienceReplay(Path(tmpdir))

            # Add some procedure uses
            replay.record_procedure_use("proc1", True, 100.0, 0.5, "coding")
            replay.record_procedure_use("proc1", True, 110.0, 0.6, "coding")

            suggestions = replay.suggest_improvements("coding")

            assert "recommended_procedures" in suggestions
            assert "avoid_procedures" in suggestions

    def test_get_learning_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            replay = ExperienceReplay(Path(tmpdir))

            # Add some data
            replay.record_experience(Experience(id="1", type=ExperienceType.ACTION, content="test"))
            replay.record_procedure_use("proc1", True, 100.0, 0.5, "context")

            summary = replay.get_learning_summary()

            assert "buffer_size" in summary
            assert "patterns_discovered" in summary
            assert "procedure_stats" in summary
