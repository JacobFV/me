"""Tests for meta-cognitive monitoring system."""

import tempfile
from datetime import datetime, UTC
from pathlib import Path

import pytest

from me.agent.metacognition import (
    PerformanceMonitor,
    PerformanceSnapshot,
    CognitiveMetric,
    StuckDetector,
    StuckDetection,
    StuckType,
    StrategyTracker,
    Strategy,
    AdaptationEngine,
    Adaptation,
    NoveltyDetector,
    MetaCognitiveMonitor,
)


# =============================================================================
# PerformanceMonitor Tests
# =============================================================================

class TestPerformanceMonitor:
    def test_record_and_get_current(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))

            monitor.record(CognitiveMetric.TASK_COMPLETION_RATE, 0.8)
            monitor.record(CognitiveMetric.TASK_COMPLETION_RATE, 0.9)

            current = monitor.get_current(CognitiveMetric.TASK_COMPLETION_RATE)
            assert current == 0.9

    def test_get_average(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))

            for value in [0.8, 0.9, 0.7, 0.8]:
                monitor.record("test_metric", value)

            avg = monitor.get_average("test_metric")
            assert abs(avg - 0.8) < 0.01

    def test_get_trend_improving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))

            # Start low, end high
            for i in range(20):
                monitor.record("improving_metric", 0.5 + i * 0.02)

            trend = monitor.get_trend("improving_metric")
            assert trend == "improving"

    def test_get_trend_degrading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))

            # Start high, end low
            for i in range(20):
                monitor.record("degrading_metric", 0.9 - i * 0.02)

            trend = monitor.get_trend("degrading_metric")
            assert trend == "degrading"

    def test_get_trend_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))

            # Consistent values
            for _ in range(20):
                monitor.record("stable_metric", 0.8)

            trend = monitor.get_trend("stable_metric")
            assert trend == "stable"

    def test_detect_anomaly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))

            # Values with some variance
            import random
            random.seed(42)
            for _ in range(20):
                monitor.record("normal_metric", 0.8 + random.uniform(-0.05, 0.05))

            # Normal value (within distribution)
            assert not monitor.detect_anomaly("normal_metric", 0.82, std_threshold=2.0)

            # Anomalous value (way outside distribution)
            assert monitor.detect_anomaly("normal_metric", 0.1, std_threshold=2.0)

    def test_get_all_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))

            monitor.record("metric_a", 0.8)
            monitor.record("metric_b", 0.9)

            all_metrics = monitor.get_all_metrics()
            assert "metric_a" in all_metrics
            assert "metric_b" in all_metrics


# =============================================================================
# StuckDetector Tests
# =============================================================================

class TestStuckDetector:
    def test_detect_action_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = StuckDetector(Path(tmpdir))

            # Record repeated action
            for _ in range(10):
                detector.record_action("same_action")

            detections = detector.check_all()
            loop_detections = [d for d in detections if d.stuck_type == StuckType.LOOP]
            assert len(loop_detections) >= 1

    def test_detect_oscillation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = StuckDetector(Path(tmpdir))

            # Record oscillating states
            for i in range(20):
                state = "state_A" if i % 2 == 0 else "state_B"
                detector.record_state(state)

            detections = detector.check_all()
            oscillation_detections = [d for d in detections if d.stuck_type == StuckType.OSCILLATION]
            assert len(oscillation_detections) >= 1

    def test_detect_stagnation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = StuckDetector(Path(tmpdir))

            # Record same state repeatedly
            for _ in range(15):
                detector.record_state("unchanging_state")

            detections = detector.check_all()
            stagnation_detections = [d for d in detections if d.stuck_type == StuckType.STAGNATION]
            assert len(stagnation_detections) >= 1

    def test_detect_error_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = StuckDetector(Path(tmpdir))

            # Record repeated error
            for _ in range(5):
                detector.record_error("FileNotFoundError: /tmp/missing.txt")

            detections = detector.check_all()
            error_loop_detections = [d for d in detections if d.stuck_type == StuckType.ERROR_LOOP]
            assert len(error_loop_detections) >= 1

    def test_no_false_positive_with_variety(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = StuckDetector(Path(tmpdir))

            # Record varied actions
            for i in range(10):
                detector.record_action(f"action_{i}")
                detector.record_state(f"state_{i}")

            detections = detector.check_all()
            # Should not detect stuck states with variety
            assert len(detections) == 0


# =============================================================================
# StrategyTracker Tests
# =============================================================================

class TestStrategyTracker:
    def test_register_and_use_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = StrategyTracker(Path(tmpdir))

            strategy = tracker.register_strategy("divide_conquer", "Break problems into smaller parts")

            tracker.record_use(strategy.id, success=True, reward=0.8, duration_ms=100.0, context="coding")
            tracker.record_use(strategy.id, success=True, reward=0.9, duration_ms=120.0, context="coding")

            stats = tracker.get_all_statistics()
            assert stats["total_uses"] == 2

    def test_get_best_for_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = StrategyTracker(Path(tmpdir))

            # Good strategy for coding
            good = tracker.register_strategy("good_strategy", "Works well")
            for _ in range(5):
                tracker.record_use(good.id, success=True, reward=0.9, duration_ms=100.0, context="coding")

            # Bad strategy
            bad = tracker.register_strategy("bad_strategy", "Doesn't work")
            for _ in range(5):
                tracker.record_use(bad.id, success=False, reward=-0.5, duration_ms=100.0, context="coding")

            best = tracker.get_best_for_context("coding")
            assert len(best) >= 1
            assert best[0].name == "good_strategy"

    def test_get_struggling_strategies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = StrategyTracker(Path(tmpdir))

            struggling = tracker.register_strategy("struggling", "Poor performance")
            for _ in range(10):
                tracker.record_use(struggling.id, success=False, reward=-0.5, duration_ms=100.0, context="test")

            struggling_list = tracker.get_struggling_strategies(min_uses=5, max_success_rate=0.3)
            assert len(struggling_list) >= 1


# =============================================================================
# NoveltyDetector Tests
# =============================================================================

class TestNoveltyDetector:
    def test_record_novel_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = NoveltyDetector(Path(tmpdir))

            is_novel1 = detector.record_action("new_action")
            assert is_novel1 is True

            is_novel2 = detector.record_action("new_action")
            assert is_novel2 is False

    def test_get_novelty_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = NoveltyDetector(Path(tmpdir))

            # All unique actions (use different words, not just numbers)
            actions = ["read_file", "write_data", "run_tests", "commit_changes",
                      "deploy_app", "check_status", "fetch_logs", "update_config",
                      "restart_server", "validate_input"]
            for action in actions:
                detector.record_action(action)

            rate = detector.get_novelty_rate(window=10)
            assert rate == 1.0

    def test_low_novelty_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = NoveltyDetector(Path(tmpdir))

            # Repeated action
            for _ in range(10):
                detector.record_action("same_action")

            rate = detector.get_novelty_rate(window=10)
            assert rate == 0.1  # Only first was novel

    def test_suggest_exploration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = NoveltyDetector(Path(tmpdir))

            # Low novelty (repeated actions)
            for _ in range(20):
                detector.record_action("repeated")

            should_explore = detector.suggest_exploration()
            assert should_explore is True


# =============================================================================
# AdaptationEngine Tests
# =============================================================================

class TestAdaptationEngine:
    def test_suggest_adaptation_for_stuck(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            performance = PerformanceMonitor(Path(tmpdir) / "perf")
            stuck = StuckDetector(Path(tmpdir) / "stuck")
            strategies = StrategyTracker(Path(tmpdir) / "strat")
            engine = AdaptationEngine(Path(tmpdir) / "adapt", performance, stuck, strategies)

            # Create stuck state (action loop)
            for _ in range(10):
                stuck.record_action("stuck_action")

            # Run check_all to detect the stuck state
            detections = stuck.check_all()
            # Should detect loop
            assert len(detections) >= 1

            # Now suggest adaptations based on detected stuck states
            adaptations = engine.suggest_adaptations()
            # Should suggest adaptation for loop
            assert len(adaptations) >= 1

    def test_suggest_adaptation_for_degrading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            performance = PerformanceMonitor(Path(tmpdir) / "perf")
            stuck = StuckDetector(Path(tmpdir) / "stuck")
            strategies = StrategyTracker(Path(tmpdir) / "strat")
            engine = AdaptationEngine(Path(tmpdir) / "adapt", performance, stuck, strategies)

            # Create degrading metric
            for i in range(20):
                performance.record("task_completion_rate", 0.9 - i * 0.03)

            adaptations = engine.suggest_adaptations()
            degrading_adaptations = [a for a in adaptations if "degrading" in a.trigger]
            assert len(degrading_adaptations) >= 1


# =============================================================================
# MetaCognitiveMonitor Integration Tests
# =============================================================================

class TestMetaCognitiveMonitor:
    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = MetaCognitiveMonitor(Path(tmpdir))

            assert monitor.performance is not None
            assert monitor.stuck is not None
            assert monitor.strategies is not None
            assert monitor.novelty is not None
            assert monitor.adaptation is not None

    def test_record_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = MetaCognitiveMonitor(Path(tmpdir))

            monitor.record_step(
                action="test_action",
                state="test_state",
                success=True,
                duration_ms=100.0,
                reward=0.5,
            )

            current = monitor.performance.get_current(CognitiveMetric.TASK_COMPLETION_RATE)
            assert current == 1.0

    def test_record_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = MetaCognitiveMonitor(Path(tmpdir))

            monitor.record_error("TestError: something went wrong")

            current = monitor.performance.get_current(CognitiveMetric.ERROR_RATE)
            assert current == 1.0

    def test_check_cognitive_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = MetaCognitiveMonitor(Path(tmpdir))

            # Normal operation
            for i in range(5):
                monitor.record_step(
                    action=f"action_{i}",
                    state=f"state_{i}",
                    success=True,
                    duration_ms=100.0,
                    reward=0.5,
                )

            state = monitor.check_cognitive_state()

            assert "status" in state
            assert "stuck_detections" in state
            assert "performance_metrics" in state
            assert "novelty_rate" in state

    def test_detect_stuck_via_monitor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = MetaCognitiveMonitor(Path(tmpdir))

            # Create stuck state
            for _ in range(10):
                monitor.record_step(
                    action="same_action",
                    state="same_state",
                    success=True,
                    duration_ms=100.0,
                    reward=0.5,
                )

            state = monitor.check_cognitive_state()

            # Should detect stagnation at minimum
            assert state["status"] == "attention_needed" or len(state["stuck_detections"]) > 0

    def test_strategy_management(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = MetaCognitiveMonitor(Path(tmpdir))

            strategy = monitor.register_strategy("test_strategy", "A test strategy")
            assert strategy is not None

            monitor.record_strategy_use(
                strategy.id,
                success=True,
                reward=0.8,
                duration_ms=100.0,
                context="testing",
            )

            recommended = monitor.get_recommended_strategy("testing")
            assert recommended is not None
            assert recommended.name == "test_strategy"

    def test_get_full_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = MetaCognitiveMonitor(Path(tmpdir))

            # Add some data
            for i in range(5):
                monitor.record_step(
                    action=f"action_{i}",
                    state=f"state_{i}",
                    success=True,
                    duration_ms=100.0,
                    reward=0.5,
                )

            report = monitor.get_full_report()

            assert "cognitive_state" in report
            assert "strategy_statistics" in report
            assert "performance_trends" in report
            assert "generated_at" in report

    def test_novelty_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = MetaCognitiveMonitor(Path(tmpdir))

            # All unique actions (use different words, not just numbers)
            actions = ["read_file", "write_data", "run_tests", "commit_changes",
                      "deploy_app", "check_status", "fetch_logs", "update_config",
                      "restart_server", "validate_input"]
            for i, action in enumerate(actions):
                monitor.record_step(
                    action=action,
                    state=f"state_{i}",
                    success=True,
                    duration_ms=100.0,
                    reward=0.5,
                )

            state = monitor.check_cognitive_state()
            assert state["novelty_rate"] == 1.0
            assert state["should_explore"] is False  # High novelty, no need

    def test_low_novelty_suggests_exploration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = MetaCognitiveMonitor(Path(tmpdir))

            # Repeated actions
            for _ in range(20):
                monitor.record_step(
                    action="same_action",
                    state="changing_state",  # Vary state to avoid stagnation
                    success=True,
                    duration_ms=100.0,
                    reward=0.5,
                )

            # Manually vary states to avoid stagnation detection
            for i in range(20):
                monitor.stuck.record_state(f"state_{i}")

            state = monitor.check_cognitive_state()
            assert state["novelty_rate"] < 0.5
            assert state["should_explore"] is True
