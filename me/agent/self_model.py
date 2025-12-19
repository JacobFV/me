"""
Self-Model - The agent's understanding of its own capabilities.

This module implements the agent's self-model, enabling:
1. Capability boundary awareness - what can/can't I do?
2. Failure prediction - when am I likely to fail?
3. Confidence calibration - how reliable are my estimates?
4. Competence tracking - where am I improving/declining?
5. Limitation acknowledgment - honest assessment of constraints

Philosophy: An agent without self-awareness will confidently attempt
tasks beyond its capabilities and fail to recognize when it needs
help. A good self-model enables:
- Knowing when to accept vs decline tasks
- Predicting failures before they occur
- Seeking help appropriately
- Accurate confidence estimation
- Continuous capability improvement

This is related to but distinct from metacognition - self-model
is the static understanding of capabilities, while metacognition
is the dynamic monitoring of cognitive processes.
"""

from __future__ import annotations

from datetime import datetime, UTC, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
import json
import hashlib
import math

from pydantic import BaseModel, Field


class CapabilityDomain(str, Enum):
    """Domains of agent capability."""
    CODING = "coding"
    DEBUGGING = "debugging"
    REASONING = "reasoning"
    PLANNING = "planning"
    COMMUNICATION = "communication"
    RESEARCH = "research"
    CREATIVITY = "creativity"
    MATH = "math"
    MEMORY = "memory"
    LEARNING = "learning"


class ConfidenceLevel(str, Enum):
    """Calibrated confidence levels."""
    VERY_LOW = "very_low"  # <20% expected success
    LOW = "low"  # 20-40%
    MODERATE = "moderate"  # 40-60%
    HIGH = "high"  # 60-80%
    VERY_HIGH = "very_high"  # 80%+


class FailureMode(str, Enum):
    """Common ways the agent can fail."""
    CAPABILITY_EXCEEDED = "capability_exceeded"
    CONTEXT_OVERFLOW = "context_overflow"
    HALLUCINATION = "hallucination"
    MISUNDERSTANDING = "misunderstanding"
    INCOMPLETE_SOLUTION = "incomplete_solution"
    WRONG_APPROACH = "wrong_approach"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_DEPENDENCY = "external_dependency"


class CapabilityProfile(BaseModel):
    """Profile of capability in a specific domain."""
    domain: CapabilityDomain
    # Performance metrics
    success_rate: float = 0.5
    average_quality: float = 0.5  # 0-1 quality of successful outputs
    average_time: float = 0.0  # Average time to complete
    # Boundaries
    complexity_ceiling: float = 0.5  # Max complexity successfully handled
    known_strengths: list[str] = Field(default_factory=list)
    known_weaknesses: list[str] = Field(default_factory=list)
    # Sample history
    sample_count: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FailurePrediction(BaseModel):
    """Prediction about potential failure."""
    task_description: str
    predicted_failure_modes: list[FailureMode]
    failure_probability: float
    confidence_in_prediction: float
    risk_factors: list[str]
    mitigation_suggestions: list[str]


class CalibrationMetric(BaseModel):
    """Measures how well-calibrated confidence estimates are."""
    confidence_bucket: float  # e.g., 0.7 for 70% confidence
    actual_success_rate: float
    sample_count: int
    calibration_error: float  # |predicted - actual|


class CompetenceTrajectory(BaseModel):
    """Track of competence over time in a domain."""
    domain: CapabilityDomain
    measurements: list[tuple[datetime, float]] = Field(default_factory=list)
    trend: str = "stable"  # improving, declining, stable
    trend_strength: float = 0.0


class LimitationAcknowledgment(BaseModel):
    """Explicit acknowledgment of a limitation."""
    id: str
    description: str
    domain: CapabilityDomain | None = None
    severity: str = "moderate"  # minor, moderate, major
    workarounds: list[str] = Field(default_factory=list)
    accepted: bool = True  # Has the agent accepted this limitation?
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskFeasibility(BaseModel):
    """Assessment of whether a task is feasible."""
    task_description: str
    feasible: bool
    confidence: float
    estimated_success_probability: float
    estimated_time: float | None = None
    required_capabilities: list[CapabilityDomain]
    potential_blockers: list[str]
    recommendations: list[str]


class CapabilityTracker:
    """
    Tracks capabilities across domains.

    Maintains profiles of what the agent can and can't do,
    updated based on actual performance.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.profiles_dir = body_dir / "self_model" / "capabilities"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[CapabilityDomain, CapabilityProfile] = {}
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load capability profiles from disk."""
        for domain in CapabilityDomain:
            path = self.profiles_dir / f"{domain.value}.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    self._profiles[domain] = CapabilityProfile.model_validate(data)
                except Exception:
                    self._profiles[domain] = CapabilityProfile(domain=domain)
            else:
                self._profiles[domain] = CapabilityProfile(domain=domain)

    def _save_profile(self, profile: CapabilityProfile) -> None:
        """Save a profile to disk."""
        path = self.profiles_dir / f"{profile.domain.value}.json"
        path.write_text(json.dumps(profile.model_dump(mode='json'), indent=2, default=str))

    def get_profile(self, domain: CapabilityDomain) -> CapabilityProfile:
        """Get capability profile for a domain."""
        return self._profiles.get(domain, CapabilityProfile(domain=domain))

    def record_outcome(
        self,
        domain: CapabilityDomain,
        success: bool,
        quality: float = 0.5,
        time_taken: float = 0.0,
        complexity: float = 0.5,
    ) -> None:
        """Record outcome of a task in a domain."""
        profile = self._profiles.get(domain, CapabilityProfile(domain=domain))

        # Update with exponential moving average
        alpha = 0.1  # Learning rate
        profile.success_rate = (1 - alpha) * profile.success_rate + alpha * (1.0 if success else 0.0)
        profile.average_quality = (1 - alpha) * profile.average_quality + alpha * quality
        profile.average_time = (1 - alpha) * profile.average_time + alpha * time_taken

        # Update complexity ceiling if we succeeded at higher complexity
        if success and complexity > profile.complexity_ceiling:
            profile.complexity_ceiling = complexity

        profile.sample_count += 1
        profile.last_updated = datetime.now(UTC)

        self._profiles[domain] = profile
        self._save_profile(profile)

    def add_strength(self, domain: CapabilityDomain, strength: str) -> None:
        """Record a known strength."""
        profile = self._profiles.get(domain, CapabilityProfile(domain=domain))
        if strength not in profile.known_strengths:
            profile.known_strengths.append(strength)
            self._save_profile(profile)

    def add_weakness(self, domain: CapabilityDomain, weakness: str) -> None:
        """Record a known weakness."""
        profile = self._profiles.get(domain, CapabilityProfile(domain=domain))
        if weakness not in profile.known_weaknesses:
            profile.known_weaknesses.append(weakness)
            self._save_profile(profile)

    def get_all_profiles(self) -> dict[CapabilityDomain, CapabilityProfile]:
        """Get all capability profiles."""
        return self._profiles.copy()


class FailurePredictor:
    """
    Predicts failures before they happen.

    Uses historical data and heuristics to identify
    situations likely to result in failure.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.capability_tracker = CapabilityTracker(body_dir)
        self.failure_history: list[dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load failure history."""
        history_path = self.body_dir / "self_model" / "failure_history.json"
        if history_path.exists():
            try:
                self.failure_history = json.loads(history_path.read_text())
            except Exception:
                self.failure_history = []

    def _save_history(self) -> None:
        """Save failure history."""
        history_path = self.body_dir / "self_model" / "failure_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(self.failure_history[-1000:], indent=2, default=str))

    def record_failure(
        self,
        task_description: str,
        failure_mode: FailureMode,
        context: dict[str, Any],
    ) -> None:
        """Record a failure for learning."""
        self.failure_history.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "task": task_description,
            "mode": failure_mode.value,
            "context": context,
        })
        self._save_history()

    def predict_failure(
        self,
        task_description: str,
        required_domains: list[CapabilityDomain],
        estimated_complexity: float,
        context: dict[str, Any] | None = None,
    ) -> FailurePrediction:
        """Predict if and how a task might fail."""
        context = context or {}
        predicted_modes: list[FailureMode] = []
        risk_factors: list[str] = []
        mitigations: list[str] = []

        failure_probability = 0.0

        for domain in required_domains:
            profile = self.capability_tracker.get_profile(domain)

            # Check complexity ceiling
            if estimated_complexity > profile.complexity_ceiling:
                predicted_modes.append(FailureMode.CAPABILITY_EXCEEDED)
                risk_factors.append(f"Task complexity ({estimated_complexity:.2f}) exceeds {domain.value} ceiling ({profile.complexity_ceiling:.2f})")
                mitigations.append(f"Break down {domain.value} components into smaller pieces")
                failure_probability = max(failure_probability, 0.7)

            # Low success rate in domain
            if profile.success_rate < 0.5 and profile.sample_count >= 5:
                risk_factors.append(f"Low success rate in {domain.value}: {profile.success_rate:.0%}")
                failure_probability = max(failure_probability, 0.6)

        # Check for context-specific risks
        if context.get("requires_external_api"):
            predicted_modes.append(FailureMode.EXTERNAL_DEPENDENCY)
            risk_factors.append("Depends on external API availability")
            mitigations.append("Have fallback for API failures")

        if context.get("estimated_tokens", 0) > 50000:
            predicted_modes.append(FailureMode.CONTEXT_OVERFLOW)
            risk_factors.append("May exceed context window")
            mitigations.append("Process in chunks or use summarization")

        if context.get("requires_precise_recall"):
            predicted_modes.append(FailureMode.HALLUCINATION)
            risk_factors.append("Task requires precise recall")
            mitigations.append("Verify all facts against source material")

        # Check failure history for similar tasks
        similar_failures = self._find_similar_failures(task_description)
        if similar_failures:
            for failure in similar_failures[:3]:
                mode = FailureMode(failure["mode"])
                if mode not in predicted_modes:
                    predicted_modes.append(mode)
            risk_factors.append(f"Similar tasks have failed {len(similar_failures)} times before")
            failure_probability = max(failure_probability, 0.5)

        # Default modes if none identified
        if not predicted_modes:
            predicted_modes = [FailureMode.INCOMPLETE_SOLUTION]

        return FailurePrediction(
            task_description=task_description,
            predicted_failure_modes=predicted_modes,
            failure_probability=min(1.0, failure_probability),
            confidence_in_prediction=0.5 + 0.1 * len(self.failure_history),  # More history = more confident
            risk_factors=risk_factors,
            mitigation_suggestions=mitigations,
        )

    def _find_similar_failures(self, task_description: str) -> list[dict[str, Any]]:
        """Find similar past failures."""
        # Simple keyword matching
        task_words = set(task_description.lower().split())
        similar = []

        for failure in self.failure_history:
            failure_words = set(failure["task"].lower().split())
            overlap = len(task_words & failure_words) / len(task_words | failure_words)
            if overlap > 0.3:
                similar.append(failure)

        return similar


class ConfidenceCalibrator:
    """
    Calibrates confidence estimates.

    Tracks how well predicted confidence matches actual outcomes
    and provides calibrated estimates.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.calibration_file = body_dir / "self_model" / "calibration.json"
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load calibration history."""
        if self.calibration_file.exists():
            try:
                self._history = json.loads(self.calibration_file.read_text())
            except Exception:
                self._history = []

    def _save_history(self) -> None:
        """Save calibration history."""
        self.calibration_file.write_text(json.dumps(self._history[-1000:], indent=2))

    def record_prediction(
        self,
        prediction_id: str,
        stated_confidence: float,
        actual_outcome: bool,
    ) -> None:
        """Record a prediction and its outcome."""
        self._history.append({
            "id": prediction_id,
            "confidence": stated_confidence,
            "outcome": actual_outcome,
            "timestamp": datetime.now(UTC).isoformat(),
        })
        self._save_history()

    def get_calibration_metrics(self) -> list[CalibrationMetric]:
        """Get calibration metrics by confidence bucket."""
        if not self._history:
            return []

        # Group by confidence buckets (0.1, 0.2, ..., 1.0)
        buckets: dict[float, list[bool]] = {}
        for entry in self._history:
            bucket = round(entry["confidence"], 1)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(entry["outcome"])

        metrics = []
        for bucket, outcomes in sorted(buckets.items()):
            actual_rate = sum(outcomes) / len(outcomes)
            metrics.append(CalibrationMetric(
                confidence_bucket=bucket,
                actual_success_rate=actual_rate,
                sample_count=len(outcomes),
                calibration_error=abs(bucket - actual_rate),
            ))

        return metrics

    def get_calibration_adjustment(self, raw_confidence: float) -> float:
        """Adjust confidence based on calibration history."""
        metrics = self.get_calibration_metrics()
        if not metrics:
            return raw_confidence  # No data to calibrate

        # Find closest bucket
        closest = min(metrics, key=lambda m: abs(m.confidence_bucket - raw_confidence))

        if closest.sample_count < 5:
            return raw_confidence  # Not enough data

        # Adjust towards actual success rate
        adjustment_strength = min(0.5, closest.sample_count / 20)
        calibrated = raw_confidence + adjustment_strength * (closest.actual_success_rate - raw_confidence)

        return max(0.0, min(1.0, calibrated))

    def get_overall_calibration_error(self) -> float:
        """Get overall calibration error (Brier score component)."""
        metrics = self.get_calibration_metrics()
        if not metrics:
            return 0.0

        total_samples = sum(m.sample_count for m in metrics)
        weighted_error = sum(m.calibration_error * m.sample_count for m in metrics)

        return weighted_error / total_samples if total_samples > 0 else 0.0


class CompetenceMonitor:
    """
    Monitors competence trajectories over time.

    Tracks whether capabilities are improving, stable, or declining
    and identifies areas needing attention.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.trajectories_file = body_dir / "self_model" / "trajectories.json"
        self.trajectories_file.parent.mkdir(parents=True, exist_ok=True)
        self._trajectories: dict[str, CompetenceTrajectory] = {}
        self._load_trajectories()

    def _load_trajectories(self) -> None:
        """Load competence trajectories."""
        if self.trajectories_file.exists():
            try:
                data = json.loads(self.trajectories_file.read_text())
                for domain_str, traj_data in data.items():
                    traj = CompetenceTrajectory.model_validate(traj_data)
                    self._trajectories[domain_str] = traj
            except Exception:
                pass

    def _save_trajectories(self) -> None:
        """Save competence trajectories."""
        data = {
            domain: traj.model_dump(mode='json')
            for domain, traj in self._trajectories.items()
        }
        self.trajectories_file.write_text(json.dumps(data, indent=2, default=str))

    def record_measurement(
        self,
        domain: CapabilityDomain,
        competence_score: float,
    ) -> None:
        """Record a competence measurement."""
        domain_str = domain.value
        if domain_str not in self._trajectories:
            self._trajectories[domain_str] = CompetenceTrajectory(domain=domain)

        traj = self._trajectories[domain_str]
        traj.measurements.append((datetime.now(UTC), competence_score))

        # Keep last 100 measurements
        if len(traj.measurements) > 100:
            traj.measurements = traj.measurements[-100:]

        # Compute trend
        if len(traj.measurements) >= 5:
            recent = [m[1] for m in traj.measurements[-10:]]
            older = [m[1] for m in traj.measurements[-20:-10]] if len(traj.measurements) >= 20 else [m[1] for m in traj.measurements[:len(traj.measurements)//2]]

            if older:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                diff = recent_avg - older_avg

                if diff > 0.1:
                    traj.trend = "improving"
                    traj.trend_strength = min(1.0, diff * 5)
                elif diff < -0.1:
                    traj.trend = "declining"
                    traj.trend_strength = min(1.0, -diff * 5)
                else:
                    traj.trend = "stable"
                    traj.trend_strength = 0.0

        self._save_trajectories()

    def get_trajectory(self, domain: CapabilityDomain) -> CompetenceTrajectory | None:
        """Get competence trajectory for a domain."""
        return self._trajectories.get(domain.value)

    def get_all_trajectories(self) -> dict[CapabilityDomain, CompetenceTrajectory]:
        """Get all competence trajectories."""
        return {
            CapabilityDomain(k): v
            for k, v in self._trajectories.items()
        }

    def get_areas_needing_attention(self) -> list[tuple[CapabilityDomain, str]]:
        """Identify areas that need attention."""
        attention = []

        for domain_str, traj in self._trajectories.items():
            domain = CapabilityDomain(domain_str)

            if traj.trend == "declining" and traj.trend_strength > 0.3:
                attention.append((domain, f"Declining competence (strength: {traj.trend_strength:.2f})"))

            if traj.measurements:
                recent = traj.measurements[-1][1]
                if recent < 0.3:
                    attention.append((domain, f"Low competence level ({recent:.2f})"))

        return attention


class LimitationRegistry:
    """
    Registry of known limitations.

    Maintains explicit acknowledgment of what the agent
    can't do or struggles with.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.limitations_dir = body_dir / "self_model" / "limitations"
        self.limitations_dir.mkdir(parents=True, exist_ok=True)

    def acknowledge_limitation(
        self,
        description: str,
        domain: CapabilityDomain | None = None,
        severity: str = "moderate",
        workarounds: list[str] | None = None,
    ) -> LimitationAcknowledgment:
        """Acknowledge a limitation."""
        limit_id = hashlib.md5(description.encode()).hexdigest()[:12]

        limitation = LimitationAcknowledgment(
            id=limit_id,
            description=description,
            domain=domain,
            severity=severity,
            workarounds=workarounds or [],
        )

        path = self.limitations_dir / f"{limit_id}.json"
        path.write_text(json.dumps(limitation.model_dump(mode='json'), indent=2, default=str))

        return limitation

    def get_all_limitations(self) -> list[LimitationAcknowledgment]:
        """Get all acknowledged limitations."""
        limitations = []
        for path in self.limitations_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                limitations.append(LimitationAcknowledgment.model_validate(data))
            except Exception:
                continue
        return limitations

    def get_limitations_for_domain(self, domain: CapabilityDomain) -> list[LimitationAcknowledgment]:
        """Get limitations for a specific domain."""
        return [l for l in self.get_all_limitations() if l.domain == domain]

    def get_relevant_limitations(self, task_description: str) -> list[LimitationAcknowledgment]:
        """Find limitations relevant to a task."""
        all_limits = self.get_all_limitations()
        relevant = []

        task_lower = task_description.lower()
        for limit in all_limits:
            # Simple keyword matching
            limit_words = limit.description.lower().split()
            if any(word in task_lower for word in limit_words if len(word) > 3):
                relevant.append(limit)

        return relevant


class FeasibilityAssessor:
    """
    Assesses whether tasks are feasible.

    Combines capability profiles, failure prediction,
    and limitation awareness to assess task feasibility.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.capability_tracker = CapabilityTracker(body_dir)
        self.failure_predictor = FailurePredictor(body_dir)
        self.limitation_registry = LimitationRegistry(body_dir)
        self.calibrator = ConfidenceCalibrator(body_dir)

    def assess_feasibility(
        self,
        task_description: str,
        required_domains: list[CapabilityDomain],
        estimated_complexity: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> TaskFeasibility:
        """Assess if a task is feasible."""
        context = context or {}

        # Get failure prediction
        failure_pred = self.failure_predictor.predict_failure(
            task_description,
            required_domains,
            estimated_complexity,
            context,
        )

        # Check relevant limitations
        relevant_limits = self.limitation_registry.get_relevant_limitations(task_description)
        blockers = [l.description for l in relevant_limits if l.severity == "major"]

        # Estimate success probability from capability profiles
        success_probs = []
        for domain in required_domains:
            profile = self.capability_tracker.get_profile(domain)

            # Base probability from success rate
            base_prob = profile.success_rate

            # Adjust for complexity
            if estimated_complexity > profile.complexity_ceiling:
                complexity_penalty = (estimated_complexity - profile.complexity_ceiling) * 0.5
                base_prob *= max(0.1, 1 - complexity_penalty)

            success_probs.append(base_prob)

        # Overall success probability (assuming independent domains)
        if success_probs:
            overall_prob = 1.0
            for prob in success_probs:
                overall_prob *= prob
        else:
            overall_prob = 0.5  # No data

        # Combine with failure prediction
        combined_prob = overall_prob * (1 - failure_pred.failure_probability)

        # Calibrate
        calibrated_prob = self.calibrator.get_calibration_adjustment(combined_prob)

        # Determine feasibility
        feasible = calibrated_prob > 0.3 and len(blockers) == 0

        # Estimate time
        estimated_time = None
        if required_domains:
            times = [
                self.capability_tracker.get_profile(d).average_time
                for d in required_domains
            ]
            estimated_time = sum(times) * (1 + estimated_complexity)

        # Build recommendations
        recommendations = failure_pred.mitigation_suggestions.copy()
        if not feasible:
            recommendations.insert(0, "Consider breaking into smaller tasks")
            if blockers:
                recommendations.append("Address major limitations first")

        return TaskFeasibility(
            task_description=task_description,
            feasible=feasible,
            confidence=0.5 + 0.1 * sum(
                self.capability_tracker.get_profile(d).sample_count
                for d in required_domains
            ),
            estimated_success_probability=calibrated_prob,
            estimated_time=estimated_time,
            required_capabilities=required_domains,
            potential_blockers=blockers + failure_pred.risk_factors,
            recommendations=recommendations,
        )


class SelfModelSystem:
    """
    Unified system for self-model management.

    Provides a single interface for:
    - Capability tracking
    - Failure prediction
    - Confidence calibration
    - Competence monitoring
    - Limitation acknowledgment
    - Feasibility assessment
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.capability_tracker = CapabilityTracker(body_dir)
        self.failure_predictor = FailurePredictor(body_dir)
        self.calibrator = ConfidenceCalibrator(body_dir)
        self.competence_monitor = CompetenceMonitor(body_dir)
        self.limitation_registry = LimitationRegistry(body_dir)
        self.feasibility_assessor = FeasibilityAssessor(body_dir)

    def record_task_outcome(
        self,
        domain: CapabilityDomain,
        success: bool,
        quality: float = 0.5,
        time_taken: float = 0.0,
        complexity: float = 0.5,
        stated_confidence: float | None = None,
    ) -> None:
        """Record the outcome of a task."""
        # Update capability profile
        self.capability_tracker.record_outcome(
            domain, success, quality, time_taken, complexity
        )

        # Update competence trajectory
        competence_score = quality if success else quality * 0.3
        self.competence_monitor.record_measurement(domain, competence_score)

        # Update calibration if confidence was stated
        if stated_confidence is not None:
            pred_id = hashlib.md5(
                f"{domain.value}:{datetime.now(UTC).isoformat()}".encode()
            ).hexdigest()[:12]
            self.calibrator.record_prediction(pred_id, stated_confidence, success)

    def record_failure(
        self,
        task_description: str,
        failure_mode: FailureMode,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a failure for learning."""
        self.failure_predictor.record_failure(
            task_description, failure_mode, context or {}
        )

    def predict_failure(
        self,
        task_description: str,
        required_domains: list[CapabilityDomain],
        estimated_complexity: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> FailurePrediction:
        """Predict potential failures."""
        return self.failure_predictor.predict_failure(
            task_description, required_domains, estimated_complexity, context
        )

    def assess_feasibility(
        self,
        task_description: str,
        required_domains: list[CapabilityDomain],
        estimated_complexity: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> TaskFeasibility:
        """Assess task feasibility."""
        return self.feasibility_assessor.assess_feasibility(
            task_description, required_domains, estimated_complexity, context
        )

    def acknowledge_limitation(
        self,
        description: str,
        domain: CapabilityDomain | None = None,
        severity: str = "moderate",
        workarounds: list[str] | None = None,
    ) -> LimitationAcknowledgment:
        """Acknowledge a limitation."""
        return self.limitation_registry.acknowledge_limitation(
            description, domain, severity, workarounds
        )

    def get_capability_summary(self) -> dict[str, Any]:
        """Get summary of all capabilities."""
        profiles = self.capability_tracker.get_all_profiles()

        return {
            "domains": {
                domain.value: {
                    "success_rate": profile.success_rate,
                    "quality": profile.average_quality,
                    "complexity_ceiling": profile.complexity_ceiling,
                    "sample_count": profile.sample_count,
                    "strengths": profile.known_strengths,
                    "weaknesses": profile.known_weaknesses,
                }
                for domain, profile in profiles.items()
            },
            "calibration_error": self.calibrator.get_overall_calibration_error(),
            "areas_needing_attention": [
                {"domain": d.value, "reason": r}
                for d, r in self.competence_monitor.get_areas_needing_attention()
            ],
            "known_limitations": len(self.limitation_registry.get_all_limitations()),
        }

    def get_confidence_for_task(
        self,
        required_domains: list[CapabilityDomain],
        estimated_complexity: float = 0.5,
    ) -> tuple[ConfidenceLevel, float]:
        """Get calibrated confidence level for a task."""
        # Compute raw confidence
        if not required_domains:
            return ConfidenceLevel.MODERATE, 0.5

        success_rates = [
            self.capability_tracker.get_profile(d).success_rate
            for d in required_domains
        ]
        raw_confidence = sum(success_rates) / len(success_rates)

        # Adjust for complexity
        avg_ceiling = sum(
            self.capability_tracker.get_profile(d).complexity_ceiling
            for d in required_domains
        ) / len(required_domains)

        if estimated_complexity > avg_ceiling:
            raw_confidence *= 0.5

        # Calibrate
        calibrated = self.calibrator.get_calibration_adjustment(raw_confidence)

        # Convert to level
        if calibrated < 0.2:
            level = ConfidenceLevel.VERY_LOW
        elif calibrated < 0.4:
            level = ConfidenceLevel.LOW
        elif calibrated < 0.6:
            level = ConfidenceLevel.MODERATE
        elif calibrated < 0.8:
            level = ConfidenceLevel.HIGH
        else:
            level = ConfidenceLevel.VERY_HIGH

        return level, calibrated

    def should_seek_help(
        self,
        task_description: str,
        required_domains: list[CapabilityDomain],
        estimated_complexity: float = 0.5,
    ) -> tuple[bool, str]:
        """Determine if help should be sought for a task."""
        feasibility = self.assess_feasibility(
            task_description, required_domains, estimated_complexity
        )

        if not feasibility.feasible:
            return True, "Task assessed as not feasible"

        if feasibility.estimated_success_probability < 0.3:
            return True, f"Low success probability ({feasibility.estimated_success_probability:.0%})"

        if feasibility.potential_blockers:
            major_blockers = [b for b in feasibility.potential_blockers if "major" in b.lower() or "critical" in b.lower()]
            if major_blockers:
                return True, f"Major blockers identified: {major_blockers[0]}"

        return False, "Task appears feasible"

    def add_strength(self, domain: CapabilityDomain, strength: str) -> None:
        """Add a known strength."""
        self.capability_tracker.add_strength(domain, strength)

    def add_weakness(self, domain: CapabilityDomain, weakness: str) -> None:
        """Add a known weakness."""
        self.capability_tracker.add_weakness(domain, weakness)
