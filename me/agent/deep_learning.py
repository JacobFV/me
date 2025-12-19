"""
Deep Learning Loop - Genuine behavioral improvement from experience.

This module implements actual learning that changes agent behavior:
1. Strategy library with performance tracking and selection
2. Prompt template refinement based on outcomes
3. Action sequence detection and procedure synthesis
4. Feedback integration with credit assignment
5. A/B testing of approaches

This is not just analytics - it changes what the agent does next time.

Philosophy: Learning is not remembering what happened. Learning is
changing what you will do. The agent must actually behave differently
after experience, not just recall that experience.
"""

from __future__ import annotations

import json
import hashlib
import random
from collections import defaultdict
from datetime import datetime, timedelta, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import re

from pydantic import BaseModel, Field


# =============================================================================
# Strategy Library - Reusable approaches with performance tracking
# =============================================================================

class StrategyOutcome(str, Enum):
    """Outcome of applying a strategy."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    ABANDONED = "abandoned"


class StrategyApplication(BaseModel):
    """Record of applying a strategy."""
    strategy_id: str
    context_hash: str  # Hash of context for similarity matching
    outcome: StrategyOutcome
    duration_seconds: float
    tokens_used: int = 0
    error_count: int = 0
    applied_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context_summary: str = ""
    notes: str = ""


class Strategy(BaseModel):
    """A reusable approach to a class of problems."""
    id: str
    name: str
    description: str

    # When to use this strategy
    applicable_domains: list[str] = Field(default_factory=list)
    applicable_patterns: list[str] = Field(default_factory=list)  # Regex patterns
    preconditions: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)  # When NOT to use

    # The strategy itself
    approach: str  # Natural language description
    steps: list[str] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    fallback_strategy: str | None = None

    # Performance tracking
    times_applied: int = 0
    times_succeeded: int = 0
    times_failed: int = 0
    avg_duration: float = 0.0
    avg_tokens: float = 0.0

    # Computed metrics
    success_rate: float = 0.0
    efficiency_score: float = 0.0  # Success / tokens
    reliability_score: float = 0.0  # Consistency of outcomes

    # Evolution
    version: int = 1
    parent_strategy: str | None = None
    refinements: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime | None = None

    # Learning
    context_embeddings: list[list[float]] = Field(default_factory=list)
    outcome_by_context: dict[str, str] = Field(default_factory=dict)  # context_hash -> outcome


class StrategyLibrary:
    """
    Library of strategies with selection and refinement.

    The library learns which strategies work in which contexts
    and can synthesize new strategies from successful patterns.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.strategies_dir = body_dir / "learning" / "strategies"
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        self.applications_file = self.strategies_dir / "applications.jsonl"

    def add_strategy(self, strategy: Strategy) -> None:
        """Add a strategy to the library."""
        path = self.strategies_dir / f"{strategy.id}.json"
        path.write_text(json.dumps(strategy.model_dump(mode='json'), indent=2, default=str))

    def get_strategy(self, strategy_id: str) -> Strategy | None:
        """Get a strategy by ID."""
        path = self.strategies_dir / f"{strategy_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return Strategy.model_validate(data)
        except Exception:
            return None

    def select_strategy(
        self,
        task_description: str,
        context: dict[str, Any],
        domain: str | None = None,
    ) -> Strategy | None:
        """
        Select the best strategy for a task.

        Uses multiple signals:
        1. Domain matching
        2. Pattern matching on task description
        3. Historical success in similar contexts
        4. Exploration bonus for undertried strategies
        """
        candidates = []
        context_hash = self._hash_context(context)

        for path in self.strategies_dir.glob("*.json"):
            if path.name == "applications.jsonl":
                continue
            try:
                strategy = Strategy.model_validate(json.loads(path.read_text()))
            except Exception:
                continue

            score = 0.0

            # Domain match
            if domain and domain in strategy.applicable_domains:
                score += 2.0

            # Pattern match
            for pattern in strategy.applicable_patterns:
                if re.search(pattern, task_description, re.IGNORECASE):
                    score += 1.5
                    break

            # Historical success
            if strategy.times_applied > 0:
                score += strategy.success_rate * 2.0
                # Bonus for efficiency
                score += min(1.0, strategy.efficiency_score * 0.5)

            # Context similarity (if we have embeddings)
            if context_hash in strategy.outcome_by_context:
                past_outcome = strategy.outcome_by_context[context_hash]
                if past_outcome == "success":
                    score += 3.0
                elif past_outcome == "failure":
                    score -= 2.0

            # Exploration bonus (UCB-like)
            if strategy.times_applied < 5:
                score += 1.0 / (strategy.times_applied + 1)

            # Check contraindications
            for contra in strategy.contraindications:
                if contra.lower() in task_description.lower():
                    score -= 5.0
                    break

            if score > 0:
                candidates.append((strategy, score))

        if not candidates:
            return None

        # Sort by score, return best
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    def record_application(
        self,
        strategy_id: str,
        context: dict[str, Any],
        outcome: StrategyOutcome,
        duration_seconds: float,
        tokens_used: int = 0,
        error_count: int = 0,
        notes: str = "",
    ) -> None:
        """Record the outcome of applying a strategy."""
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return

        context_hash = self._hash_context(context)
        context_summary = self._summarize_context(context)

        # Record application
        application = StrategyApplication(
            strategy_id=strategy_id,
            context_hash=context_hash,
            outcome=outcome,
            duration_seconds=duration_seconds,
            tokens_used=tokens_used,
            error_count=error_count,
            context_summary=context_summary,
            notes=notes,
        )

        # Append to applications log
        with open(self.applications_file, 'a') as f:
            f.write(json.dumps(application.model_dump(mode='json'), default=str) + '\n')

        # Update strategy statistics
        strategy.times_applied += 1
        strategy.last_used = datetime.now(UTC)

        if outcome == StrategyOutcome.SUCCESS:
            strategy.times_succeeded += 1
        elif outcome == StrategyOutcome.FAILURE:
            strategy.times_failed += 1

        # Update running averages
        n = strategy.times_applied
        strategy.avg_duration = (strategy.avg_duration * (n-1) + duration_seconds) / n
        strategy.avg_tokens = (strategy.avg_tokens * (n-1) + tokens_used) / n

        # Recompute metrics
        strategy.success_rate = strategy.times_succeeded / strategy.times_applied
        if strategy.avg_tokens > 0:
            strategy.efficiency_score = strategy.success_rate / (strategy.avg_tokens / 1000)

        # Record context -> outcome mapping
        strategy.outcome_by_context[context_hash] = outcome.value

        # Save updated strategy
        self.add_strategy(strategy)

    def synthesize_strategy(
        self,
        name: str,
        description: str,
        from_episodes: list[str],
        approach: str,
        steps: list[str],
    ) -> Strategy:
        """Synthesize a new strategy from successful episodes."""
        strategy_id = hashlib.md5(f"{name}:{approach}".encode()).hexdigest()[:12]

        strategy = Strategy(
            id=strategy_id,
            name=name,
            description=description,
            approach=approach,
            steps=steps,
        )

        self.add_strategy(strategy)
        return strategy

    def refine_strategy(
        self,
        strategy_id: str,
        refinement: str,
        new_steps: list[str] | None = None,
        new_contraindications: list[str] | None = None,
    ) -> Strategy | None:
        """Create a refined version of a strategy."""
        original = self.get_strategy(strategy_id)
        if not original:
            return None

        # Create new strategy as child
        new_id = f"{strategy_id}-v{original.version + 1}"
        refined = original.model_copy(deep=True)
        refined.id = new_id
        refined.version = original.version + 1
        refined.parent_strategy = strategy_id
        refined.refinements.append(refinement)
        refined.created_at = datetime.now(UTC)

        # Reset stats for new version
        refined.times_applied = 0
        refined.times_succeeded = 0
        refined.times_failed = 0
        refined.outcome_by_context = {}

        if new_steps:
            refined.steps = new_steps
        if new_contraindications:
            refined.contraindications.extend(new_contraindications)

        self.add_strategy(refined)
        return refined

    def get_best_strategies(self, min_applications: int = 3, top_k: int = 5) -> list[Strategy]:
        """Get the best performing strategies."""
        strategies = []
        for path in self.strategies_dir.glob("*.json"):
            if path.name == "applications.jsonl":
                continue
            try:
                s = Strategy.model_validate(json.loads(path.read_text()))
                if s.times_applied >= min_applications:
                    strategies.append(s)
            except Exception:
                continue

        strategies.sort(key=lambda s: (s.success_rate, s.efficiency_score), reverse=True)
        return strategies[:top_k]

    def _hash_context(self, context: dict[str, Any]) -> str:
        """Create a hash of context for similarity matching."""
        # Extract key features
        features = []
        for key in sorted(context.keys()):
            value = context[key]
            if isinstance(value, str):
                features.append(f"{key}:{value[:50]}")
            elif isinstance(value, (int, float, bool)):
                features.append(f"{key}:{value}")
        return hashlib.md5("|".join(features).encode()).hexdigest()[:16]

    def _summarize_context(self, context: dict[str, Any]) -> str:
        """Create a brief summary of context."""
        parts = []
        for key in ["task", "goal", "domain", "file"]:
            if key in context:
                parts.append(f"{key}={context[key][:30]}")
        return "; ".join(parts[:3])


# =============================================================================
# Prompt Template Refinement
# =============================================================================

class PromptTemplate(BaseModel):
    """A refinable prompt template."""
    id: str
    name: str
    purpose: str

    # The template
    template: str
    variables: list[str] = Field(default_factory=list)

    # Performance tracking
    times_used: int = 0
    avg_quality_score: float = 0.0  # 0-1 rating of outputs
    avg_tokens_generated: float = 0.0

    # Variations
    version: int = 1
    parent_template: str | None = None
    variations: list[str] = Field(default_factory=list)

    # A/B testing
    is_active: bool = True
    is_control: bool = False
    experiment_id: str | None = None


class PromptRefinementEngine:
    """
    Refines prompts based on output quality.

    Tracks which prompts produce better outputs and
    generates variations to test improvements.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.prompts_dir = body_dir / "learning" / "prompts"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def add_template(self, template: PromptTemplate) -> None:
        """Add a prompt template."""
        path = self.prompts_dir / f"{template.id}.json"
        path.write_text(json.dumps(template.model_dump(mode='json'), indent=2, default=str))

    def get_template(self, template_id: str) -> PromptTemplate | None:
        """Get a template by ID."""
        path = self.prompts_dir / f"{template_id}.json"
        if not path.exists():
            return None
        try:
            return PromptTemplate.model_validate(json.loads(path.read_text()))
        except Exception:
            return None

    def record_usage(
        self,
        template_id: str,
        quality_score: float,
        tokens_generated: int,
    ) -> None:
        """Record usage of a template with quality feedback."""
        template = self.get_template(template_id)
        if not template:
            return

        n = template.times_used + 1
        template.times_used = n
        template.avg_quality_score = (
            template.avg_quality_score * (n-1) + quality_score
        ) / n
        template.avg_tokens_generated = (
            template.avg_tokens_generated * (n-1) + tokens_generated
        ) / n

        self.add_template(template)

    def generate_variation(
        self,
        template_id: str,
        variation_type: str,
    ) -> PromptTemplate | None:
        """Generate a variation of a template for testing."""
        original = self.get_template(template_id)
        if not original:
            return None

        new_template = original.template

        if variation_type == "more_specific":
            new_template = f"Be extremely specific and detailed.\n\n{original.template}"
        elif variation_type == "more_concise":
            new_template = f"Be concise - use minimal words.\n\n{original.template}"
        elif variation_type == "step_by_step":
            new_template = f"{original.template}\n\nWork through this step by step."
        elif variation_type == "examples_first":
            new_template = f"Here are examples of good outputs:\n[EXAMPLES]\n\n{original.template}"

        new_id = f"{template_id}-v{original.version + 1}"
        variation = PromptTemplate(
            id=new_id,
            name=f"{original.name} ({variation_type})",
            purpose=original.purpose,
            template=new_template,
            variables=original.variables,
            version=original.version + 1,
            parent_template=template_id,
            variations=[variation_type],
        )

        self.add_template(variation)
        return variation

    def select_best_template(self, purpose: str, min_uses: int = 3) -> PromptTemplate | None:
        """Select the best template for a purpose."""
        candidates = []

        for path in self.prompts_dir.glob("*.json"):
            try:
                template = PromptTemplate.model_validate(json.loads(path.read_text()))
                if template.purpose == purpose and template.times_used >= min_uses:
                    candidates.append(template)
            except Exception:
                continue

        if not candidates:
            return None

        # Sort by quality score
        candidates.sort(key=lambda t: t.avg_quality_score, reverse=True)
        return candidates[0]


# =============================================================================
# Action Sequence Detection and Procedure Synthesis
# =============================================================================

class ActionSequence(BaseModel):
    """A detected sequence of actions."""
    id: str
    actions: list[str]
    frequency: int = 1
    contexts: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)

    # Success tracking
    success_count: int = 0
    failure_count: int = 0

    # Synthesis
    is_synthesized: bool = False
    procedure_id: str | None = None


class SequenceDetector:
    """
    Detects repeated action sequences that could become procedures.

    Watches action streams and identifies patterns that occur
    frequently enough to be worth codifying.
    """

    def __init__(self, body_dir: Path, min_frequency: int = 3):
        self.body_dir = body_dir
        self.sequences_dir = body_dir / "learning" / "sequences"
        self.sequences_dir.mkdir(parents=True, exist_ok=True)
        self.min_frequency = min_frequency
        self.action_buffer: list[tuple[str, str, datetime]] = []  # (action, context, time)
        self.max_buffer_size = 1000

    def observe_action(self, action: str, context: str) -> None:
        """Observe an action for sequence detection."""
        self.action_buffer.append((action, context, datetime.now(UTC)))

        # Trim buffer
        if len(self.action_buffer) > self.max_buffer_size:
            self.action_buffer = self.action_buffer[-self.max_buffer_size:]

    def detect_sequences(self, min_length: int = 2, max_length: int = 10) -> list[ActionSequence]:
        """Detect repeated sequences in the action buffer."""
        if len(self.action_buffer) < min_length:
            return []

        # Extract just actions
        actions = [a[0] for a in self.action_buffer]
        contexts = [a[1] for a in self.action_buffer]

        # Find repeated subsequences
        sequence_counts: dict[tuple, list[int]] = defaultdict(list)

        for length in range(min_length, min(max_length + 1, len(actions))):
            for i in range(len(actions) - length + 1):
                seq = tuple(actions[i:i+length])
                sequence_counts[seq].append(i)

        # Filter to frequent sequences
        frequent = []
        for seq, positions in sequence_counts.items():
            if len(positions) >= self.min_frequency:
                # Check this isn't a subsequence of a longer frequent sequence
                is_subseq = False
                for other_seq in sequence_counts:
                    if len(other_seq) > len(seq) and len(sequence_counts[other_seq]) >= self.min_frequency:
                        if self._is_subsequence(seq, other_seq):
                            is_subseq = True
                            break

                if not is_subseq:
                    seq_contexts = [contexts[p] for p in positions[:5]]
                    seq_id = hashlib.md5(str(seq).encode()).hexdigest()[:12]

                    frequent.append(ActionSequence(
                        id=seq_id,
                        actions=list(seq),
                        frequency=len(positions),
                        contexts=seq_contexts,
                    ))

        return frequent

    def synthesize_procedure(
        self,
        sequence: ActionSequence,
        name: str,
        description: str,
    ) -> dict[str, Any]:
        """Synthesize a procedure from a detected sequence."""
        # Generate procedure definition
        steps = []
        for i, action in enumerate(sequence.actions, 1):
            steps.append(f"{i}. {action}")

        procedure = {
            "name": name,
            "description": description,
            "when_to_use": f"Detected from {sequence.frequency} occurrences",
            "steps": steps,
            "source_sequence": sequence.id,
            "contexts": sequence.contexts,
        }

        # Mark sequence as synthesized
        sequence.is_synthesized = True
        sequence.procedure_id = name
        self._save_sequence(sequence)

        return procedure

    def _is_subsequence(self, shorter: tuple, longer: tuple) -> bool:
        """Check if shorter is a contiguous subsequence of longer."""
        s = "".join(shorter)
        l = "".join(longer)
        return s in l

    def _save_sequence(self, sequence: ActionSequence) -> None:
        """Save a sequence to disk."""
        path = self.sequences_dir / f"{sequence.id}.json"
        path.write_text(json.dumps(sequence.model_dump(mode='json'), indent=2, default=str))


# =============================================================================
# Feedback Integration with Credit Assignment
# =============================================================================

class FeedbackType(str, Enum):
    """Types of feedback."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    USER = "user"


class Feedback(BaseModel):
    """A piece of feedback about agent performance."""
    id: str
    feedback_type: FeedbackType
    value: float  # -1 to 1
    target: str  # What this feedback is about
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Credit assignment
    contributing_actions: list[str] = Field(default_factory=list)
    contributing_strategies: list[str] = Field(default_factory=list)
    contributing_decisions: list[str] = Field(default_factory=list)


class CreditAssigner:
    """
    Assigns credit/blame to actions that led to outcomes.

    When feedback is received, distributes it to the actions,
    strategies, and decisions that contributed.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.feedback_dir = body_dir / "learning" / "feedback"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.credit_file = self.feedback_dir / "credit.json"
        self._credit_scores: dict[str, float] = self._load_credit()

    def _load_credit(self) -> dict[str, float]:
        """Load credit scores from file."""
        if self.credit_file.exists():
            try:
                return json.loads(self.credit_file.read_text())
            except Exception:
                pass
        return {}

    def _save_credit(self) -> None:
        """Save credit scores to file."""
        self.credit_file.write_text(json.dumps(self._credit_scores, indent=2))

    def assign_credit(
        self,
        feedback: Feedback,
        decay_factor: float = 0.9,
    ) -> dict[str, float]:
        """
        Assign credit to contributing factors.

        Uses temporal credit assignment - more recent actions
        get more credit/blame.
        """
        assignments = {}

        # Assign to actions (most recent gets most credit)
        for i, action in enumerate(reversed(feedback.contributing_actions)):
            weight = decay_factor ** i
            credit = feedback.value * weight

            key = f"action:{action}"
            self._credit_scores[key] = self._credit_scores.get(key, 0) + credit
            assignments[key] = credit

        # Assign to strategies (equal weight)
        for strategy in feedback.contributing_strategies:
            credit = feedback.value * 0.5
            key = f"strategy:{strategy}"
            self._credit_scores[key] = self._credit_scores.get(key, 0) + credit
            assignments[key] = credit

        # Assign to decisions
        for decision in feedback.contributing_decisions:
            credit = feedback.value * 0.3
            key = f"decision:{decision}"
            self._credit_scores[key] = self._credit_scores.get(key, 0) + credit
            assignments[key] = credit

        self._save_credit()

        # Save feedback record
        path = self.feedback_dir / f"feedback-{feedback.id}.json"
        path.write_text(json.dumps(feedback.model_dump(mode='json'), indent=2, default=str))

        return assignments

    def get_credit(self, key: str) -> float:
        """Get accumulated credit for a key."""
        return self._credit_scores.get(key, 0)

    def get_best_actions(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Get actions with highest credit."""
        action_scores = [
            (k.replace("action:", ""), v)
            for k, v in self._credit_scores.items()
            if k.startswith("action:")
        ]
        action_scores.sort(key=lambda x: -x[1])
        return action_scores[:top_k]

    def get_worst_actions(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Get actions with lowest (most negative) credit."""
        action_scores = [
            (k.replace("action:", ""), v)
            for k, v in self._credit_scores.items()
            if k.startswith("action:")
        ]
        action_scores.sort(key=lambda x: x[1])
        return action_scores[:top_k]


# =============================================================================
# A/B Testing Framework
# =============================================================================

class ABTest(BaseModel):
    """An A/B test comparing approaches."""
    id: str
    name: str
    hypothesis: str

    # Variants
    control_id: str
    treatment_id: str
    control_description: str
    treatment_description: str

    # Allocation
    treatment_probability: float = 0.5

    # Results
    control_trials: int = 0
    treatment_trials: int = 0
    control_successes: int = 0
    treatment_successes: int = 0
    control_total_score: float = 0.0
    treatment_total_score: float = 0.0

    # Status
    status: str = "running"  # running, concluded, abandoned
    winner: str | None = None
    conclusion: str | None = None

    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    concluded_at: datetime | None = None
    min_trials_per_variant: int = 10


class ABTestRunner:
    """
    Runs A/B tests to compare approaches.

    Allows the agent to scientifically determine which
    approach works better.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.tests_dir = body_dir / "learning" / "ab_tests"
        self.tests_dir.mkdir(parents=True, exist_ok=True)

    def create_test(
        self,
        name: str,
        hypothesis: str,
        control_id: str,
        control_description: str,
        treatment_id: str,
        treatment_description: str,
        treatment_probability: float = 0.5,
        min_trials: int = 10,
    ) -> ABTest:
        """Create a new A/B test."""
        test_id = hashlib.md5(f"{name}:{control_id}:{treatment_id}".encode()).hexdigest()[:12]

        test = ABTest(
            id=test_id,
            name=name,
            hypothesis=hypothesis,
            control_id=control_id,
            treatment_id=treatment_id,
            control_description=control_description,
            treatment_description=treatment_description,
            treatment_probability=treatment_probability,
            min_trials_per_variant=min_trials,
        )

        self._save_test(test)
        return test

    def select_variant(self, test_id: str) -> tuple[str, bool]:
        """
        Select which variant to use for a trial.

        Returns (variant_id, is_treatment).
        """
        test = self._load_test(test_id)
        if not test or test.status != "running":
            return (test.control_id if test else "", False)

        is_treatment = random.random() < test.treatment_probability
        variant_id = test.treatment_id if is_treatment else test.control_id

        return (variant_id, is_treatment)

    def record_trial(
        self,
        test_id: str,
        is_treatment: bool,
        success: bool,
        score: float = 0.0,
    ) -> None:
        """Record the result of a trial."""
        test = self._load_test(test_id)
        if not test or test.status != "running":
            return

        if is_treatment:
            test.treatment_trials += 1
            if success:
                test.treatment_successes += 1
            test.treatment_total_score += score
        else:
            test.control_trials += 1
            if success:
                test.control_successes += 1
            test.control_total_score += score

        # Check if we can conclude
        if (test.control_trials >= test.min_trials_per_variant and
            test.treatment_trials >= test.min_trials_per_variant):
            self._maybe_conclude(test)

        self._save_test(test)

    def _maybe_conclude(self, test: ABTest) -> None:
        """Check if test can be concluded with statistical significance."""
        # Simple significance test
        control_rate = test.control_successes / max(1, test.control_trials)
        treatment_rate = test.treatment_successes / max(1, test.treatment_trials)

        # Require meaningful difference
        diff = abs(treatment_rate - control_rate)
        if diff < 0.1:  # Less than 10% difference
            if test.control_trials + test.treatment_trials >= test.min_trials_per_variant * 4:
                test.status = "concluded"
                test.winner = "tie"
                test.conclusion = f"No significant difference: control={control_rate:.2%}, treatment={treatment_rate:.2%}"
                test.concluded_at = datetime.now(UTC)
            return

        # Determine winner
        if treatment_rate > control_rate:
            test.winner = "treatment"
            test.conclusion = f"Treatment wins: {treatment_rate:.2%} vs {control_rate:.2%}"
        else:
            test.winner = "control"
            test.conclusion = f"Control wins: {control_rate:.2%} vs {treatment_rate:.2%}"

        test.status = "concluded"
        test.concluded_at = datetime.now(UTC)

    def get_active_tests(self) -> list[ABTest]:
        """Get all active tests."""
        tests = []
        for path in self.tests_dir.glob("*.json"):
            try:
                test = ABTest.model_validate(json.loads(path.read_text()))
                if test.status == "running":
                    tests.append(test)
            except Exception:
                continue
        return tests

    def _save_test(self, test: ABTest) -> None:
        """Save test to disk."""
        path = self.tests_dir / f"{test.id}.json"
        path.write_text(json.dumps(test.model_dump(mode='json'), indent=2, default=str))

    def _load_test(self, test_id: str) -> ABTest | None:
        """Load test from disk."""
        path = self.tests_dir / f"{test_id}.json"
        if not path.exists():
            return None
        try:
            return ABTest.model_validate(json.loads(path.read_text()))
        except Exception:
            return None


# =============================================================================
# Unified Deep Learning System
# =============================================================================

class DeepLearningSystem:
    """
    Unified system for deep behavioral learning.

    Coordinates:
    - Strategy selection and refinement
    - Prompt template optimization
    - Action sequence detection
    - Credit assignment
    - A/B testing
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.learning_dir = body_dir / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)

        # Initialize subsystems
        self.strategies = StrategyLibrary(body_dir)
        self.prompts = PromptRefinementEngine(body_dir)
        self.sequences = SequenceDetector(body_dir)
        self.credit = CreditAssigner(body_dir)
        self.ab_tests = ABTestRunner(body_dir)

        # Active context
        self._current_strategy: str | None = None
        self._current_actions: list[str] = []

    def start_task(
        self,
        task_description: str,
        context: dict[str, Any],
        domain: str | None = None,
    ) -> Strategy | None:
        """
        Start a task - select strategy and prepare tracking.

        Returns selected strategy (or None if no match).
        """
        self._current_actions = []

        # Select strategy
        strategy = self.strategies.select_strategy(task_description, context, domain)
        if strategy:
            self._current_strategy = strategy.id

        return strategy

    def observe_action(self, action: str, context: str = "") -> None:
        """Observe an action during task execution."""
        self._current_actions.append(action)
        self.sequences.observe_action(action, context)

    def complete_task(
        self,
        success: bool,
        context: dict[str, Any],
        duration_seconds: float,
        tokens_used: int = 0,
        quality_score: float | None = None,
    ) -> None:
        """
        Complete a task - record outcomes and assign credit.
        """
        # Record strategy application
        if self._current_strategy:
            outcome = StrategyOutcome.SUCCESS if success else StrategyOutcome.FAILURE
            self.strategies.record_application(
                self._current_strategy,
                context,
                outcome,
                duration_seconds,
                tokens_used,
            )

        # Create and assign feedback
        feedback = Feedback(
            id=hashlib.md5(f"{datetime.now()}:{success}".encode()).hexdigest()[:12],
            feedback_type=FeedbackType.SUCCESS if success else FeedbackType.FAILURE,
            value=1.0 if success else -1.0,
            target="task",
            context=context,
            contributing_actions=self._current_actions.copy(),
            contributing_strategies=[self._current_strategy] if self._current_strategy else [],
        )
        self.credit.assign_credit(feedback)

        # Check for synthesizable sequences
        new_sequences = self.sequences.detect_sequences()
        for seq in new_sequences:
            if not seq.is_synthesized and seq.frequency >= 5:
                # Auto-synthesize high-frequency sequences
                self.sequences.synthesize_procedure(
                    seq,
                    name=f"auto-procedure-{seq.id}",
                    description=f"Auto-detected sequence ({seq.frequency} occurrences)",
                )

        # Reset
        self._current_strategy = None
        self._current_actions = []

    def get_learning_summary(self) -> str:
        """Get a summary of what the agent has learned."""
        lines = ["# Learning Summary\n"]

        # Best strategies
        best = self.strategies.get_best_strategies(top_k=5)
        if best:
            lines.append("## Top Strategies")
            for s in best:
                lines.append(f"- **{s.name}**: {s.success_rate:.0%} success ({s.times_applied} uses)")
            lines.append("")

        # Best actions
        best_actions = self.credit.get_best_actions(top_k=5)
        if best_actions:
            lines.append("## Best Actions (by credit)")
            for action, credit in best_actions:
                lines.append(f"- {action}: {credit:+.2f}")
            lines.append("")

        # Worst actions
        worst_actions = self.credit.get_worst_actions(top_k=3)
        if worst_actions and worst_actions[0][1] < 0:
            lines.append("## Actions to Avoid")
            for action, credit in worst_actions:
                if credit < 0:
                    lines.append(f"- {action}: {credit:+.2f}")
            lines.append("")

        # Active A/B tests
        active_tests = self.ab_tests.get_active_tests()
        if active_tests:
            lines.append("## Active Experiments")
            for test in active_tests:
                total = test.control_trials + test.treatment_trials
                lines.append(f"- {test.name}: {total} trials")
            lines.append("")

        return "\n".join(lines)
