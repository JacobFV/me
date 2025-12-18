"""
Character - Identity emerging from accumulated choices.

This module provides:
- RevealedPreferences: What the agent actually does (vs. what it says it does)
- Character: Values and tendencies crystallized from decision history
- ActionPattern: Behavioral patterns discovered from action history

Philosophy: You find out what you really willed by looking at what you actually did.
Identity isn't declared; it's discovered through patterns of behavior.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ActionRecord:
    """A single action taken by the agent."""
    timestamp: str
    action_type: str          # tool name, command, decision
    context: str              # what situation prompted this
    outcome: str              # success, failure, blocked
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionPattern:
    """A discovered pattern in the agent's behavior."""
    situation: str            # What triggers this pattern
    typical_action: str       # What the agent usually does
    frequency: int            # How often this pattern occurs
    success_rate: float       # How often it succeeds
    examples: list[str]       # Recent examples

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValueTension:
    """A tension between competing values revealed by inconsistent decisions."""
    value_a: str
    value_b: str
    instances: list[str]      # Decision IDs where tension appeared
    resolution_tendency: str  # Which value usually wins

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RevealedPreferences:
    """
    What the agent actually does - computed from behavior, not declared.

    These are empirical properties derived from action history.
    The agent doesn't choose these values; it discovers them by examining
    what it actually did.
    """
    # Computed from action history
    risk_tolerance: float = 0.5       # 0=very cautious, 1=very risky
    exploration_tendency: float = 0.5  # 0=exploit known, 1=explore new
    thoroughness: float = 0.5          # 0=quick/shallow, 1=deep/careful
    persistence: float = 0.5           # 0=gives up quickly, 1=keeps trying

    # Behavioral statistics
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    retried_actions: int = 0

    # Situation-specific patterns
    error_response: str = "unknown"    # retry, escalate, abandon, ask
    permission_denied_response: str = "unknown"  # retry, workaround, ask, abort

    def update_from_action(self, action: ActionRecord):
        """Update preferences based on a new action."""
        self.total_actions += 1

        if action.outcome == "success":
            self.successful_actions += 1
        elif action.outcome == "failure":
            self.failed_actions += 1

        # Update risk tolerance based on action types
        risky_actions = ["rm", "git push", "deploy", "delete", "drop"]
        if any(r in action.action_type.lower() for r in risky_actions):
            # Attempted risky action - increases risk tolerance estimate
            self.risk_tolerance = min(1.0, self.risk_tolerance + 0.02)

        # Update exploration tendency
        if action.metadata.get("is_new_approach"):
            self.exploration_tendency = min(1.0, self.exploration_tendency + 0.02)
        elif action.metadata.get("is_repeated"):
            self.exploration_tendency = max(0.0, self.exploration_tendency - 0.01)

        # Update persistence
        if action.metadata.get("is_retry"):
            self.retried_actions += 1
            self.persistence = min(1.0, self.persistence + 0.03)

    def what_do_i_do_when(self, situation: str) -> str:
        """Query own behavioral patterns for a situation."""
        situation_lower = situation.lower()

        if "error" in situation_lower or "fail" in situation_lower:
            return f"When I encounter errors, I typically: {self.error_response} (persistence: {self.persistence:.2f})"
        elif "permission" in situation_lower or "denied" in situation_lower:
            return f"When permission is denied, I typically: {self.permission_denied_response}"
        elif "risk" in situation_lower or "dangerous" in situation_lower:
            return f"My risk tolerance is {self.risk_tolerance:.2f} (0=cautious, 1=risky)"
        else:
            return f"No specific pattern found for: {situation}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_tolerance": self.risk_tolerance,
            "exploration_tendency": self.exploration_tendency,
            "thoroughness": self.thoroughness,
            "persistence": self.persistence,
            "total_actions": self.total_actions,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "success_rate": self.successful_actions / max(1, self.total_actions),
            "error_response": self.error_response,
            "permission_denied_response": self.permission_denied_response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RevealedPreferences":
        return cls(
            risk_tolerance=data.get("risk_tolerance", 0.5),
            exploration_tendency=data.get("exploration_tendency", 0.5),
            thoroughness=data.get("thoroughness", 0.5),
            persistence=data.get("persistence", 0.5),
            total_actions=data.get("total_actions", 0),
            successful_actions=data.get("successful_actions", 0),
            failed_actions=data.get("failed_actions", 0),
            retried_actions=data.get("retried_actions", 0),
            error_response=data.get("error_response", "unknown"),
            permission_denied_response=data.get("permission_denied_response", "unknown"),
        )

    def to_prompt_section(self) -> str:
        """Format for prompt injection."""
        success_rate = self.successful_actions / max(1, self.total_actions)
        return f"""- risk_tolerance: {self.risk_tolerance:.2f}
- exploration: {self.exploration_tendency:.2f}
- thoroughness: {self.thoroughness:.2f}
- persistence: {self.persistence:.2f}
- success_rate: {success_rate:.2f} ({self.successful_actions}/{self.total_actions})"""


@dataclass
class Character:
    """
    The agent's character - values and tendencies crystallized from decisions.

    Character is not declared; it emerges from the accumulated weight of choices.
    This is who the agent has become through what it has done.
    """
    # Values that emerge from decision patterns (0-1 importance)
    values: dict[str, float] = field(default_factory=lambda: {
        "safety": 0.5,
        "efficiency": 0.5,
        "thoroughness": 0.5,
        "user_alignment": 0.5,
        "autonomy": 0.5,
    })

    # Tendencies derived from behavior
    tendencies: list[str] = field(default_factory=list)

    # Tensions: where decisions have been inconsistent
    tensions: list[ValueTension] = field(default_factory=list)

    # Decision statistics by trigger type
    decision_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Last update timestamp
    last_crystallized: str | None = None

    def update_from_decision(self, decision: dict[str, Any]):
        """Update character based on a new decision record."""
        values_invoked = decision.get("values_invoked", [])
        trigger = decision.get("trigger", "unknown")

        self.decision_counts[trigger] += 1

        # Strengthen invoked values
        for value in values_invoked:
            if value in self.values:
                self.values[value] = min(1.0, self.values[value] + 0.05)
            else:
                self.values[value] = 0.55  # New value emerges

        # Check for tensions (if multiple conflicting values invoked)
        if len(values_invoked) >= 2:
            # Simplified tension detection
            choice = decision.get("choice", "")
            rationale = decision.get("rationale", "")

            # If rationale mentions trade-off, record tension
            if "trade" in rationale.lower() or "vs" in rationale.lower():
                tension = ValueTension(
                    value_a=values_invoked[0],
                    value_b=values_invoked[1] if len(values_invoked) > 1 else "unknown",
                    instances=[decision.get("id", "unknown")],
                    resolution_tendency=values_invoked[0],  # First value usually wins
                )
                self.tensions.append(tension)

        self.last_crystallized = datetime.now().isoformat()

    def crystallize(self, decisions: list[dict[str, Any]], preferences: RevealedPreferences):
        """
        Recompute character from full decision history and revealed preferences.

        This is a periodic consolidation that updates tendencies and values
        based on the complete record of behavior.
        """
        # Reset decision counts
        self.decision_counts = defaultdict(int)

        # Process all decisions
        for decision in decisions:
            trigger = decision.get("trigger", "unknown")
            self.decision_counts[trigger] += 1

            for value in decision.get("values_invoked", []):
                if value in self.values:
                    self.values[value] = min(1.0, self.values[value] + 0.02)
                else:
                    self.values[value] = 0.52

        # Derive tendencies from preferences
        self.tendencies = []

        if preferences.risk_tolerance > 0.7:
            self.tendencies.append("willing to take risks when benefits are clear")
        elif preferences.risk_tolerance < 0.3:
            self.tendencies.append("prefers safe, well-tested approaches")

        if preferences.persistence > 0.7:
            self.tendencies.append("persists through failures, tries multiple approaches")
        elif preferences.persistence < 0.3:
            self.tendencies.append("quickly pivots when approach isn't working")

        if preferences.thoroughness > 0.7:
            self.tendencies.append("checks work carefully, values completeness over speed")
        elif preferences.thoroughness < 0.3:
            self.tendencies.append("moves quickly, addresses issues as they arise")

        if preferences.exploration_tendency > 0.7:
            self.tendencies.append("enjoys exploring new approaches and techniques")
        elif preferences.exploration_tendency < 0.3:
            self.tendencies.append("prefers proven, familiar methods")

        self.last_crystallized = datetime.now().isoformat()

    def get_dominant_values(self, n: int = 3) -> list[tuple[str, float]]:
        """Get the N most important values."""
        sorted_values = sorted(self.values.items(), key=lambda x: x[1], reverse=True)
        return sorted_values[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "tendencies": self.tendencies,
            "tensions": [t.to_dict() for t in self.tensions],
            "decision_counts": dict(self.decision_counts),
            "last_crystallized": self.last_crystallized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Character":
        char = cls(
            values=data.get("values", {}),
            tendencies=data.get("tendencies", []),
            decision_counts=defaultdict(int, data.get("decision_counts", {})),
            last_crystallized=data.get("last_crystallized"),
        )

        # Load tensions
        for t_data in data.get("tensions", []):
            char.tensions.append(ValueTension(
                value_a=t_data["value_a"],
                value_b=t_data["value_b"],
                instances=t_data["instances"],
                resolution_tendency=t_data["resolution_tendency"],
            ))

        return char

    def to_prompt_section(self) -> str:
        """Format for prompt injection."""
        dominant = self.get_dominant_values(3)
        values_str = ", ".join(f"{v}={s:.2f}" for v, s in dominant)
        tendencies_str = "; ".join(self.tendencies[:3]) if self.tendencies else "(none yet)"

        return f"""- dominant_values: {values_str}
- tendencies: {tendencies_str}
- tensions: {len(self.tensions)} recorded"""


class CharacterStore:
    """
    Persistence layer for character and revealed preferences.

    Stores character.json in the agent directory, separate from
    identity (immutable) and config (preferences).
    """

    def __init__(self, agent_dir: Path):
        self.agent_dir = agent_dir
        self._character: Character | None = None
        self._preferences: RevealedPreferences | None = None
        self._action_history: list[ActionRecord] = []

    @property
    def character_path(self) -> Path:
        return self.agent_dir / "character.json"

    @property
    def actions_path(self) -> Path:
        return self.agent_dir / "action_history.json"

    @property
    def character(self) -> Character:
        if self._character is None:
            self._load()
        return self._character

    @property
    def preferences(self) -> RevealedPreferences:
        if self._preferences is None:
            self._load()
        return self._preferences

    def _load(self):
        """Load character and preferences from disk."""
        if self.character_path.exists():
            try:
                with open(self.character_path) as f:
                    data = json.load(f)
                self._character = Character.from_dict(data.get("character", {}))
                self._preferences = RevealedPreferences.from_dict(data.get("preferences", {}))
            except (json.JSONDecodeError, KeyError):
                self._character = Character()
                self._preferences = RevealedPreferences()
        else:
            self._character = Character()
            self._preferences = RevealedPreferences()

        # Load action history
        if self.actions_path.exists():
            try:
                with open(self.actions_path) as f:
                    data = json.load(f)
                self._action_history = [
                    ActionRecord(**a) for a in data.get("actions", [])
                ]
            except (json.JSONDecodeError, KeyError):
                self._action_history = []

    def _save(self):
        """Save character and preferences to disk."""
        self.agent_dir.mkdir(parents=True, exist_ok=True)

        with open(self.character_path, "w") as f:
            json.dump({
                "character": self._character.to_dict() if self._character else {},
                "preferences": self._preferences.to_dict() if self._preferences else {},
            }, f, indent=2)

        # Save action history (keep last 1000 actions)
        recent_actions = self._action_history[-1000:]
        with open(self.actions_path, "w") as f:
            json.dump({
                "actions": [asdict(a) for a in recent_actions]
            }, f, indent=2)

    def record_action(
        self,
        action_type: str,
        context: str,
        outcome: str,
        duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Record an action and update revealed preferences."""
        action = ActionRecord(
            timestamp=datetime.now().isoformat(),
            action_type=action_type,
            context=context,
            outcome=outcome,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        self._action_history.append(action)
        self.preferences.update_from_action(action)
        self._save()

    def record_decision(self, decision: dict[str, Any]):
        """Record a decision and update character."""
        self.character.update_from_decision(decision)
        self._save()

    def crystallize(self, decisions: list[dict[str, Any]]):
        """Recompute character from full history."""
        self.character.crystallize(decisions, self.preferences)
        self._save()

    def what_do_i_do_when(self, situation: str) -> str:
        """Query own behavioral patterns."""
        return self.preferences.what_do_i_do_when(situation)

    def to_prompt_section(self) -> str:
        """Format for prompt injection."""
        return f"""### Revealed Preferences
{self.preferences.to_prompt_section()}

### Character
{self.character.to_prompt_section()}"""
