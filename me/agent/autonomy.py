"""
Autonomy Modes - Different levels of agent independence.

This module defines how much freedom the agent has to act without
human confirmation. The autonomy level affects:
1. Which actions require confirmation
2. How much the agent can modify its own configuration
3. Whether it can pursue intrinsic goals
4. Risk thresholds for different operations

Philosophy: Autonomy is not binary. It exists on a spectrum from
fully supervised (human confirms everything) to fully autonomous
(agent acts within defined boundaries). The agent should be able
to operate at different autonomy levels based on context and trust.
"""

from __future__ import annotations

from enum import Enum
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
import json

from pydantic import BaseModel, Field


class AutonomyLevel(str, Enum):
    """Levels of agent autonomy."""

    SUPERVISED = "supervised"
    """Human confirms every action. Training wheels mode."""

    ASSISTED = "assisted"
    """Human confirms risky actions. Default mode."""

    AUTONOMOUS = "autonomous"
    """Agent acts freely within defined boundaries."""

    LEARNING = "learning"
    """Agent explicitly learning - more exploration allowed."""


class ActionRisk(str, Enum):
    """Risk levels for different actions."""
    SAFE = "safe"           # Read operations, queries
    LOW = "low"             # Write to user files, run safe commands
    MEDIUM = "medium"       # System modifications, network calls
    HIGH = "high"           # Destructive ops, external services
    CRITICAL = "critical"   # Irreversible operations


class AutonomyPolicy(BaseModel):
    """Policy defining what's allowed at each autonomy level."""

    # Base autonomy level
    level: AutonomyLevel = AutonomyLevel.ASSISTED

    # Risk thresholds - actions above this risk require confirmation
    risk_threshold: ActionRisk = ActionRisk.LOW

    # Specific permissions
    can_modify_own_config: bool = False
    can_pursue_intrinsic_goals: bool = False
    can_create_files: bool = True
    can_delete_files: bool = False
    can_run_commands: bool = True
    can_access_network: bool = True
    can_spawn_agents: bool = False
    can_modify_memory: bool = True

    # Self-modification boundaries
    can_modify_character: bool = False
    can_modify_skills: bool = True
    can_create_daemons: bool = False

    # Learning permissions
    can_experiment: bool = False
    max_experiment_risk: ActionRisk = ActionRisk.LOW

    # Time limits (seconds)
    max_task_duration: int | None = None
    max_autonomous_duration: int = 3600  # 1 hour

    # Resource limits
    max_tokens_per_task: int | None = None
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB


# Default policies for each autonomy level
DEFAULT_POLICIES: dict[AutonomyLevel, AutonomyPolicy] = {
    AutonomyLevel.SUPERVISED: AutonomyPolicy(
        level=AutonomyLevel.SUPERVISED,
        risk_threshold=ActionRisk.SAFE,
        can_modify_own_config=False,
        can_pursue_intrinsic_goals=False,
        can_delete_files=False,
        can_spawn_agents=False,
        can_modify_character=False,
        can_create_daemons=False,
        can_experiment=False,
    ),
    AutonomyLevel.ASSISTED: AutonomyPolicy(
        level=AutonomyLevel.ASSISTED,
        risk_threshold=ActionRisk.LOW,
        can_modify_own_config=False,
        can_pursue_intrinsic_goals=False,
        can_delete_files=True,
        can_spawn_agents=False,
        can_modify_character=False,
        can_create_daemons=False,
        can_experiment=False,
    ),
    AutonomyLevel.AUTONOMOUS: AutonomyPolicy(
        level=AutonomyLevel.AUTONOMOUS,
        risk_threshold=ActionRisk.MEDIUM,
        can_modify_own_config=True,
        can_pursue_intrinsic_goals=True,
        can_delete_files=True,
        can_spawn_agents=True,
        can_modify_character=True,
        can_create_daemons=True,
        can_experiment=True,
        max_experiment_risk=ActionRisk.LOW,
    ),
    AutonomyLevel.LEARNING: AutonomyPolicy(
        level=AutonomyLevel.LEARNING,
        risk_threshold=ActionRisk.MEDIUM,
        can_modify_own_config=True,
        can_pursue_intrinsic_goals=True,
        can_delete_files=True,
        can_spawn_agents=False,
        can_modify_character=True,
        can_create_daemons=True,
        can_experiment=True,
        max_experiment_risk=ActionRisk.MEDIUM,
    ),
}


class ActionRequest(BaseModel):
    """A request to perform an action."""
    action_type: str
    description: str
    risk_level: ActionRisk
    target: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ActionDecision(BaseModel):
    """Decision about whether an action is allowed."""
    allowed: bool
    requires_confirmation: bool = False
    reason: str
    risk_level: ActionRisk
    policy_level: AutonomyLevel


class AutonomyManager:
    """
    Manages agent autonomy and action permissions.

    Evaluates action requests against the current policy and
    determines whether they're allowed, require confirmation,
    or are blocked.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.config_file = body_dir / "autonomy.json"
        self._policy: AutonomyPolicy | None = None
        self._load_policy()

    def _load_policy(self) -> None:
        """Load policy from file or use default."""
        if self.config_file.exists():
            try:
                data = json.loads(self.config_file.read_text())
                self._policy = AutonomyPolicy.model_validate(data)
            except Exception:
                self._policy = DEFAULT_POLICIES[AutonomyLevel.ASSISTED]
        else:
            self._policy = DEFAULT_POLICIES[AutonomyLevel.ASSISTED]

    def _save_policy(self) -> None:
        """Save policy to file."""
        if self._policy:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.write_text(
                json.dumps(self._policy.model_dump(mode='json'), indent=2)
            )

    @property
    def policy(self) -> AutonomyPolicy:
        """Get current policy."""
        if not self._policy:
            self._load_policy()
        return self._policy  # type: ignore

    @property
    def level(self) -> AutonomyLevel:
        """Get current autonomy level."""
        return self.policy.level

    def set_level(self, level: AutonomyLevel) -> None:
        """Set autonomy level (loads default policy for that level)."""
        self._policy = DEFAULT_POLICIES[level].model_copy()
        self._save_policy()

    def set_policy(self, policy: AutonomyPolicy) -> None:
        """Set a custom policy."""
        self._policy = policy
        self._save_policy()

    def evaluate_action(self, request: ActionRequest) -> ActionDecision:
        """Evaluate whether an action is allowed."""
        policy = self.policy

        # Check risk level against threshold
        risk_order = list(ActionRisk)
        request_risk_idx = risk_order.index(request.risk_level)
        threshold_idx = risk_order.index(policy.risk_threshold)

        # Actions at or below threshold are allowed
        if request_risk_idx <= threshold_idx:
            return ActionDecision(
                allowed=True,
                requires_confirmation=False,
                reason="Within risk threshold",
                risk_level=request.risk_level,
                policy_level=policy.level,
            )

        # Actions above threshold but not critical require confirmation
        if request.risk_level != ActionRisk.CRITICAL:
            return ActionDecision(
                allowed=True,
                requires_confirmation=True,
                reason=f"Risk level {request.risk_level.value} above threshold {policy.risk_threshold.value}",
                risk_level=request.risk_level,
                policy_level=policy.level,
            )

        # Critical actions in supervised/assisted mode are blocked
        if policy.level in (AutonomyLevel.SUPERVISED, AutonomyLevel.ASSISTED):
            return ActionDecision(
                allowed=False,
                requires_confirmation=False,
                reason="Critical actions blocked at current autonomy level",
                risk_level=request.risk_level,
                policy_level=policy.level,
            )

        # Critical actions in autonomous mode still require confirmation
        return ActionDecision(
            allowed=True,
            requires_confirmation=True,
            reason="Critical action requires confirmation even in autonomous mode",
            risk_level=request.risk_level,
            policy_level=policy.level,
        )

    def can_perform(self, action_type: str) -> tuple[bool, str]:
        """Quick check if an action type is allowed."""
        policy = self.policy

        checks = {
            "modify_config": policy.can_modify_own_config,
            "pursue_goals": policy.can_pursue_intrinsic_goals,
            "create_file": policy.can_create_files,
            "delete_file": policy.can_delete_files,
            "run_command": policy.can_run_commands,
            "network": policy.can_access_network,
            "spawn_agent": policy.can_spawn_agents,
            "modify_memory": policy.can_modify_memory,
            "modify_character": policy.can_modify_character,
            "modify_skills": policy.can_modify_skills,
            "create_daemon": policy.can_create_daemons,
            "experiment": policy.can_experiment,
        }

        if action_type in checks:
            allowed = checks[action_type]
            if not allowed:
                return False, f"Action '{action_type}' not allowed at {policy.level.value} level"

        return True, "Allowed"

    def get_prompt_section(self) -> str:
        """Get autonomy info for prompt injection."""
        policy = self.policy

        lines = [f"### Autonomy: {policy.level.value}"]
        lines.append(f"Risk threshold: {policy.risk_threshold.value}")

        restrictions = []
        if not policy.can_delete_files:
            restrictions.append("no file deletion")
        if not policy.can_spawn_agents:
            restrictions.append("no spawning agents")
        if not policy.can_pursue_intrinsic_goals:
            restrictions.append("no self-directed goals")
        if not policy.can_experiment:
            restrictions.append("no experiments")

        if restrictions:
            lines.append(f"Restrictions: {', '.join(restrictions)}")

        return "\n".join(lines)


class ConfirmationRequest(BaseModel):
    """A request for human confirmation."""
    id: str
    action: ActionRequest
    decision: ActionDecision
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    confirmed: bool | None = None
    confirmed_at: datetime | None = None
    confirmed_by: str | None = None


class ConfirmationQueue:
    """
    Queue for actions requiring human confirmation.

    When autonomy policy requires confirmation, actions are queued here
    until a human approves or rejects them.
    """

    def __init__(self, body_dir: Path):
        self.body_dir = body_dir
        self.queue_dir = body_dir / "autonomy" / "confirmations"
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def request_confirmation(
        self,
        action: ActionRequest,
        decision: ActionDecision,
    ) -> ConfirmationRequest:
        """Add an action to the confirmation queue."""
        import hashlib
        req_id = hashlib.md5(
            f"{action.action_type}:{action.description}:{action.timestamp}".encode()
        ).hexdigest()[:12]

        request = ConfirmationRequest(
            id=req_id,
            action=action,
            decision=decision,
        )

        path = self.queue_dir / f"pending-{req_id}.json"
        path.write_text(json.dumps(request.model_dump(mode='json'), indent=2, default=str))

        return request

    def get_pending(self) -> list[ConfirmationRequest]:
        """Get all pending confirmation requests."""
        requests = []
        for path in self.queue_dir.glob("pending-*.json"):
            try:
                data = json.loads(path.read_text())
                requests.append(ConfirmationRequest.model_validate(data))
            except Exception:
                continue
        return sorted(requests, key=lambda r: r.created_at)

    def confirm(self, req_id: str, confirmed_by: str = "user") -> bool:
        """Confirm a pending action."""
        return self._resolve(req_id, True, confirmed_by)

    def reject(self, req_id: str, confirmed_by: str = "user") -> bool:
        """Reject a pending action."""
        return self._resolve(req_id, False, confirmed_by)

    def _resolve(self, req_id: str, confirmed: bool, confirmed_by: str) -> bool:
        """Resolve a confirmation request."""
        pending_path = self.queue_dir / f"pending-{req_id}.json"
        if not pending_path.exists():
            return False

        try:
            data = json.loads(pending_path.read_text())
            request = ConfirmationRequest.model_validate(data)
            request.confirmed = confirmed
            request.confirmed_at = datetime.now(UTC)
            request.confirmed_by = confirmed_by

            # Move to resolved
            resolved_path = self.queue_dir / f"resolved-{req_id}.json"
            resolved_path.write_text(
                json.dumps(request.model_dump(mode='json'), indent=2, default=str)
            )
            pending_path.unlink()

            return True
        except Exception:
            return False

    def check_confirmation(self, req_id: str) -> bool | None:
        """Check if a request has been confirmed. Returns None if still pending."""
        resolved_path = self.queue_dir / f"resolved-{req_id}.json"
        if resolved_path.exists():
            data = json.loads(resolved_path.read_text())
            return data.get("confirmed")

        pending_path = self.queue_dir / f"pending-{req_id}.json"
        if pending_path.exists():
            return None  # Still pending

        return None  # Not found


# =============================================================================
# Action Risk Classification
# =============================================================================

# Action type to risk level mapping
ACTION_RISK_MAP: dict[str, ActionRisk] = {
    # Safe actions
    "read_file": ActionRisk.SAFE,
    "list_directory": ActionRisk.SAFE,
    "search": ActionRisk.SAFE,
    "query": ActionRisk.SAFE,
    "recall_memory": ActionRisk.SAFE,

    # Low risk
    "write_file": ActionRisk.LOW,
    "create_file": ActionRisk.LOW,
    "store_memory": ActionRisk.LOW,
    "run_safe_command": ActionRisk.LOW,

    # Medium risk
    "delete_file": ActionRisk.MEDIUM,
    "run_command": ActionRisk.MEDIUM,
    "network_request": ActionRisk.MEDIUM,
    "modify_config": ActionRisk.MEDIUM,
    "create_daemon": ActionRisk.MEDIUM,

    # High risk
    "run_as_root": ActionRisk.HIGH,
    "modify_system": ActionRisk.HIGH,
    "spawn_agent": ActionRisk.HIGH,
    "external_api": ActionRisk.HIGH,

    # Critical
    "delete_recursive": ActionRisk.CRITICAL,
    "git_push_force": ActionRisk.CRITICAL,
    "modify_identity": ActionRisk.CRITICAL,
    "shutdown": ActionRisk.CRITICAL,
}


def classify_action_risk(action_type: str, context: dict[str, Any] | None = None) -> ActionRisk:
    """Classify the risk level of an action."""
    context = context or {}

    # Check explicit mapping
    if action_type in ACTION_RISK_MAP:
        base_risk = ACTION_RISK_MAP[action_type]
    else:
        base_risk = ActionRisk.MEDIUM  # Default to medium for unknown

    # Adjust based on context
    if context.get("target_is_system_file"):
        base_risk = ActionRisk.HIGH
    if context.get("involves_secrets"):
        base_risk = ActionRisk.HIGH
    if context.get("irreversible"):
        base_risk = ActionRisk.CRITICAL

    return base_risk


# =============================================================================
# Autonomy Tools (for agent self-use)
# =============================================================================

def get_autonomy_tools() -> list[dict[str, Any]]:
    """Get tool definitions for autonomy management."""
    return [
        {
            "name": "check_autonomy",
            "description": "Check current autonomy level and what actions are allowed",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "request_autonomy_change",
            "description": "Request a change in autonomy level (requires human approval)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "requested_level": {
                        "type": "string",
                        "enum": ["supervised", "assisted", "autonomous", "learning"],
                        "description": "The autonomy level to request",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this autonomy level is needed",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "How long to maintain this level (optional)",
                    },
                },
                "required": ["requested_level", "reason"],
            },
        },
        {
            "name": "check_pending_confirmations",
            "description": "Check if there are actions waiting for human confirmation",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
    ]
