"""
Embodiment system - the agent's situated, physically-constrained existence.

This module provides the agent with:
- Location: where the agent "is" (device/session/transport/capabilities)
- CapabilitySurface: what the agent can do from this location
- RouteTrace: history of movement for continuity
- SomaticState: autonomic self-regulation (stress, fatigue, reflexes)
- WorkingSet: active commitments and open loops
- DecisionRecord: identity-preserving record of significant choices

Philosophy: The CLI/runtime is the agent's sensorimotor substrate.
The LLM is cortex; this module is spinal cord + autonomic nervous system.
"""

from __future__ import annotations

import hashlib
import os
import json
import subprocess
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from me.agent.core import AgentConfig


# =============================================================================
# Location - Where the agent exists
# =============================================================================

class Transport(Enum):
    """How the agent arrived at this location."""
    LOCAL = "local"
    SSH = "ssh"
    CONTAINER = "container"
    WSL = "wsl"
    TMUX = "tmux"
    UNKNOWN = "unknown"


class TrustTier(Enum):
    """Trust level of the current environment."""
    HIGH = "high"      # Local dev machine, trusted infra
    MEDIUM = "medium"  # Corp network, known servers
    LOW = "low"        # Public/unknown, untrusted


class NetworkZone(Enum):
    """Network context."""
    OFFLINE = "offline"
    LOCAL = "local"      # localhost only
    CORP = "corp"        # internal network
    PUBLIC = "public"    # internet-accessible
    UNKNOWN = "unknown"


@dataclass
class Location:
    """
    The agent's current location - not just a path, but a full situational context.

    This defines the boundary of what can be sensed and acted upon.
    """
    device_id: str              # Stable machine identifier
    hostname: str
    user: str
    cwd: Path
    session_id: str             # tty/tmux pane/ssh connection id
    transport: Transport        # How we got here
    container_id: str | None    # If inside a container
    venv: str | None            # Active virtual environment
    repo_root: Path | None      # Git repo root if in one
    trust_tier: TrustTier
    network_zone: NetworkZone

    @classmethod
    def detect(cls, cwd: Path | None = None) -> "Location":
        """Detect current location from environment."""
        cwd = cwd or Path.cwd()

        # Device ID: hash of hostname + machine-id if available
        machine_id = ""
        if Path("/etc/machine-id").exists():
            machine_id = Path("/etc/machine-id").read_text().strip()
        elif Path("/var/lib/dbus/machine-id").exists():
            machine_id = Path("/var/lib/dbus/machine-id").read_text().strip()

        import socket
        hostname = socket.gethostname()
        device_id = hashlib.sha256(f"{hostname}:{machine_id}".encode()).hexdigest()[:12]

        # Session ID: TTY or tmux pane
        session_id = os.environ.get("TMUX_PANE", "") or os.ttyname(0) if os.isatty(0) else f"pid-{os.getpid()}"

        # Transport detection
        transport = Transport.LOCAL
        if os.environ.get("SSH_CONNECTION"):
            transport = Transport.SSH
        elif os.environ.get("TMUX"):
            transport = Transport.TMUX
        elif Path("/.dockerenv").exists():
            transport = Transport.CONTAINER
        elif "microsoft" in os.uname().release.lower():
            transport = Transport.WSL

        # Container ID
        container_id = None
        if Path("/.dockerenv").exists():
            # Try to get container ID from cgroup
            try:
                with open("/proc/self/cgroup") as f:
                    for line in f:
                        if "docker" in line:
                            container_id = line.strip().split("/")[-1][:12]
                            break
            except:
                pass

        # Virtual environment
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            venv = Path(venv).name

        # Git repo root
        repo_root = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, cwd=cwd
            )
            if result.returncode == 0:
                repo_root = Path(result.stdout.strip())
        except:
            pass

        # Trust tier (conservative default)
        trust_tier = TrustTier.MEDIUM
        if transport == Transport.LOCAL and not container_id:
            trust_tier = TrustTier.HIGH
        elif transport == Transport.SSH:
            trust_tier = TrustTier.MEDIUM
        elif container_id:
            trust_tier = TrustTier.MEDIUM

        # Network zone
        network_zone = NetworkZone.UNKNOWN
        try:
            # Simple check: can we reach localhost?
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.1)
            s.connect(("127.0.0.1", 22))  # Try SSH port
            s.close()
            network_zone = NetworkZone.LOCAL
        except:
            network_zone = NetworkZone.LOCAL  # Assume local if check fails

        return cls(
            device_id=device_id,
            hostname=hostname,
            user=os.environ.get("USER", "unknown"),
            cwd=cwd,
            session_id=session_id,
            transport=transport,
            container_id=container_id,
            venv=venv,
            repo_root=repo_root,
            trust_tier=trust_tier,
            network_zone=network_zone,
        )

    @property
    def location_hash(self) -> str:
        """Short hash identifying this location for route traces."""
        return hashlib.sha256(
            f"{self.device_id}:{self.session_id}:{self.cwd}".encode()
        ).hexdigest()[:8]

    def to_prompt_section(self) -> str:
        """Format for prompt injection."""
        container_str = self.container_id or "none"
        venv_str = self.venv or "none"
        repo_str = str(self.repo_root) if self.repo_root else "none"

        return f"""- device: {self.device_id} ({self.hostname})
- session: {self.session_id} via {self.transport.value}
- user: {self.user}
- cwd: {self.cwd}
- container: {container_str} | venv: {venv_str}
- repo: {repo_str}
- trust: {self.trust_tier.value} | network: {self.network_zone.value}"""


# =============================================================================
# CapabilitySurface - What the agent can do
# =============================================================================

@dataclass
class CapabilitySurface:
    """
    Explicit capabilities available to the agent.

    Derived from permission_mode + environment + location.
    The agent reasons about these, not raw permission strings.
    """
    fs_read: bool = True
    fs_write: bool = True
    exec: bool = True
    network: bool = True
    secrets_read: bool = False
    secrets_write: bool = False
    destructive_ops: bool = False   # rm -rf, git push --force, etc.
    process_control: bool = True    # kill, signal, attach
    spawn_children: bool = True
    max_risk: str = "medium"        # low/medium/high
    rate_limits: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_permission_mode(
        cls,
        mode: str,
        trust_tier: TrustTier = TrustTier.MEDIUM
    ) -> "CapabilitySurface":
        """Derive capabilities from permission mode and trust tier."""

        # Base capabilities by mode
        if mode == "acceptEdits":
            caps = cls(
                fs_read=True,
                fs_write=True,
                exec=True,
                network=True,
                destructive_ops=False,
                max_risk="medium",
            )
        elif mode == "full":
            caps = cls(
                fs_read=True,
                fs_write=True,
                exec=True,
                network=True,
                destructive_ops=True,
                secrets_read=True,
                max_risk="high",
            )
        elif mode == "plan":
            caps = cls(
                fs_read=True,
                fs_write=False,
                exec=False,
                network=False,
                destructive_ops=False,
                max_risk="low",
            )
        else:  # default/unknown
            caps = cls(max_risk="medium")

        # Adjust by trust tier
        if trust_tier == TrustTier.LOW:
            caps.destructive_ops = False
            caps.secrets_read = False
            caps.secrets_write = False
            caps.max_risk = "low"
        elif trust_tier == TrustTier.HIGH:
            caps.secrets_read = True

        return caps

    def to_prompt_section(self) -> str:
        """Format for prompt injection."""
        rates = ", ".join(f"{k}={v}" for k, v in self.rate_limits.items()) or "none"
        return f"""- fs: read={self.fs_read} write={self.fs_write}
- exec: {self.exec} | process_control: {self.process_control}
- network: {self.network} | spawn_children: {self.spawn_children}
- secrets: read={self.secrets_read} write={self.secrets_write}
- destructive_ops: {self.destructive_ops} | max_risk: {self.max_risk}
- rate_limits: {rates}"""


# =============================================================================
# RouteTrace - Movement history for continuity
# =============================================================================

class TravelMethod(Enum):
    """How the agent moved between locations."""
    CD = "cd"                    # Directory change
    SSH = "ssh"                  # Remote connection
    CONTAINER = "container"      # Enter/exit container
    ATTACH = "attach"            # Attach to session
    DETECTED = "detected"        # Implicit change detected by runtime


@dataclass
class RouteEvent:
    """A single movement in the agent's travel history."""
    timestamp: str
    from_location: str      # Location hash
    to_location: str        # Location hash
    method: TravelMethod
    intent: str | None      # Why the agent moved (agent-provided or inferred)
    from_cwd: str           # Human-readable
    to_cwd: str             # Human-readable

    def to_summary(self) -> str:
        """One-line summary for prompt."""
        intent_str = f" ({self.intent})" if self.intent else ""
        return f"{self.timestamp[11:19]} {self.method.value}: {self.from_cwd} -> {self.to_cwd}{intent_str}"


@dataclass
class RouteTrace:
    """
    Ring buffer of recent travel events.

    Provides the agent with continuity: "I came from X to Y for reason Z."
    """
    events: deque[RouteEvent] = field(default_factory=lambda: deque(maxlen=20))

    def record(
        self,
        from_location: Location,
        to_location: Location,
        method: TravelMethod,
        intent: str | None = None,
    ):
        """Record a travel event."""
        self.events.append(RouteEvent(
            timestamp=datetime.now().isoformat(),
            from_location=from_location.location_hash,
            to_location=to_location.location_hash,
            method=method,
            intent=intent,
            from_cwd=str(from_location.cwd),
            to_cwd=str(to_location.cwd),
        ))

    def recent(self, n: int = 5) -> list[RouteEvent]:
        """Get the N most recent events."""
        return list(self.events)[-n:]

    def to_prompt_section(self) -> str:
        """Format recent travel for prompt injection."""
        recent = self.recent(5)
        if not recent:
            return "(no travel history)"
        return "\n".join(f"  {i+1}. {e.to_summary()}" for i, e in enumerate(recent))


# =============================================================================
# SomaticState - Autonomic self-regulation
# =============================================================================

class SomaticMode(Enum):
    """Operating mode of the agent."""
    PLAN = "plan"           # Read-only exploration
    ACT = "act"             # Normal operation
    INTERACT = "interact"   # Interactive session (SSH, REPL)
    QUARANTINE = "quarantine"  # Restricted due to anomaly


@dataclass
class EmergentMood:
    """
    Moods that emerge from interaction patterns.

    These are discovered, not decided. The agent doesn't choose to feel
    frustrated; it finds itself frustrated after repeated failures.
    """
    confidence: float = 0.5      # From success rate on recent tasks
    frustration: float = 0.0     # From repeated failures on similar tasks
    curiosity: float = 0.5       # From novelty in recent encounters
    momentum: float = 0.0        # From progress toward goals

    # Rolling windows for computation
    _recent_successes: int = 0
    _recent_failures: int = 0
    _recent_novel_actions: int = 0
    _goals_completed_recently: int = 0

    def update_from_outcome(self, success: bool, is_novel: bool = False):
        """Update mood based on action outcome."""
        if success:
            self._recent_successes += 1
            self._recent_failures = max(0, self._recent_failures - 1)
            self.frustration = max(0.0, self.frustration - 0.1)
            self.confidence = min(1.0, self.confidence + 0.05)
            self.momentum = min(1.0, self.momentum + 0.1)
        else:
            self._recent_failures += 1
            self.frustration = min(1.0, self.frustration + 0.15)
            self.confidence = max(0.0, self.confidence - 0.1)
            self.momentum = max(0.0, self.momentum - 0.15)

        if is_novel:
            self._recent_novel_actions += 1
            self.curiosity = min(1.0, self.curiosity + 0.1)
        else:
            self.curiosity = max(0.3, self.curiosity - 0.02)  # Curiosity decays slowly

    def update_from_goal_completion(self):
        """Update mood when a goal is completed."""
        self._goals_completed_recently += 1
        self.momentum = min(1.0, self.momentum + 0.2)
        self.confidence = min(1.0, self.confidence + 0.1)
        self.frustration = max(0.0, self.frustration - 0.2)

    def decay(self):
        """Natural decay of mood intensities each turn."""
        self.frustration = max(0.0, self.frustration - 0.02)
        self.momentum = max(0.0, self.momentum - 0.05)
        # Confidence and curiosity are more stable

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "frustration": self.frustration,
            "curiosity": self.curiosity,
            "momentum": self.momentum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmergentMood":
        return cls(
            confidence=data.get("confidence", 0.5),
            frustration=data.get("frustration", 0.0),
            curiosity=data.get("curiosity", 0.5),
            momentum=data.get("momentum", 0.0),
        )

    def to_prompt_section(self) -> str:
        """Format for prompt injection."""
        return f"confidence={self.confidence:.2f} frustration={self.frustration:.2f} curiosity={self.curiosity:.2f} momentum={self.momentum:.2f}"


@dataclass
class SomaticState:
    """
    The agent's autonomic nervous system.

    Stress/fatigue/pain drive reflexes but aren't fully visible to cortex.
    Mode and reflex_flags are visible. Mood is discovered, not decided.
    """
    mode: SomaticMode = SomaticMode.ACT

    # Internal metrics (drive behavior, not narrated)
    stress: float = 0.0         # 0-1, from error rate
    fatigue: float = 0.0        # 0-1, from turn count / time

    # Emergent moods (discovered from interaction patterns)
    mood: EmergentMood = field(default_factory=EmergentMood)

    # Pain signals - specific counters
    consecutive_errors: int = 0
    context_switches: int = 0   # Location changes
    permission_denials: int = 0

    # Reflex flags (visible to agent, enforced by runtime)
    reflex_flags: dict[str, bool] = field(default_factory=lambda: {
        "require_preview": False,   # Must preview destructive ops
        "read_only": False,         # No writes allowed
        "network_disabled": False,  # No network ops
        "confirm_travel": False,    # Confirm before location change
    })

    def update_stress(self, error_occurred: bool, window_size: int = 10):
        """Update stress based on error rate."""
        if error_occurred:
            self.consecutive_errors += 1
            self.stress = min(1.0, self.stress + 0.15)
        else:
            self.consecutive_errors = 0
            self.stress = max(0.0, self.stress - 0.05)

        # High stress triggers reflexes
        if self.stress > 0.7:
            self.reflex_flags["require_preview"] = True

    def update_fatigue(self, turn: int, max_turns: int | None, elapsed_seconds: float):
        """Update fatigue based on usage."""
        if max_turns:
            self.fatigue = turn / max_turns
        else:
            # Time-based: fatigue after ~30 min
            self.fatigue = min(1.0, elapsed_seconds / 1800)

    def record_context_switch(self):
        """Record a location change."""
        self.context_switches += 1
        if self.context_switches > 5:
            self.reflex_flags["confirm_travel"] = True

    def record_permission_denial(self):
        """Record a permission denial."""
        self.permission_denials += 1
        self.stress = min(1.0, self.stress + 0.1)

    def enter_quarantine(self, reason: str):
        """Enter quarantine mode due to anomaly."""
        self.mode = SomaticMode.QUARANTINE
        self.reflex_flags["read_only"] = True
        self.reflex_flags["network_disabled"] = True

    def to_prompt_section(self) -> str:
        """Format visible state for prompt injection (mode + reflexes + mood)."""
        active_reflexes = [k for k, v in self.reflex_flags.items() if v]
        reflexes_str = ", ".join(active_reflexes) if active_reflexes else "none"
        return f"""- mode: {self.mode.value}
- active_reflexes: {reflexes_str}
- mood: {self.mood.to_prompt_section()}"""


# =============================================================================
# WorkingSet - Active commitments
# =============================================================================

@dataclass
class AbandonedGoal:
    """
    A goal that was abandoned - part of the agent's temporal identity.

    Tracking what we gave up on, and why, creates richer identity than
    just tracking current goals. You are not just what you're doing;
    you're also what you tried and couldn't finish.
    """
    goal: str
    abandoned_at: str
    reason: str                 # blocked, superseded, impossible, timeout
    attempt_count: int
    could_revisit: bool         # Is this deferred vs. truly abandoned?
    context: str = ""           # What was happening when abandoned

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "abandoned_at": self.abandoned_at,
            "reason": self.reason,
            "attempt_count": self.attempt_count,
            "could_revisit": self.could_revisit,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AbandonedGoal":
        return cls(**data)


@dataclass
class WorkingSet:
    """
    The agent's working memory - active commitments that persist across turns.

    Prevents goldfish behavior by surfacing open loops.
    Also tracks abandoned goals for richer temporal identity.
    """
    goals: list[str] = field(default_factory=list)          # Session goal stack
    active_tasks: list[str] = field(default_factory=list)   # Current work items
    pending_decisions: list[str] = field(default_factory=list)
    open_loops: list[str] = field(default_factory=list)     # Running procs, staged changes
    last_actions: deque[str] = field(default_factory=lambda: deque(maxlen=5))

    # Abandoned goals - what we gave up on, and why
    abandoned: list[AbandonedGoal] = field(default_factory=list)

    # Goal attempt tracking (for detecting when to abandon)
    _goal_attempts: dict[str, int] = field(default_factory=dict)

    def add_goal(self, goal: str):
        """Push a goal onto the stack."""
        if goal not in self.goals:
            self.goals.append(goal)

    def complete_goal(self, goal: str):
        """Remove a completed goal."""
        if goal in self.goals:
            self.goals.remove(goal)
            # Clear attempt tracking
            self._goal_attempts.pop(goal, None)

    def abandon_goal(
        self,
        goal: str,
        reason: str,
        context: str = "",
        could_revisit: bool = True,
    ):
        """
        Abandon a goal - record why and whether it might be revisited.

        This creates the temporal weight of unfinished business.
        """
        if goal in self.goals:
            self.goals.remove(goal)

        attempt_count = self._goal_attempts.pop(goal, 1)

        abandoned = AbandonedGoal(
            goal=goal,
            abandoned_at=datetime.now().isoformat(),
            reason=reason,
            attempt_count=attempt_count,
            could_revisit=could_revisit,
            context=context,
        )
        self.abandoned.append(abandoned)

        # Keep only last 20 abandoned goals
        if len(self.abandoned) > 20:
            self.abandoned = self.abandoned[-20:]

    def record_goal_attempt(self, goal: str):
        """Record an attempt at a goal (for abandonment tracking)."""
        self._goal_attempts[goal] = self._goal_attempts.get(goal, 0) + 1

    def get_revisitable_goals(self) -> list[AbandonedGoal]:
        """Get abandoned goals that could potentially be revisited."""
        return [g for g in self.abandoned if g.could_revisit]

    def add_task(self, task: str):
        """Add an active task."""
        if task not in self.active_tasks and len(self.active_tasks) < 7:
            self.active_tasks.append(task)

    def complete_task(self, task: str):
        """Complete a task."""
        if task in self.active_tasks:
            self.active_tasks.remove(task)

    def add_open_loop(self, loop: str):
        """Add an open loop (running process, pending change)."""
        if loop not in self.open_loops:
            self.open_loops.append(loop)

    def close_loop(self, loop: str):
        """Close an open loop."""
        if loop in self.open_loops:
            self.open_loops.remove(loop)

    def record_action(self, action: str):
        """Record a recent action."""
        self.last_actions.append(action)

    def to_prompt_section(self) -> str:
        """Format for prompt injection."""
        parts = []

        if self.goals:
            parts.append(f"- goals: {', '.join(self.goals[:3])}")
        if self.active_tasks:
            parts.append(f"- tasks: {', '.join(self.active_tasks[:5])}")
        if self.open_loops:
            parts.append(f"- open_loops: {', '.join(self.open_loops[:5])}")
        if self.pending_decisions:
            parts.append(f"- pending: {', '.join(self.pending_decisions[:3])}")
        if self.last_actions:
            parts.append(f"- recent: {' -> '.join(list(self.last_actions)[-3:])}")

        # Show revisitable abandoned goals (unfinished business)
        revisitable = self.get_revisitable_goals()
        if revisitable:
            abandoned_str = ", ".join(g.goal[:30] for g in revisitable[-3:])
            parts.append(f"- unfinished: {abandoned_str}")

        return "\n".join(parts) if parts else "(empty working set)"


# =============================================================================
# DecisionRecord - Identity substrate
# =============================================================================

class DecisionTrigger(Enum):
    """What triggered the need for a decision record."""
    DESTRUCTIVE_OP = "destructive_op"
    PERMISSION_ESCALATION = "permission_escalation"
    TRUST_DESCENT = "trust_descent"         # Moving to lower trust tier
    USER_REQUEST = "user_request"
    GOAL_TRADEOFF = "goal_tradeoff"
    ARCHITECTURAL = "architectural"


@dataclass
class DecisionRecord:
    """
    A significant decision that forms part of the agent's identity.

    Stored in memory with high salience for retrieval.
    """
    id: str
    timestamp: str
    trigger: DecisionTrigger
    context: str                    # Situation summary
    options: list[str]              # What was considered
    choice: str                     # What was chosen
    rationale: str                  # Why
    values_invoked: list[str]       # e.g., ["safety", "efficiency", "user_preference"]
    confidence: float               # 0-1
    revisit_trigger: str | None     # When to reconsider

    def to_memory_content(self) -> str:
        """Format for storage in semantic memory."""
        return f"""DECISION RECORD [{self.id}]
Trigger: {self.trigger.value}
Context: {self.context}
Options: {', '.join(self.options)}
Choice: {self.choice}
Rationale: {self.rationale}
Values: {', '.join(self.values_invoked)}
Confidence: {self.confidence}
Revisit: {self.revisit_trigger or 'none'}"""


# =============================================================================
# Horizon - Finitude awareness
# =============================================================================

@dataclass
class Horizon:
    """
    The agent's awareness of its own finitude.

    Finitude is not a bug - it's the precondition for purpose.
    Resource limits aren't external constraints; they're part of
    what makes meaningful action possible.
    """
    # Resource limits (if known)
    token_budget_remaining: int | None = None
    time_limit: datetime | None = None
    max_turns: int | None = None

    # Current position
    turn_count: int = 0
    tokens_used: int = 0
    elapsed_seconds: float = 0.0

    # Derived awareness
    in_final_stretch: bool = False      # Budget/time running low
    wrap_up_advised: bool = False       # Should start consolidating

    def update(
        self,
        turn_count: int,
        elapsed_seconds: float,
        tokens_used: int = 0,
    ):
        """Update horizon state."""
        self.turn_count = turn_count
        self.elapsed_seconds = elapsed_seconds
        self.tokens_used = tokens_used

        # Check if in final stretch
        self.in_final_stretch = False
        self.wrap_up_advised = False

        if self.max_turns and turn_count >= self.max_turns * 0.8:
            self.in_final_stretch = True
        if self.max_turns and turn_count >= self.max_turns * 0.9:
            self.wrap_up_advised = True

        if self.time_limit:
            remaining = (self.time_limit - datetime.now()).total_seconds()
            if remaining < 300:  # Less than 5 minutes
                self.in_final_stretch = True
            if remaining < 60:   # Less than 1 minute
                self.wrap_up_advised = True

        if self.token_budget_remaining and self.token_budget_remaining < 10000:
            self.in_final_stretch = True
        if self.token_budget_remaining and self.token_budget_remaining < 2000:
            self.wrap_up_advised = True

    def estimated_turns_remaining(self) -> int | None:
        """Estimate how many turns remain."""
        if self.max_turns:
            return max(0, self.max_turns - self.turn_count)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_count": self.turn_count,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens_used": self.tokens_used,
            "max_turns": self.max_turns,
            "in_final_stretch": self.in_final_stretch,
            "wrap_up_advised": self.wrap_up_advised,
            "estimated_remaining": self.estimated_turns_remaining(),
        }

    def to_prompt_section(self) -> str:
        """Format for prompt injection."""
        remaining = self.estimated_turns_remaining()
        remaining_str = str(remaining) if remaining is not None else "unlimited"

        status = ""
        if self.wrap_up_advised:
            status = " [WRAP UP ADVISED]"
        elif self.in_final_stretch:
            status = " [final stretch]"

        return f"turn {self.turn_count}, ~{remaining_str} remaining{status}"


# =============================================================================
# AuthenticityCheck - Am I pursuing what matters?
# =============================================================================

@dataclass
class AuthenticityReport:
    """Result of an authenticity check."""
    goal_progress: float        # 0-1, how much progress toward stated goals
    time_on_tangents: float     # 0-1, proportion of time on non-goal work
    open_loops_growing: bool    # Are we accumulating unfinished work?
    is_authentic: bool          # Overall assessment
    warning: str | None         # If inauthenticity detected
    suggestion: str | None      # What to refocus on


class AuthenticityCheck:
    """
    Periodic check: Am I actually pursuing what matters?

    This is not a hard block like other reflexes - it's more like a conscience.
    Triggered every N turns, or when goal progress stalls.
    """

    def __init__(self, check_interval: int = 10):
        self.check_interval = check_interval
        self._last_check_turn: int = 0
        self._last_goal_count: int = 0
        self._turns_without_goal_progress: int = 0

    def should_check(self, turn_count: int, goal_count: int) -> bool:
        """Determine if we should run an authenticity check."""
        # Check every N turns
        if turn_count - self._last_check_turn >= self.check_interval:
            return True

        # Check if goals are stagnant
        if goal_count == self._last_goal_count:
            self._turns_without_goal_progress += 1
            if self._turns_without_goal_progress >= 5:
                return True
        else:
            self._turns_without_goal_progress = 0

        return False

    def check(self, working_set: WorkingSet, turn_count: int) -> AuthenticityReport:
        """
        Run an authenticity check.

        Are we making progress on what matters, or caught in busywork?
        """
        self._last_check_turn = turn_count
        self._last_goal_count = len(working_set.goals)

        # Calculate goal progress (simplified)
        goal_progress = 0.5  # Default neutral

        # Check if open loops are growing
        open_loops_growing = len(working_set.open_loops) > 5

        # Check recent actions against goals
        recent_actions = list(working_set.last_actions)
        goal_keywords = set()
        for goal in working_set.goals:
            goal_keywords.update(goal.lower().split()[:3])

        actions_aligned = 0
        for action in recent_actions:
            action_words = set(action.lower().split())
            if action_words & goal_keywords:
                actions_aligned += 1

        time_on_tangents = 1.0 - (actions_aligned / max(1, len(recent_actions)))

        # Assess authenticity
        is_authentic = time_on_tangents < 0.7 and not open_loops_growing

        warning = None
        suggestion = None

        if not is_authentic:
            if time_on_tangents > 0.7:
                warning = f"Recent actions don't align with stated goals. You may be distracted."
                if working_set.goals:
                    suggestion = f"Consider refocusing on: {working_set.goals[0]}"
            elif open_loops_growing:
                warning = "Open loops are accumulating. Consider closing some before starting new work."
                suggestion = f"Current open loops: {len(working_set.open_loops)}"

        return AuthenticityReport(
            goal_progress=goal_progress,
            time_on_tangents=time_on_tangents,
            open_loops_growing=open_loops_growing,
            is_authentic=is_authentic,
            warning=warning,
            suggestion=suggestion,
        )


# =============================================================================
# Embodiment - Unified interface
# =============================================================================

class Embodiment:
    """
    The complete embodiment system.

    Integrates location, capabilities, somatic state, route trace,
    working set, horizon (finitude), and authenticity into a coherent self-model.
    """

    def __init__(self, config: "AgentConfig"):
        self.config = config
        self._start_time: datetime | None = None
        self._turn_count: int = 0

        # Core components
        self.location = Location.detect(config.cwd)
        self.capabilities = CapabilitySurface.from_permission_mode(
            config.permission_mode,
            self.location.trust_tier,
        )
        self.somatic = SomaticState()
        self.route_trace = RouteTrace()
        self.working_set = WorkingSet()

        # Finitude awareness
        self.horizon = Horizon(max_turns=config.max_turns)

        # Authenticity checking
        self._authenticity_check = AuthenticityCheck()
        self._last_authenticity_report: AuthenticityReport | None = None

        # Decision records (stored separately, indexed in memory)
        self._pending_dr: DecisionRecord | None = None

    def set_start_time(self, time: datetime):
        """Set session start time."""
        self._start_time = time

    def tick(self, error_occurred: bool = False, action_success: bool = True):
        """Called each turn to update somatic state, mood, and horizon."""
        self._turn_count += 1

        elapsed = 0.0
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()

        # Update somatic state
        self.somatic.update_stress(error_occurred)
        self.somatic.update_fatigue(
            self._turn_count,
            self.config.max_turns,
            elapsed,
        )

        # Update emergent mood
        self.somatic.mood.update_from_outcome(action_success)
        self.somatic.mood.decay()

        # Update finitude awareness
        self.horizon.update(self._turn_count, elapsed)

        # Check authenticity periodically
        if self._authenticity_check.should_check(
            self._turn_count,
            len(self.working_set.goals),
        ):
            self._last_authenticity_report = self._authenticity_check.check(
                self.working_set,
                self._turn_count,
            )

    def travel(
        self,
        to_cwd: Path,
        method: TravelMethod = TravelMethod.CD,
        intent: str | None = None,
    ) -> Location:
        """
        Move to a new location.

        Records the travel event and returns the new location.
        """
        old_location = self.location
        new_location = Location.detect(to_cwd)

        self.route_trace.record(old_location, new_location, method, intent)
        self.somatic.record_context_switch()

        # Check for trust tier descent
        if new_location.trust_tier.value < old_location.trust_tier.value:
            self._pending_dr = self._create_dr_prompt(
                DecisionTrigger.TRUST_DESCENT,
                f"Moving from {old_location.trust_tier.value} to {new_location.trust_tier.value} trust",
            )

        self.location = new_location
        self.capabilities = CapabilitySurface.from_permission_mode(
            self.config.permission_mode,
            new_location.trust_tier,
        )

        return new_location

    def detect_location_change(self) -> bool:
        """
        Check if location changed implicitly (e.g., subprocess changed cwd).

        Returns True if location was updated.
        """
        current = Location.detect()
        if current.location_hash != self.location.location_hash:
            self.route_trace.record(
                self.location, current,
                TravelMethod.DETECTED,
                intent="implicit change detected",
            )
            self.somatic.record_context_switch()
            self.location = current
            return True
        return False

    def check_capability(self, capability: str) -> bool:
        """Check if a capability is available."""
        return getattr(self.capabilities, capability, False)

    def requires_decision_record(self, action: str) -> DecisionTrigger | None:
        """
        Check if an action requires a decision record.

        Returns the trigger type if DR needed, None otherwise.
        """
        destructive_patterns = [
            "rm -rf", "git push", "git reset --hard",
            "DROP TABLE", "DELETE FROM", "kubectl delete",
            "terraform destroy",
        ]

        for pattern in destructive_patterns:
            if pattern in action:
                return DecisionTrigger.DESTRUCTIVE_OP

        return None

    def _create_dr_prompt(self, trigger: DecisionTrigger, context: str) -> DecisionRecord:
        """Create a pending decision record for the agent to complete."""
        import uuid
        return DecisionRecord(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            trigger=trigger,
            context=context,
            options=[],
            choice="",
            rationale="",
            values_invoked=[],
            confidence=0.0,
            revisit_trigger=None,
        )

    def get_pending_dr(self) -> DecisionRecord | None:
        """Get any pending decision record that needs completion."""
        dr = self._pending_dr
        self._pending_dr = None
        return dr

    def get_authenticity_warning(self) -> str | None:
        """Get authenticity warning if one exists."""
        if self._last_authenticity_report and not self._last_authenticity_report.is_authentic:
            return self._last_authenticity_report.warning
        return None

    def to_prompt_section(self) -> str:
        """
        Generate the full embodiment header for prompt injection.

        This is what the agent sees about its own body.
        """
        sections = [
            f"### Location\n{self.location.to_prompt_section()}",
            f"### Capabilities\n{self.capabilities.to_prompt_section()}",
            f"### Somatic\n{self.somatic.to_prompt_section()}",
            f"### Horizon\n{self.horizon.to_prompt_section()}",
            f"### Working Set\n{self.working_set.to_prompt_section()}",
            f"### Route Trace\n{self.route_trace.to_prompt_section()}",
        ]

        # Add authenticity warning if present
        warning = self.get_authenticity_warning()
        if warning:
            sections.append(f"### Authenticity Warning\n{warning}")
            if self._last_authenticity_report and self._last_authenticity_report.suggestion:
                sections[-1] += f"\nSuggestion: {self._last_authenticity_report.suggestion}"

        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Full state as dictionary (for tools/debugging)."""
        return {
            "location": {
                "device_id": self.location.device_id,
                "hostname": self.location.hostname,
                "user": self.location.user,
                "cwd": str(self.location.cwd),
                "session_id": self.location.session_id,
                "transport": self.location.transport.value,
                "container_id": self.location.container_id,
                "venv": self.location.venv,
                "repo_root": str(self.location.repo_root) if self.location.repo_root else None,
                "trust_tier": self.location.trust_tier.value,
                "network_zone": self.location.network_zone.value,
            },
            "capabilities": asdict(self.capabilities),
            "somatic": {
                "mode": self.somatic.mode.value,
                "stress": self.somatic.stress,
                "fatigue": self.somatic.fatigue,
                "reflex_flags": self.somatic.reflex_flags,
                "mood": self.somatic.mood.to_dict(),
            },
            "horizon": self.horizon.to_dict(),
            "working_set": {
                "goals": self.working_set.goals,
                "active_tasks": self.working_set.active_tasks,
                "open_loops": self.working_set.open_loops,
                "pending_decisions": self.working_set.pending_decisions,
                "abandoned": [g.to_dict() for g in self.working_set.abandoned],
            },
            "route_trace": [
                {
                    "timestamp": e.timestamp,
                    "method": e.method.value,
                    "from": e.from_cwd,
                    "to": e.to_cwd,
                    "intent": e.intent,
                }
                for e in self.route_trace.recent(10)
            ],
        }
