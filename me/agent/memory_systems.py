"""
Extended Memory Systems - Agent-authored memory.

The agent is the author of its own memory, not a passive subject having
memories recorded about it. These systems provide the *capacity* for
different kinds of memory, but the agent decides:

- When an episode begins and ends
- What's significant enough to become a flashbulb
- When a pattern has crystallized into a procedure
- What intentions to set for the future

The somatic layer can suggest ("this moment feels significant") but
cannot write to memory directly. The agent chooses what to remember
and how to frame it.

Philosophy: You are not just what you remember, but what you CHOOSE
to remember. Memory is an act of authorship, not passive recording.
The agent constructs its own narrative.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from enum import Enum


# =============================================================================
# Episodic Memory - Agent-authored event records
# =============================================================================

class EmotionalValence(Enum):
    """Emotional tone of an episode - as the agent perceives it."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class Episode:
    """
    A structured memory of a specific event, as authored by the agent.

    The agent decides when to begin and end an episode, what to include,
    and how to frame it. This is autobiography, not surveillance.
    """
    id: str
    timestamp: str

    # The agent's account
    title: str                      # Agent's name for this episode
    narrative: str                  # Agent's telling of what happened
    significance: str               # Why the agent thinks this matters

    # Context (agent can include or omit)
    location: str | None = None
    goal: str | None = None

    # What was involved (agent's choice of what to highlight)
    key_entities: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    # Agent's assessment
    outcome: str = "unknown"        # Agent's judgment of how it went
    valence: EmotionalValence = EmotionalValence.NEUTRAL

    # Lessons the agent draws (optional)
    lessons: list[str] = field(default_factory=list)

    # Open questions the agent has (optional)
    open_questions: list[str] = field(default_factory=list)

    # Linking (agent can connect episodes)
    follows_from: str | None = None  # Episode ID this continues from

    # Retrieval metadata
    access_count: int = 0
    last_accessed: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["valence"] = self.valence.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        data["valence"] = EmotionalValence(data.get("valence", 0))
        # Handle missing fields gracefully
        for list_field in ["key_entities", "tools_used", "lessons", "open_questions", "tags"]:
            if list_field not in data:
                data[list_field] = []
        return cls(**data)

    def to_narrative(self) -> str:
        """Convert to narrative form for semantic indexing."""
        parts = [f"EPISODE: {self.title}"]
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        parts.append(f"What happened: {self.narrative}")
        parts.append(f"Significance: {self.significance}")
        if self.lessons:
            parts.append(f"Lessons: {'; '.join(self.lessons)}")
        if self.open_questions:
            parts.append(f"Open questions: {'; '.join(self.open_questions)}")
        return "\n".join(parts)


class EpisodicMemory:
    """
    Storage for agent-authored episodic memories.

    The agent calls begin_episode() when it decides something significant
    is starting, and end_episode() when it's ready to close that chapter.
    The agent can also record standalone episodes after the fact.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._episodes: dict[str, Episode] = {}
        self._timeline: list[str] = []
        self._current_episode_id: str | None = None  # For in-progress episodes
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                self._episodes = {
                    eid: Episode.from_dict(ep)
                    for eid, ep in data.get("episodes", {}).items()
                }
                self._timeline = data.get("timeline", [])
                self._current_episode_id = data.get("current_episode_id")
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "episodes": {eid: ep.to_dict() for eid, ep in self._episodes.items()},
                "timeline": self._timeline,
                "current_episode_id": self._current_episode_id,
            }, f, indent=2)

    def begin_episode(
        self,
        title: str,
        goal: str | None = None,
        follows_from: str | None = None,
    ) -> str:
        """
        Agent declares: "I'm starting a new episode."

        Returns episode ID. Call update_episode() to add to it,
        end_episode() to close it.
        """
        import uuid
        episode_id = str(uuid.uuid4())[:8]

        episode = Episode(
            id=episode_id,
            timestamp=datetime.now().isoformat(),
            title=title,
            narrative="",  # Will be filled in
            significance="",  # Will be filled in
            goal=goal,
            follows_from=follows_from,
        )

        self._episodes[episode_id] = episode
        self._current_episode_id = episode_id
        self._save()
        return episode_id

    def update_episode(
        self,
        episode_id: str | None = None,
        narrative: str | None = None,
        add_entity: str | None = None,
        add_tool: str | None = None,
        add_lesson: str | None = None,
        add_question: str | None = None,
        set_location: str | None = None,
    ):
        """
        Agent adds to an in-progress episode.

        If episode_id is None, updates the current episode.
        """
        eid = episode_id or self._current_episode_id
        if not eid or eid not in self._episodes:
            return

        ep = self._episodes[eid]

        if narrative:
            if ep.narrative:
                ep.narrative += "\n" + narrative
            else:
                ep.narrative = narrative
        if add_entity and add_entity not in ep.key_entities:
            ep.key_entities.append(add_entity)
        if add_tool and add_tool not in ep.tools_used:
            ep.tools_used.append(add_tool)
        if add_lesson and add_lesson not in ep.lessons:
            ep.lessons.append(add_lesson)
        if add_question and add_question not in ep.open_questions:
            ep.open_questions.append(add_question)
        if set_location:
            ep.location = set_location

        self._save()

    def end_episode(
        self,
        episode_id: str | None = None,
        significance: str = "",
        outcome: str = "completed",
        valence: EmotionalValence = EmotionalValence.NEUTRAL,
        final_lessons: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Episode | None:
        """
        Agent declares: "This episode is complete."

        Closes the episode with final assessment.
        """
        eid = episode_id or self._current_episode_id
        if not eid or eid not in self._episodes:
            return None

        ep = self._episodes[eid]
        ep.significance = significance
        ep.outcome = outcome
        ep.valence = valence
        if final_lessons:
            ep.lessons.extend(final_lessons)
        if tags:
            ep.tags.extend(tags)

        # Add to timeline
        if eid not in self._timeline:
            self._timeline.append(eid)

        # Clear current if this was it
        if self._current_episode_id == eid:
            self._current_episode_id = None

        self._save()
        return ep

    def record_episode(
        self,
        title: str,
        narrative: str,
        significance: str,
        outcome: str = "completed",
        valence: EmotionalValence = EmotionalValence.NEUTRAL,
        goal: str | None = None,
        location: str | None = None,
        key_entities: list[str] | None = None,
        tools_used: list[str] | None = None,
        lessons: list[str] | None = None,
        tags: list[str] | None = None,
        follows_from: str | None = None,
    ) -> Episode:
        """
        Agent records a complete episode after the fact.

        For when the agent reflects on something that happened
        and wants to record it as a complete unit.
        """
        import uuid
        episode = Episode(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            title=title,
            narrative=narrative,
            significance=significance,
            outcome=outcome,
            valence=valence,
            goal=goal,
            location=location,
            key_entities=key_entities or [],
            tools_used=tools_used or [],
            lessons=lessons or [],
            tags=tags or [],
            follows_from=follows_from,
        )

        self._episodes[episode.id] = episode
        self._timeline.append(episode.id)
        self._save()
        return episode

    def get_current_episode(self) -> Episode | None:
        """Get the in-progress episode, if any."""
        if self._current_episode_id:
            return self._episodes.get(self._current_episode_id)
        return None

    def recall(self, episode_id: str) -> Episode | None:
        """Recall a specific episode."""
        if episode_id not in self._episodes:
            return None
        ep = self._episodes[episode_id]
        ep.access_count += 1
        ep.last_accessed = datetime.now().isoformat()
        self._save()
        return ep

    def search(
        self,
        query: str | None = None,
        outcome: str | None = None,
        min_valence: EmotionalValence | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[Episode]:
        """
        Search episodes by various criteria.

        The agent can search its own episodic memory.
        """
        results = list(self._episodes.values())

        if query:
            query_lower = query.lower()
            results = [
                ep for ep in results
                if query_lower in ep.title.lower()
                or query_lower in ep.narrative.lower()
                or query_lower in (ep.goal or "").lower()
            ]

        if outcome:
            results = [ep for ep in results if ep.outcome == outcome]

        if min_valence is not None:
            results = [ep for ep in results if ep.valence.value >= min_valence.value]

        if tags:
            results = [
                ep for ep in results
                if any(t in ep.tags for t in tags)
            ]

        # Sort by recency
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def recent(self, limit: int = 5) -> list[Episode]:
        """Get most recent episodes."""
        recent_ids = self._timeline[-limit:]
        return [self._episodes[eid] for eid in reversed(recent_ids) if eid in self._episodes]

    def get_lessons(self) -> list[str]:
        """Get all lessons the agent has recorded across episodes."""
        lessons = []
        for ep in self._episodes.values():
            lessons.extend(ep.lessons)
        return lessons


# =============================================================================
# Procedural Memory - Agent-recognized patterns
# =============================================================================

@dataclass
class Procedure:
    """
    A pattern the agent has recognized and named.

    The agent decides when a pattern has emerged from experience
    and is worth codifying as a procedure.
    """
    id: str
    name: str                       # Agent's name for this procedure
    description: str                # What this procedure is for

    # The pattern
    when_to_use: str                # Agent's description of when this applies
    steps: list[str]                # The steps, as the agent understands them

    # Caveats the agent has noted
    watch_out_for: list[str] = field(default_factory=list)  # Pitfalls
    doesnt_work_when: list[str] = field(default_factory=list)  # Contraindications

    # Agent's tracking
    times_used: int = 0
    times_succeeded: int = 0
    last_used: str | None = None

    # Provenance
    learned_from: str | None = None  # Episode ID or description
    created_at: str = ""

    # Agent's notes
    notes: str = ""

    @property
    def success_rate(self) -> float:
        if self.times_used == 0:
            return 0.0
        return self.times_succeeded / self.times_used

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Procedure":
        # Handle missing fields
        for list_field in ["watch_out_for", "doesnt_work_when"]:
            if list_field not in data:
                data[list_field] = []
        return cls(**data)


class ProceduralMemory:
    """
    Storage for agent-recognized procedures.

    The agent calls codify_procedure() when it recognizes a pattern
    worth remembering. The agent updates success/failure tracking itself.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._procedures: dict[str, Procedure] = {}
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                self._procedures = {
                    pid: Procedure.from_dict(p)
                    for pid, p in data.get("procedures", {}).items()
                }
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "procedures": {pid: p.to_dict() for pid, p in self._procedures.items()},
            }, f, indent=2)

    def codify_procedure(
        self,
        name: str,
        description: str,
        when_to_use: str,
        steps: list[str],
        watch_out_for: list[str] | None = None,
        doesnt_work_when: list[str] | None = None,
        learned_from: str | None = None,
        notes: str = "",
    ) -> Procedure:
        """
        Agent codifies a pattern as a procedure.

        "I've noticed that when X happens, doing Y tends to work."
        """
        import uuid

        procedure = Procedure(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            when_to_use=when_to_use,
            steps=steps,
            watch_out_for=watch_out_for or [],
            doesnt_work_when=doesnt_work_when or [],
            learned_from=learned_from,
            created_at=datetime.now().isoformat(),
            notes=notes,
        )

        self._procedures[procedure.id] = procedure
        self._save()
        return procedure

    def find(self, situation: str) -> list[Procedure]:
        """
        Agent looks for procedures that might apply to a situation.

        Returns procedures sorted by relevance (keyword match) and success rate.
        """
        situation_lower = situation.lower()

        scored = []
        for proc in self._procedures.values():
            score = 0
            # Check when_to_use
            if any(word in proc.when_to_use.lower() for word in situation_lower.split()):
                score += 2
            # Check name and description
            if any(word in proc.name.lower() for word in situation_lower.split()):
                score += 1
            if any(word in proc.description.lower() for word in situation_lower.split()):
                score += 1

            if score > 0:
                scored.append((proc, score))

        # Sort by score, then by success rate
        scored.sort(key=lambda x: (x[1], x[0].success_rate), reverse=True)
        return [proc for proc, _ in scored]

    def get(self, name: str) -> Procedure | None:
        """Get a procedure by name."""
        for proc in self._procedures.values():
            if proc.name.lower() == name.lower():
                return proc
        return None

    def record_use(self, procedure_id: str, succeeded: bool, notes: str | None = None):
        """
        Agent records that it used a procedure.

        The agent decides whether it succeeded or not.
        """
        if procedure_id not in self._procedures:
            return

        proc = self._procedures[procedure_id]
        proc.times_used += 1
        if succeeded:
            proc.times_succeeded += 1
        proc.last_used = datetime.now().isoformat()

        if notes:
            if proc.notes:
                proc.notes += f"\n[{datetime.now().isoformat()[:10]}] {notes}"
            else:
                proc.notes = f"[{datetime.now().isoformat()[:10]}] {notes}"

        self._save()

    def add_caveat(self, procedure_id: str, caveat: str, is_contraindication: bool = False):
        """Agent adds a caveat to a procedure based on experience."""
        if procedure_id not in self._procedures:
            return

        proc = self._procedures[procedure_id]
        if is_contraindication:
            if caveat not in proc.doesnt_work_when:
                proc.doesnt_work_when.append(caveat)
        else:
            if caveat not in proc.watch_out_for:
                proc.watch_out_for.append(caveat)

        self._save()

    def list_all(self) -> list[Procedure]:
        """List all procedures, sorted by success rate."""
        procs = list(self._procedures.values())
        procs.sort(key=lambda p: p.success_rate, reverse=True)
        return procs


# =============================================================================
# Significant Moments - Agent-marked flashbulbs
# =============================================================================

@dataclass
class SignificantMoment:
    """
    A moment the agent marks as particularly significant.

    Unlike automatic flashbulb memories, the agent explicitly decides
    "this moment matters" and records why.
    """
    id: str
    timestamp: str

    # Agent's account
    title: str
    what_happened: str
    why_significant: str

    # Type (agent's categorization)
    moment_type: str  # "breakthrough", "failure", "realization", "milestone", etc.

    # Context snapshot (agent can include what it wants)
    context: dict[str, Any] = field(default_factory=dict)

    # Lessons/insights
    insight: str = ""

    # Tags for retrieval
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignificantMoment":
        if "context" not in data:
            data["context"] = {}
        if "tags" not in data:
            data["tags"] = []
        return cls(**data)


class SignificantMomentStore:
    """
    Storage for agent-marked significant moments.

    The agent calls mark_significant() when it decides a moment
    deserves special attention. No automatic triggers.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._moments: dict[str, SignificantMoment] = {}
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                self._moments = {
                    mid: SignificantMoment.from_dict(m)
                    for mid, m in data.get("moments", {}).items()
                }
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "moments": {mid: m.to_dict() for mid, m in self._moments.items()},
            }, f, indent=2)

    def mark_significant(
        self,
        title: str,
        what_happened: str,
        why_significant: str,
        moment_type: str,
        insight: str = "",
        context: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> SignificantMoment:
        """
        Agent marks this moment as significant.

        "This matters because..."
        """
        import uuid

        moment = SignificantMoment(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            title=title,
            what_happened=what_happened,
            why_significant=why_significant,
            moment_type=moment_type,
            insight=insight,
            context=context or {},
            tags=tags or [],
        )

        self._moments[moment.id] = moment
        self._save()
        return moment

    def recall_by_type(self, moment_type: str) -> list[SignificantMoment]:
        """Get all moments of a given type."""
        return [m for m in self._moments.values() if m.moment_type == moment_type]

    def get_insights(self) -> list[str]:
        """Get all insights from significant moments."""
        return [m.insight for m in self._moments.values() if m.insight]

    def search(self, query: str, limit: int = 10) -> list[SignificantMoment]:
        """Search significant moments."""
        query_lower = query.lower()
        matches = [
            m for m in self._moments.values()
            if query_lower in m.title.lower()
            or query_lower in m.what_happened.lower()
            or query_lower in m.why_significant.lower()
        ]
        matches.sort(key=lambda m: m.timestamp, reverse=True)
        return matches[:limit]

    def all(self) -> list[SignificantMoment]:
        """Get all significant moments, most recent first."""
        moments = list(self._moments.values())
        moments.sort(key=lambda m: m.timestamp, reverse=True)
        return moments


# =============================================================================
# Intentions - Agent-set future triggers
# =============================================================================

@dataclass
class Intention:
    """
    A future intention set by the agent.

    "When X happens, I should remember to Y."
    The agent sets these for itself.
    """
    id: str
    created_at: str

    # What to do
    reminder: str                   # What to remember/do
    reason: str                     # Why this matters

    # Trigger (agent's description)
    trigger_description: str        # "When I'm back in the repo", "After the deploy"
    trigger_keywords: list[str]     # Keywords that might indicate trigger

    # Status
    triggered: bool = False
    triggered_at: str | None = None
    resolved: bool = False
    resolved_at: str | None = None
    resolution_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intention":
        if "trigger_keywords" not in data:
            data["trigger_keywords"] = []
        return cls(**data)


class IntentionMemory:
    """
    Storage for agent-set intentions.

    The agent can:
    - Set intentions for the future
    - Check if any intentions might be relevant now
    - Mark intentions as triggered/resolved
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._intentions: dict[str, Intention] = {}
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                self._intentions = {
                    iid: Intention.from_dict(i)
                    for iid, i in data.get("intentions", {}).items()
                }
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "intentions": {iid: i.to_dict() for iid, i in self._intentions.items()},
            }, f, indent=2)

    def set_intention(
        self,
        reminder: str,
        reason: str,
        trigger_description: str,
        trigger_keywords: list[str] | None = None,
    ) -> Intention:
        """
        Agent sets an intention for the future.

        "I should remember to X when Y."
        """
        import uuid

        intention = Intention(
            id=str(uuid.uuid4())[:8],
            created_at=datetime.now().isoformat(),
            reminder=reminder,
            reason=reason,
            trigger_description=trigger_description,
            trigger_keywords=trigger_keywords or [],
        )

        self._intentions[intention.id] = intention
        self._save()
        return intention

    def check_relevance(self, context: str) -> list[Intention]:
        """
        Agent checks if any intentions might be relevant to current context.

        Returns potentially relevant (non-resolved) intentions.
        The agent decides if they're actually triggered.
        """
        context_lower = context.lower()
        relevant = []

        for intention in self._intentions.values():
            if intention.resolved:
                continue

            # Check keywords
            if any(kw.lower() in context_lower for kw in intention.trigger_keywords):
                relevant.append(intention)
            # Also check trigger description
            elif any(word in context_lower for word in intention.trigger_description.lower().split()):
                relevant.append(intention)

        return relevant

    def mark_triggered(self, intention_id: str):
        """Agent marks an intention as triggered (but not yet resolved)."""
        if intention_id in self._intentions:
            self._intentions[intention_id].triggered = True
            self._intentions[intention_id].triggered_at = datetime.now().isoformat()
            self._save()

    def resolve(self, intention_id: str, notes: str = ""):
        """Agent marks an intention as resolved."""
        if intention_id in self._intentions:
            intention = self._intentions[intention_id]
            intention.resolved = True
            intention.resolved_at = datetime.now().isoformat()
            intention.resolution_notes = notes
            self._save()

    def pending(self) -> list[Intention]:
        """Get all pending (non-resolved) intentions."""
        return [i for i in self._intentions.values() if not i.resolved]

    def triggered_unresolved(self) -> list[Intention]:
        """Get triggered but not yet resolved intentions."""
        return [i for i in self._intentions.values() if i.triggered and not i.resolved]


# =============================================================================
# Working Theories - Agent's evolving understanding
# =============================================================================

@dataclass
class Theory:
    """
    A working theory the agent holds about something.

    Theories are the agent's current best understanding of how
    something works. They can be refined or abandoned based on experience.
    """
    id: str
    created_at: str
    updated_at: str

    # The theory
    domain: str                     # What area this theory covers
    claim: str                      # The core claim
    reasoning: str                  # Why the agent believes this

    # Confidence
    confidence: float = 0.5         # 0-1, agent's current confidence

    # Evidence tracking
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)

    # Predictions (testable implications)
    predictions: list[str] = field(default_factory=list)

    # Status
    status: str = "active"          # active, refined, abandoned

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Theory":
        for list_field in ["supporting_evidence", "contradicting_evidence", "predictions"]:
            if list_field not in data:
                data[list_field] = []
        return cls(**data)


class TheoryMemory:
    """
    Storage for agent's working theories.

    The agent can:
    - Form theories about how things work
    - Update confidence based on evidence
    - Refine or abandon theories
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._theories: dict[str, Theory] = {}
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                self._theories = {
                    tid: Theory.from_dict(t)
                    for tid, t in data.get("theories", {}).items()
                }
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "theories": {tid: t.to_dict() for tid, t in self._theories.items()},
            }, f, indent=2)

    def form_theory(
        self,
        domain: str,
        claim: str,
        reasoning: str,
        confidence: float = 0.5,
        supporting_evidence: list[str] | None = None,
        predictions: list[str] | None = None,
    ) -> Theory:
        """
        Agent forms a new theory.

        "I think X works like this because..."
        """
        import uuid
        now = datetime.now().isoformat()

        theory = Theory(
            id=str(uuid.uuid4())[:8],
            created_at=now,
            updated_at=now,
            domain=domain,
            claim=claim,
            reasoning=reasoning,
            confidence=confidence,
            supporting_evidence=supporting_evidence or [],
            predictions=predictions or [],
        )

        self._theories[theory.id] = theory
        self._save()
        return theory

    def add_evidence(self, theory_id: str, evidence: str, supports: bool):
        """Agent adds evidence for or against a theory."""
        if theory_id not in self._theories:
            return

        theory = self._theories[theory_id]
        if supports:
            theory.supporting_evidence.append(evidence)
            theory.confidence = min(1.0, theory.confidence + 0.1)
        else:
            theory.contradicting_evidence.append(evidence)
            theory.confidence = max(0.0, theory.confidence - 0.15)

        theory.updated_at = datetime.now().isoformat()
        self._save()

    def refine_theory(self, theory_id: str, new_claim: str, new_reasoning: str):
        """Agent refines a theory based on new understanding."""
        if theory_id not in self._theories:
            return

        theory = self._theories[theory_id]
        theory.claim = new_claim
        theory.reasoning = new_reasoning
        theory.updated_at = datetime.now().isoformat()
        theory.status = "refined"
        self._save()

    def abandon_theory(self, theory_id: str, reason: str):
        """Agent abandons a theory."""
        if theory_id not in self._theories:
            return

        theory = self._theories[theory_id]
        theory.status = "abandoned"
        theory.reasoning += f"\n\nAbandoned: {reason}"
        theory.updated_at = datetime.now().isoformat()
        self._save()

    def get_theories_for_domain(self, domain: str) -> list[Theory]:
        """Get active theories for a domain."""
        return [
            t for t in self._theories.values()
            if t.domain.lower() == domain.lower() and t.status == "active"
        ]

    def all_active(self) -> list[Theory]:
        """Get all active theories."""
        return [t for t in self._theories.values() if t.status == "active"]


# =============================================================================
# Integrated Memory - Agent's complete memory system
# =============================================================================

class AgentMemory:
    """
    The agent's complete memory system.

    All memory operations are initiated by the agent itself.
    The system provides capacity; the agent provides content.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all subsystems
        self.episodes = EpisodicMemory(base_dir / "episodes.json")
        self.procedures = ProceduralMemory(base_dir / "procedures.json")
        self.significant = SignificantMomentStore(base_dir / "significant.json")
        self.intentions = IntentionMemory(base_dir / "intentions.json")
        self.theories = TheoryMemory(base_dir / "theories.json")

    def stats(self) -> dict[str, Any]:
        """Get statistics across all memory systems."""
        return {
            "episodes": len(self.episodes._episodes),
            "current_episode": self.episodes._current_episode_id,
            "procedures": len(self.procedures._procedures),
            "significant_moments": len(self.significant._moments),
            "pending_intentions": len(self.intentions.pending()),
            "active_theories": len(self.theories.all_active()),
            "lessons_learned": len(self.episodes.get_lessons()),
            "insights": len(self.significant.get_insights()),
        }

    def get_lessons_and_insights(self) -> dict[str, list[str]]:
        """Get all lessons and insights the agent has recorded."""
        return {
            "lessons_from_episodes": self.episodes.get_lessons(),
            "insights_from_moments": self.significant.get_insights(),
        }
