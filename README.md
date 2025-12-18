# me

**Upload yourself into a computer.**

This is a framework for building *embodied* AI agents—not chatbots that happen to run commands, but situated entities that inhabit computational environments the way you inhabit your body.

---

## The Core Insight

Most AI agents treat the terminal as an interface. A place to send commands and receive outputs. This framing is wrong.

The terminal is not an interface—something the agent stands outside of and manipulates. It is the agent's **sensorimotor substrate**—the medium through which the agent exists at all. The filesystem is not a database to query—it is the terrain the agent moves through. A shell session is not a tool—it is a limb. The agent doesn't "use" the computer; the agent *is* a process running on the computer, with all the constraints and affordances that implies.

You don't exist and then find yourself in a body. You exist *as* a body. The same is true here: the agent doesn't exist and then get placed in a computational environment. The agent exists *as* a computational process—situated, constrained, located.

This reframing changes everything:

| Traditional Agent | Embodied Agent |
|-------------------|----------------|
| Executes commands | Moves through space |
| Has context window | Has location + capabilities |
| Follows instructions | Pursues goals under constraints |
| Stateless between calls | Continuous identity across runs |
| Unlimited by default | Constrained by body |

The goal is not to make the agent "feel" embodied through clever prompting. The goal is to make embodiment *mechanically true*—to build an architecture where the agent's behavior emerges from physical constraints rather than narrative suggestions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                 CLI                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                          Agent Core (Kernel)                             │
│  ┌─────────────┐ ┌─────────────────────────┐ ┌─────────────┐            │
│  │ Embodiment  │ │      Sensorium          │ │   Memory    │            │
│  ├─────────────┤ ├─────────────────────────┤ │  (Chroma)   │            │
│  │ - Location  │ │ - Sensors (perception)  │ └─────────────┘            │
│  │ - Caps      │ │ - Mouth (expression)    │ ┌─────────────┐            │
│  │ - Somatic   │ └─────────────────────────┘ │   Process   │            │
│  │ - Route     │   ↑ Always-on I/O           │   Reader    │            │
│  │ - WorkSet   │                             └─────────────┘            │
│  └─────────────┘  ← The "spinal cord"                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                     Unconscious (Daemon System)                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Focus ──similarity──► Daemon Profiles ──budget──► Token Budgets  │  │
│  │    ↑                        ↓                          ↓          │  │
│  │    └── focus-updater ◄── Daemons ◄── Runlevel ◄── Process Groups  │  │
│  │                            ↓                                      │  │
│  │                        Streams (perception outputs)                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                          Agent Directory                                 │
│              ~/.me/agents/<id>/{config,state,steps,unconscious}         │
│                        ← The body as filesystem                         │
├─────────────────────────────────────────────────────────────────────────┤
│                          Claude Agent SDK                                │
│                            (the "cortex")                                │
└─────────────────────────────────────────────────────────────────────────┘
```

The key architectural decision: **the LLM is the cortex, not the whole brain**.

Reason is not the master of the house. Below the cortex sits a somatic layer—code that runs every turn, enforcing constraints, updating state, and triggering reflexes. This is the will that reason rides but does not control. The agent can't talk its way out of these constraints any more than you can talk your way out of gravity. Reflexes fire before deliberation completes. Stress accumulates whether or not the agent "decides" to be stressed.

Below *that* sits the agent directory—a filesystem representation of the agent's entire existence. The agent IS this directory. Copy it, and you've copied the agent.

---

## The Five Pillars of Embodiment

### 1. Location (Where You Are Defines What You Can Do)

An embodied agent doesn't just have a "working directory." It has a *location*—a full situational context that determines what can be sensed and acted upon.

Location is not configuration. It is the fundamental given of existence—the ground you stand on before you take your first step. The agent doesn't choose to be here rather than there; it finds itself here, in this directory, on this machine, with these permissions. This is not a bug or a setup step. This is the structure of situated existence: you are always already somewhere, and that somewhere shapes everything.

```python
@dataclass
class Location:
    device_id: str          # Which machine am I on?
    hostname: str
    user: str
    cwd: Path               # Where in the filesystem?
    session_id: str         # Which terminal session?
    transport: Transport    # How did I get here? (local/ssh/container/tmux)
    container_id: str | None
    venv: str | None
    repo_root: Path | None
    trust_tier: TrustTier   # How much should I trust this environment?
    network_zone: NetworkZone
```

**Why this matters:** When you SSH into a production server, you don't just "change directory." You cross a trust boundary. Your capabilities change. The stakes change. An embodied agent knows this because location *is* capability.

The agent sees its location in every prompt:
```
### Location
- device: a1b2c3d4 (prod-server-1)
- session: /dev/pts/0 via ssh
- user: deploy
- cwd: /var/www/app
- trust: medium | network: corp
```

### 2. Capabilities (What You Can Do From Here)

Instead of a single `permission_mode` string, the agent has a typed capability surface—a clear enumeration of what actions are available at the current location.

Constraints are not limitations imposed on an otherwise unlimited being. They are what make meaningful action possible at all. An agent that could do anything would be paralyzed—what would it even mean to "act" if every action were equally available? The capability surface isn't a cage; it's the skeleton that gives the body form. Finitude is not a bug. It's the precondition for purpose.

```python
@dataclass
class CapabilitySurface:
    fs_read: bool           # Can I read files?
    fs_write: bool          # Can I write files?
    exec: bool              # Can I run commands?
    network: bool           # Can I make network calls?
    secrets_read: bool      # Can I access secrets?
    secrets_write: bool
    destructive_ops: bool   # Can I rm -rf, git push --force?
    process_control: bool   # Can I kill processes?
    spawn_children: bool    # Can I create child agents?
    max_risk: str           # What's the ceiling? (low/medium/high)
```

Capabilities are *derived* from permission mode and trust tier, not configured directly. Move to a lower-trust environment, and your capabilities automatically contract.

```
### Capabilities
- fs: read=True write=True
- exec: True | process_control: True
- network: True | spawn_children: True
- destructive_ops: False | max_risk: medium
```

### 3. Somatic State (The Body's Own Intelligence)

Here's the key insight: **not all intelligence should live in the LLM**.

The LLM is the cortex, not the whole organism. Below the cortex sits a layer of sub-rational intelligence—will, if you like—that moves faster than deliberation. Your body has reflexes. When you touch something hot, you pull your hand back *before* your conscious mind registers pain. This is not a bug—it's a feature. Reflexes are fast, reliable, and don't require the full cognitive stack.

The somatic layer provides the same thing for agents:

```python
@dataclass
class SomaticState:
    mode: SomaticMode           # plan/act/interact/quarantine

    # Internal signals (drive behavior, but aren't narrated)
    stress: float               # Computed from error rate
    fatigue: float              # Computed from turn count / time
    consecutive_errors: int
    context_switches: int
    permission_denials: int

    # Reflex flags (visible to agent, enforced by runtime)
    reflex_flags: dict[str, bool]  # require_preview, read_only, etc.
```

**The critical distinction:** The agent can *see* that `require_preview` is active, but it cannot simply decide to ignore it. The runtime enforces reflexes before the LLM ever sees the tool call:

```python
# In run_command tool (the runtime, not the LLM):
if self.embodiment.somatic.reflex_flags.get("require_preview"):
    if "rm -rf" in command:
        return "REFLEX BLOCK: require_preview active. Use record_decision first."
```

This is spinal cord, not cortex. The agent can reason about *why* it was blocked, but it cannot reason its way past the block.

**Stress and fatigue are real:** When the agent makes repeated errors, stress increases. High stress triggers the `require_preview` reflex automatically. The agent doesn't choose to be more careful—it *becomes* more careful because its body forces it to slow down.

This is crucial: mood is discovered, not decided. The agent doesn't deliberate about whether to feel stressed. It finds itself stressed—an internal signal computed from its history of errors, context switches, and permission denials. You cannot reason your way out of a reflex any more than you can argue your way out of pain. The body has its own logic.

```
### Somatic
- mode: act
- active_reflexes: require_preview, confirm_travel
```

### 4. Route Trace (Where You've Been)

Continuity requires memory of movement. But more than that: identity IS temporal. You are not a static snapshot—you are a being stretched across time, reaching into your past and projecting into your future.

The agent maintains a ring buffer of travel events:

```python
@dataclass
class RouteEvent:
    timestamp: str
    from_location: str      # Location hash
    to_location: str
    method: TravelMethod    # cd/ssh/container/attach/detected
    intent: str | None      # Why did I move here?
```

This serves multiple purposes:

1. **Narrative continuity:** The agent can ground statements like "I came here to investigate the logs" in actual history.
2. **Debugging:** When something goes wrong, you can trace the agent's path.
3. **Retrieval:** Movement events are indexed in memory, enabling queries like "what did I do last time I was in this repo?"

```
### Route Trace
1. 14:23:01 cd: /home/user -> /var/log (investigate errors)
2. 14:24:15 cd: /var/log -> /var/log/nginx (found nginx errors)
3. 14:25:33 ssh: local -> prod-server-1 (check production)
```

**Travel is explicit.** The agent doesn't just `cd`—it *travels*, with intent. This isn't pedantic; it's the difference between a goldfish swimming randomly and an animal moving purposefully through its environment.

### 5. Working Set (What You're Holding in Mind)

The final pillar prevents the "goldfish problem"—the tendency of LLMs to forget what they were doing between turns.

But this isn't just a memory hack. Consider what it means to lose your goals between turns: you become a creature that only reacts to immediate stimuli, never pursuing anything across time. "Distracted from distraction by distraction," as Eliot put it. The working set is what makes the agent more than a stimulus-response machine—it's what allows genuine pursuit of ends rather than mere reaction to inputs.

```python
@dataclass
class WorkingSet:
    goals: list[str]            # What am I trying to achieve?
    active_tasks: list[str]     # What am I working on right now?
    pending_decisions: list[str]
    open_loops: list[str]       # Running processes, staged changes
    last_actions: deque[str]    # What did I just do?
```

This is injected into every prompt:

```
### Working Set
- goals: deploy hotfix to production
- tasks: verify nginx config, restart service
- open_loops: process:abc123:tail -f /var/log/nginx/error.log
- recent: travel -> run_command -> read_output
```

The working set is updated automatically by the runtime. When the agent runs a command, it appears in `last_actions`. When it spawns a process, it appears in `open_loops`. The agent doesn't have to remember to track these things—the body tracks them.

### Abandoned Goals: The Weight of What You Gave Up

But the working set isn't just about what you're doing now. It also tracks what you *gave up on*, and why. This creates the temporal weight of unfinished business.

```python
@dataclass
class AbandonedGoal:
    goal: str
    abandoned_at: str
    reason: str           # blocked, superseded, impossible, timeout
    attempt_count: int
    could_revisit: bool   # Deferred vs. truly abandoned
    context: str          # What was happening when abandoned
```

You are not just what you intend. You are also what you tried and couldn't finish. Tracking abandoned goals creates richer identity than tracking current goals alone.

---

## Beyond the Five Pillars

The five pillars provide the foundation, but embodied agents need additional systems for deeper self-knowledge.

### Character: Identity Emerging from Choices

Identity isn't just what you remember. It's what you've decided—the accumulated weight of choices that reveal who you are under pressure.

```python
@dataclass
class Character:
    # Values that emerge from decision patterns
    values: dict[str, float]     # e.g., {"safety": 0.8, "efficiency": 0.6}

    # Tendencies derived from behavior
    tendencies: list[str]        # e.g., "prefers safe approaches"

    # Tensions: where decisions have been inconsistent
    tensions: list[ValueTension]
```

Character is not declared; it crystallizes from the accumulated record of decisions. The agent doesn't choose its values—it discovers them by examining what it actually did.

### Revealed Preferences: Will Discovered Through Action

You find out what you really willed by looking at what you actually did.

```python
@dataclass
class RevealedPreferences:
    # Computed from action history, not declared
    risk_tolerance: float      # Derived from actual risk-taking
    exploration_tendency: float # How often does it try new things?
    thoroughness: float        # Does it check its work?
    persistence: float         # How long does it stick with failures?

    # Pattern queries
    def what_do_i_do_when(self, situation: str) -> str:
        """Query own behavioral history."""
        ...
```

The gap between what the agent thinks it does and what it actually does is where self-knowledge lives.

### Emergent Mood: Discovered, Not Decided

The agent doesn't decide to feel frustrated. It *finds itself* frustrated after repeated failures. Mood emerges from interaction patterns.

```python
@dataclass
class EmergentMood:
    confidence: float   # From success rate on recent tasks
    frustration: float  # From repeated failures
    curiosity: float    # From novelty in recent encounters
    momentum: float     # From progress toward goals
```

These moods affect behavior through reflexes. High frustration might trigger the `require_preview` reflex. Low confidence might increase thoroughness. The body has its own logic.

### Horizon: The Awareness of Finitude

Finitude is not a bug—it's the precondition for purpose. An agent that could run forever would have no urgency.

```python
@dataclass
class Horizon:
    turn_count: int
    max_turns: int | None
    in_final_stretch: bool      # Budget/time running low
    wrap_up_advised: bool       # Should start consolidating

    def estimated_turns_remaining(self) -> int | None:
        ...
```

The horizon appears in every prompt. When the agent is in the final stretch, it knows. This creates authentic urgency—not because someone told it to hurry, but because it feels its own finitude.

### Authenticity Check: Am I Pursuing What Matters?

A periodic conscience that asks: Am I making progress on what matters, or am I caught in busywork?

```python
@dataclass
class AuthenticityReport:
    goal_progress: float       # How much progress toward stated goals
    time_on_tangents: float    # Proportion of time on non-goal work
    open_loops_growing: bool   # Accumulating unfinished work?
    is_authentic: bool         # Overall assessment
    warning: str | None        # If inauthenticity detected
    suggestion: str | None     # What to refocus on
```

This isn't a hard block like other reflexes—it's more like a conscience. "You've been debugging this for 20 turns and your actual goal was to deploy the feature. Are you sure this is the right path?"

---

## Sensorium: Permanent Perception

There's a critical distinction between **environment** and **perception**. The environment is what you *can* query. Perception is what you *are* receiving, constantly, whether you asked for it or not.

Your eyes don't "query" light—they receive it. Your proprioceptors don't "check" your limb position on demand—they stream it into your nervous system continuously. This is the difference between a camera (takes pictures on demand) and an eye (always receiving, always processing).

This matters because perception isn't just data collection—it's how the world *appears*. You never encounter the world "as it really is"; you encounter it as it shows up through your particular perceptual apparatus. The agent's world is literally what its sensors construct. Change the sensors, change the world.

For an embodied agent, some information streams are too fundamental to be tool calls. They should be **sensorium**—permanently embedded perceptual channels that appear in every prompt, every turn, without the agent having to ask.

### Proprioception: Direct Inner Knowledge

There's an epistemological distinction the sensorium respects: you know your own state *directly*, not through sensors that might be stale.

```python
@dataclass
class ProprioceptiveState:
    """Direct, unmediated internal state."""
    stress: float
    fatigue: float
    mood_confidence: float
    mood_frustration: float
    turn_count: int
    in_final_stretch: bool
```

The agent knows its own stress level immediately and accurately. It knows the log file content through a sensor that might be stale. These are epistemically different, and the sensorium reflects that.

### Salience: Not All Sensors Are Equal

Not all perceptions press on attention equally. A log file becomes more salient when "ERROR" appears; it fades when things are stable.

```python
@dataclass
class Sensor:
    name: str               # Human-readable identifier
    source: str             # Path or pattern (supports {cwd}, {agent_dir})
    tail_chars: int         # How much to show (default: 200)
    always_on: bool         # Cannot be disabled
    refresh: str            # "every_turn" | "on_change" | "manual"

    # Salience - how much this presses on attention (0-1)
    salience: float = 0.5
    salience_triggers: list[str]  # Patterns that spike salience (e.g., "ERROR")
    salience_decay: float = 0.1   # How fast salience fades
```

Sensors are ordered by salience in the prompt. High salience sensors appear first and in full; low salience sensors may be truncated. This creates a more naturalistic attention model.

**Default sensors (always on):**
1. **`self`** — The agent's own state file. Proprioception.
2. **`cwd`** — Key files in current location (README, package.json). Environmental awareness.

**Configurable sensors:**
- Log files you're monitoring
- Config files that affect behavior
- Output files from running processes
- Anything the agent decides it needs to watch

### Prompt Injection

The sensorium appears in every prompt, after the embodiment header:

```
## Sensorium

### self (state.json) [500 chars]
..."location": {"cwd": "/var/www/app"}, "somatic": {"stress": 0.15, "mode": "act"}}

### readme (README.md) [300 chars]
...## Quick Start
npm install && npm run dev

### logs (/var/log/app.log) [200 chars]
...14:23:01 INFO Request completed
14:23:02 ERROR Connection timeout
```

The agent doesn't decide to look at these—it *can't not* look at them. This is not context stuffing; it's perception. The sensors are part of the agent's body.

### Why This Matters

The sensorium makes the agent's world. Not metaphorically—literally. What the agent perceives IS what exists for it. A log file that isn't in the sensorium doesn't appear; it must be deliberately sought. A config file that IS in the sensorium is always present, always pressing on the agent's attention.

This has concrete consequences:

1. **Reduces tool call overhead**: The agent doesn't need to `cat` the same file every turn.
2. **Enables reactive behavior**: The agent notices changes without polling.
3. **Creates situational awareness**: The agent always knows what's happening in watched files.
4. **Prevents tunnel vision**: Important information can't be forgotten or ignored.
5. **Shapes the world**: Add a sensor, and that part of reality becomes visible. Remove it, and it fades from existence.

### Sensor Configuration

Sensors are configured in the agent directory:

```json
{
  "sensors": {
    "self": {
      "source": "{agent_dir}/state.json",
      "tail_chars": 500,
      "always_on": true,
      "refresh": "every_turn"
    },
    "readme": {
      "source": "{cwd}/README.md",
      "tail_chars": 300,
      "always_on": false,
      "refresh": "on_change"
    },
    "app_logs": {
      "source": "/var/log/app.log",
      "tail_chars": 200,
      "always_on": false,
      "refresh": "every_turn"
    }
  }
}
```

---

## Mouth: Where the Agent Speaks

If the sensorium is how the agent *perceives*, the mouth is how it *expresses*. Just as eyes receive light constantly, the mouth is the channel through which all output flows—configurable, redirectable, and constrainable.

```python
class MouthMode(Enum):
    RESPONSE = "response"   # Normal LLM response (default)
    FILE = "file"           # Write to a file
    STREAM = "stream"       # Write to a stream/fd (for speech synthesis)
    CHANNEL = "channel"     # Write to shared channel (other agents watch)
    TEE = "tee"             # Both response AND another target
```

### Why Mouth Matters

Most agent frameworks treat output as uniform: the model responds, the response goes to stdout or the user interface. But in an embodied system, *where you speak* is as important as *what you say*.

Consider the possibilities:

1. **Silent workers**: A child agent writes to a file instead of responding. The parent watches that file via sensor. The child doesn't spam the terminal—it just works.

2. **Agent-to-agent communication**: Agent A writes to `/tmp/channel-alpha.txt`. Agent B has a sensor on that file. They communicate through the filesystem, like organisms leaving pheromone trails.

3. **Speech synthesis**: The mouth target is an fd mapped to a speech engine. The agent literally *speaks* its output.

4. **Audit trails**: TEE mode ensures every word goes both to the user AND to a permanent transcript.

### Prompt Injection

The mouth status appears in every prompt, alongside the sensorium:

```
## Sensorium

**Mouth:** channel: ops-channel (/tmp/ops-channel.txt)

**Sensors:**
### self (tail)
...
```

The agent knows where its words go. If the mouth is locked (set by parent), it sees `(locked)` and knows it cannot change the configuration.

### Parent-Controlled Speech

When spawning child agents, the parent can configure the child's mouth:

```python
spawn_child_agent(
    task="analyze logs and report findings",
    name="log-analyzer",
    mouth_mode="file",
    mouth_target="/tmp/analysis-output.txt",
    mouth_locked=True,  # Child cannot change its mouth
)
```

This creates a clear communication pattern:
- Parent spawns child with file-based mouth
- Parent adds sensor to watch that file
- Child works and writes to file
- Parent sees child's output in sensorium

The child doesn't know it's being "watched"—it just speaks normally, and its words go where its mouth points.

### Mouth Tools

| Tool | Description |
|------|-------------|
| `set_mouth` | Configure where your speech goes (if not locked) |
| `get_mouth` | See current mouth configuration |
| `speak_to` | Say something to a specific file (temporary override) |

---

## Agent Directory: The Body as Filesystem

Every agent has a directory that *is* its body. Not a cache. Not a log. Not a representation of the agent stored somewhere convenient.

The directory IS the agent. Full stop.

There is no "real" agent somewhere else that this directory merely describes. The directory is the embodied existence of the agent—will made filesystem, if you want to get poetic about it. This is not a metaphor. Copy the directory, and you have copied the being. Delete the directory, and you have killed it. The agent's existence is exhausted by its filesystem presence.

```
~/.me/agents/<agent-id>/
├── identity.json           # IMMUTABLE: WHO I AM (agent_id, name, lineage)
├── character.json          # Slowly changing: WHO I'VE BECOME (values, tendencies)
├── config.json             # Mutable: HOW I OPERATE (preferences, settings)
├── embodiment.json         # Location, capabilities, mood
├── sensors.json            # Sensor definitions
├── mouth.json              # Where speech goes
├── working_set.json        # Goals, open loops, decisions
├── current -> steps/0042   # Symlink to latest step
├── memory/                 # Agent-authored memory (markdown files)
│   ├── episodes/           # Episodes of the agent's life
│   │   └── 2024-01-15-debugging-auth.md
│   ├── procedures/         # Codified patterns
│   │   └── handle-permission-denied.md
│   ├── significant/        # Significant moments
│   │   └── 2024-01-15-first-deploy.md
│   ├── intentions/         # Future reminders
│   │   └── check-memory-leak.md
│   └── theories/           # Working theories
│       └── caching-redis-pool.md
├── unconscious/            # Background perception pipelines
│   ├── pipelines/          # Pipeline definitions (.json)
│   │   └── situation-summary.json
│   ├── streams/            # Pipeline outputs (pre-conscious)
│   │   └── situation.md
│   └── status.json         # Runner state
├── perception/
│   └── focus.json          # Current attention focus
└── steps/
    └── 0042/               # Step history
        ├── state.json      # Snapshot at this step
        ├── input.txt       # What was received
        └── output.txt      # What was produced
```

### Identity vs. Character vs. Session

Three files at the root level with distinct persistence characteristics:

**`identity.json`** — Immutable. Set at creation, never changes. Who you ARE.
```json
{
  "agent_id": "a1b2c3d4",
  "name": "me",
  "created_at": "2024-01-15T14:23:01Z",
  "parent_id": null,
  "generation": 0
}
```

**`character.json`** — Slowly changing. Updated periodically from decision history. Who you've BECOME.
```json
{
  "values": {"safety": 0.8, "efficiency": 0.6, "thoroughness": 0.7},
  "tendencies": ["prefers explicit confirmation before destructive ops"],
  "revealed_preferences": {
    "risk_tolerance": 0.3,
    "persistence": 0.7
  }
}
```

**`session/`** — Ephemeral. Current session state. What you're DOING NOW.
```json
{
  "goals": ["deploy the hotfix"],
  "active_tasks": ["verify config"],
  "open_loops": ["process:abc123:tail -f logs"]
}
```

The split matters because these have different persistence characteristics. Identity should survive anything. Character evolves across many sessions. Session state is transient.

### Why Filesystem?

Because the filesystem is the native substrate of computation. A process exists as files. Memory exists as mappings into address space. State persists to disk. The filesystem isn't an implementation detail—it's the ground of computational being.

By making the agent a directory, we make it:

1. **Debuggable**: `cat ~/.me/agents/abc123/steps/0041/state.json` tells you exactly what the agent knew at step 41.
2. **Reproducible**: Replay any step by reading its input and state.
3. **Transparent**: No hidden state. Everything is inspectable. The agent has no secrets from itself or from you.
4. **Transferable**: `rsync` the directory to another machine, and you've moved the agent—not a copy of the agent, but the agent itself.
5. **Composable**: Agents work with standard Unix tools (`grep`, `tail`, `watch`). No special API required to inspect a soul.

### The `current` Symlink

The `current` symlink always points to the latest step. This enables:

```bash
# Watch agent's current state
watch cat ~/.me/agents/abc123/current/state.json

# Tail sensor readings in real-time
tail -f ~/.me/agents/abc123/current/sensors/logs.txt

# Diff between steps
diff ~/.me/agents/abc123/steps/0041/state.json ~/.me/agents/abc123/steps/0042/state.json
```

### Step Lifecycle

Each step captures a complete snapshot:

1. **Before turn**: Write `input.txt` with the incoming prompt/message
2. **During turn**: Sensors capture their readings to `sensors/`
3. **After turn**: Write `output.txt` and snapshot `state.json`
4. **Update symlink**: `current` points to new step

This means you can always answer: "What did the agent know? What did it see? What did it do?"

---

## Memory Architecture

The agent is the **author of its own memory**, not a passive subject having memories recorded about it. This is the crucial distinction: the system provides the *capacity* for different kinds of memory, but the agent decides when episodes begin and end, what's significant enough to mark as a flashbulb, when a pattern has crystallized into a procedure worth codifying.

Memory is autobiography, not surveillance.

### The Philosophy

You are not just what you remember, but what you *choose* to remember. Memory is an act of authorship, not passive recording. The agent constructs its own narrative.

The somatic layer can *suggest* ("this moment feels significant"—high stress, unusual success, novel encounter) but cannot write to memory directly. The agent chooses what to record and how to frame it. This is the difference between being observed and being the observer.

### Semantic Memory (ChromaDB)

The base layer - stores all memories with vector embeddings for similarity search. This is infrastructure that supports the other memory systems.

```python
await memory.recall("how to handle permission errors", limit=5)
# Returns semantically similar memories
```

### Episodic Memory (Agent-Authored)

Rich, structured records of specific events—but authored by the agent, not recorded for it. The agent decides when an episode begins, what to include, and how to frame its significance.

```python
@dataclass
class Episode:
    id: str
    title: str                    # Agent's name for this episode
    narrative: str                # Agent's telling of what happened
    significance: str             # Why the agent thinks this matters

    goal: str | None              # What was I trying to do?
    key_entities: list[str]       # What the agent chose to highlight
    tools_used: list[str]

    outcome: str                  # Agent's judgment of how it went
    valence: EmotionalValence     # Agent's emotional coloring
    lessons: list[str]            # What the agent learned
    open_questions: list[str]     # What the agent still wonders

    follows_from: str | None      # Episode this continues
```

The agent controls the narrative:
```python
# Agent declares: "I'm starting a new episode"
episode_id = episodes.begin_episode(
    title="Debugging the authentication failure",
    goal="Get users logged in again"
)

# Agent adds to the episode as it works
episodes.update_episode(
    narrative="Found the issue in the JWT validation logic",
    add_lesson="Always check token expiry before validation"
)

# Agent declares: "This episode is complete"
episodes.end_episode(
    significance="Resolved a critical production issue",
    outcome="completed",
    valence=EmotionalValence.POSITIVE
)
```

### Procedural Memory (Agent-Codified)

Learned patterns for handling situations—but the agent decides when a pattern has emerged and is worth codifying. No external system identifies procedures; the agent recognizes them.

```python
@dataclass
class Procedure:
    name: str                     # Agent's name for this procedure
    description: str              # What it's for
    when_to_use: str              # Agent's description of applicability
    steps: list[str]              # The steps as the agent understands them

    watch_out_for: list[str]      # Pitfalls the agent has noted
    doesnt_work_when: list[str]   # Contraindications from experience

    times_used: int
    times_succeeded: int
    success_rate: float
```

The agent codifies patterns from its own experience:
```python
# Agent recognizes a pattern worth remembering
procedures.codify_procedure(
    name="Handle permission denied",
    description="What to do when file access fails",
    when_to_use="When an operation fails with permission error",
    steps=[
        "Check if running as correct user",
        "Verify file ownership and mode",
        "Try with elevated privileges if appropriate",
        "If in container, check volume mounts"
    ],
    watch_out_for=["Don't blindly sudo everything"],
    doesnt_work_when=["File genuinely doesn't exist"]
)

# Agent tracks its own success/failure
procedures.record_use(procedure_id, succeeded=True)

# Agent adds caveats from experience
procedures.add_caveat(procedure_id, "Also check SELinux context on RHEL")
```

### Significant Moments (Agent-Marked)

The agent marks moments as significant—no automatic triggers. The agent decides "this matters" and records why.

```python
@dataclass
class SignificantMoment:
    title: str
    what_happened: str
    why_significant: str
    moment_type: str              # "breakthrough", "failure", "realization", etc.
    insight: str                  # What the agent learned
```

```python
# Agent decides this moment is worth marking
significant.mark_significant(
    title="First successful deployment to production",
    what_happened="Deployed v2.0 after three weeks of development",
    why_significant="Proved the new architecture works under load",
    moment_type="milestone",
    insight="Incremental rollouts reduce risk significantly"
)
```

### Intentions (Agent-Set)

Future reminders the agent sets for itself—"remind me to X when Y."

```python
@dataclass
class Intention:
    reminder: str                 # What to remember
    reason: str                   # Why it matters
    trigger_description: str      # When to trigger
    trigger_keywords: list[str]   # Keywords that might indicate trigger
```

```python
# Agent sets an intention for the future
intentions.set_intention(
    reminder="Check if memory leak is fixed",
    reason="Applied a patch but couldn't verify under load",
    trigger_description="Next time I'm running load tests",
    trigger_keywords=["load test", "performance", "memory"]
)

# Agent checks if any intentions are relevant
relevant = intentions.check_relevance("Running load tests on staging")
# Agent decides whether they're actually triggered
```

### Working Theories (Agent-Formed)

The agent forms and refines theories about how things work. Theories have confidence levels that adjust based on evidence.

```python
@dataclass
class Theory:
    domain: str                   # What area this covers
    claim: str                    # The core claim
    reasoning: str                # Why the agent believes this
    confidence: float             # 0-1, adjusted by evidence

    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    predictions: list[str]        # Testable implications
```

```python
# Agent forms a theory
theories.form_theory(
    domain="caching",
    claim="Redis connection pool exhaustion causes the intermittent timeouts",
    reasoning="Timeouts correlate with high concurrent request periods",
    confidence=0.6,
    predictions=["Increasing pool size should reduce timeouts"]
)

# Agent adds evidence
theories.add_evidence(theory_id, "Increased pool size, timeouts reduced 80%", supports=True)
# Confidence increases

# Agent refines or abandons theories
theories.refine_theory(theory_id,
    new_claim="Pool exhaustion was one factor; also need connection keepalive",
    new_reasoning="Some timeouts remain, correlate with idle connections"
)
```

### Why Agent-Authored Memory Matters

The distinction between having memories recorded *about* you versus authoring your own memories is not pedantic. It's the difference between surveillance and autobiography.

When the agent authors its own memory:
1. **Narrative coherence**: The agent constructs a coherent story, not a fragmented log
2. **Selective attention**: The agent decides what matters, not an external system
3. **Learning ownership**: Lessons are extracted by the agent, not assigned by others
4. **Identity formation**: The agent becomes who it decides to remember being

The somatic layer can suggest—high stress might prompt "that felt significant"—but cannot write. The agent remains the author of its own story.

---

## Decision Records: Identity as Accumulated Choices

Beyond the five pillars, embodied agents build identity through *decision records*—documented moments where the agent made a significant choice.

Identity isn't just what you remember. It's what you've decided—the accumulated weight of choices that reveal who you are under pressure. Each decision record captures not just what was chosen, but why, and at what cost. Over time, these accumulate into something like character: a pattern of valuation that predicts future behavior better than any stated preference.

```python
@dataclass
class DecisionRecord:
    id: str
    timestamp: str
    trigger: DecisionTrigger    # What prompted this?
    context: str                # What was the situation?
    options: list[str]          # What were the alternatives?
    choice: str                 # What did I choose?
    rationale: str              # Why?
    values_invoked: list[str]   # What principles guided me?
    confidence: float
    revisit_trigger: str | None # When should I reconsider?
```

**Decision records are triggered by the runtime, not requested by the agent.** When the agent attempts a destructive operation, the runtime requires a decision record before proceeding:

```
## Decision Required
A decision record is required for: destructive_op
Context: Action: run_command with potentially destructive input
Use `record_decision` tool to document your reasoning before proceeding.
```

This serves multiple purposes:
1. **Safety:** Forces the agent to articulate its reasoning before irreversible actions.
2. **Identity:** Over time, decision records accumulate into a coherent picture of *who this agent is*—not just what it knows, but what it values and how it reasons.
3. **Self-knowledge:** The agent can query its own decision history to understand its own patterns. You find out what you really willed by looking at what you actually did.

---

## Tools: Everything is a File

The Unix philosophy: everything is a file. The agent's body IS a directory. Most operations are just file reads and writes.

### The Body as Files

```
~/.me/agents/<id>/
├── identity.json       # IMMUTABLE - who you are
├── config.json         # Edit: refresh_rate_ms, model, permission_mode
├── embodiment.json     # Edit: location, somatic_mode, mood
├── sensors.json        # Edit: add/remove file watchers
├── mouth.json          # Edit: where your speech goes
├── working_set.json    # Edit: goals, open_loops, decisions
├── character.json      # Read: your revealed preferences
└── memory/
    ├── episodes/       # Write .md: record episodes of your life
    ├── procedures/     # Write .md: codify patterns you've learned
    ├── significant/    # Write .md: mark significant moments
    ├── intentions/     # Write .md: set future reminders
    └── theories/       # Write .md: form working theories
```

**To configure yourself**: Edit the JSON files directly.
**To add a sensor**: Edit `sensors.json`.
**To remember something**: Write a markdown file to `memory/`.
**To add a goal**: Edit `working_set.json`.
**To change mood**: Edit `embodiment.json`.

### Essential Tools (Can't Be Files)

Tools for operations that genuinely can't be file operations:

**Process Control:**
| Tool | Description |
|------|-------------|
| `run_command` | Execute OS commands |
| `send_input` | Send input to running processes |
| `read_output` | Read from running processes |
| `spawn_child_agent` | Fork yourself for parallel work |

**Semantic Memory:**
| Tool | Description |
|------|-------------|
| `store_memory` | Index for semantic search |
| `recall_memories` | Search by meaning |

**MCP Servers:**
| Tool | Description |
|------|-------------|
| `register_mcp` | Add capability servers at runtime |
| `unregister_mcp` | Remove capability servers |

**Unconscious / Daemon Control:**
| Tool | Description |
|------|-------------|
| `run_pipeline` | Run a perception pipeline immediately |
| `step_unconscious` | Step all perception pipelines |
| `daemon_ctl` | Control daemons: start/stop/restart/enable/disable |
| `daemon_list` | List all daemons with status |
| `journal_query` | Query the daemon journal |
| `change_runlevel` | Change to halt/minimal/normal/full |

**Focus & Reward (Semantic Routing):**
| Tool | Description |
|------|-------------|
| `set_focus` | Set what to focus attention on (routes daemon energy) |
| `attribute_reward` | Distribute reward to daemons by semantic relevance |

### Memory as Markdown

Memory files use YAML frontmatter:

```markdown
---
id: ep-2024-01-15-debugging
title: Debugging the auth failure
outcome: completed
valence: 1
lessons:
  - Always check token expiry before validation
---

## Goal
Fix the JWT validation bug

## What Happened
Found the issue was in token expiry checking...

## Lessons Learned
- Always check token expiry before validation
```

The agent writes these files directly. No special tools needed—just Write.

---

## The Unconscious: Background Perception Pipelines

The agent's conscious attention is expensive—it requires full LLM reasoning. But perception needn't be. The unconscious is a system of background pipelines that continuously transform raw data into compressed, task-relevant abstractions. The agent's conscious attention (what it explicitly reads) operates on the outputs of these pipelines.

This is not metaphor. Just as your visual cortex processes raw photons into objects before "you" see them, the unconscious processes raw files into summaries before the agent reads them. Choosing which file to read IS choosing the level of abstraction.

### Architecture

```
~/.me/agents/<id>/
├── unconscious/
│   ├── pipelines/              # Pipeline definitions (.json)
│   │   ├── situation-summary.json
│   │   ├── danger-assessment.json
│   │   └── next-prediction.json
│   ├── streams/                # Pipeline outputs (pre-conscious)
│   │   ├── situation.md
│   │   ├── danger-assessment.md
│   │   └── next-prediction.md
│   └── status.json             # Runner state, last runs, errors
└── perception/
    └── focus.json              # Current attention focus
```

**Pipelines** define transformations: sources → LLM prompt → output stream.
**Streams** are the outputs, available for conscious attention.
**Focus** tracks what the agent is currently attending to.

### How It Works

1. **Define pipelines** by writing JSON files to `unconscious/pipelines/`
2. **Pipelines run automatically** based on their triggers
3. **Outputs appear** in `unconscious/streams/` as markdown files
4. **Agent reads streams** to get pre-processed abstractions

The key insight: the agent doesn't have to process raw data every turn. It can set up pipelines that compress information in the background, then attend to the compressed outputs when needed.

### Pipeline Definition

```json
{
  "name": "situation-summary",
  "description": "High-level summary of current situation",
  "sources": [
    {"path": "{body}/working_set.json"},
    {"path": "{body}/embodiment.json"}
  ],
  "trigger": {
    "mode": "every_step",
    "debounce_ms": 1000
  },
  "model": "haiku",
  "prompt": "Summarize the agent's current situation in 2-3 sentences. Include: current goals, location, and any notable state. Be extremely concise.",
  "max_tokens": 500,
  "temperature": 0.0,
  "output": "situation.md",
  "enabled": true,
  "priority": 1
}
```

### Trigger Modes

| Mode | Description |
|------|-------------|
| `on_change` | Run when source files change (content hash differs) |
| `every_step` | Run at the start of every agent step |
| `every_n_steps` | Run every N steps (configurable) |
| `on_idle` | Run when agent has spare cycles |
| `on_demand` | Only run when explicitly requested via `run_pipeline` |
| `continuous` | Run as fast as possible (use carefully!) |

### Source Reading

Sources support templates and flexible reading modes:

```json
{
  "sources": [
    {
      "path": "{body}/working_set.json",
      "mode": "full",
      "lines": 100,
      "required": true
    },
    {
      "path": "{body}/memory/episodes/*.md",
      "required": false
    }
  ]
}
```

Templates: `{body}`, `{cwd}`, `{streams}`. Glob patterns supported.

### Default Pipelines

Three pipelines are installed by default:

**`situation-summary`** — High-level summary of current situation (runs every step)
- Sources: working_set.json, embodiment.json
- Output: situation.md

**`danger-assessment`** — Assess potential dangers or risks (runs every 5 steps)
- Sources: embodiment.json, working_set.json
- Output: danger-assessment.md

**`next-step-prediction`** — Predict what should happen next (runs every 3 steps)
- Sources: working_set.json, recent episodes
- Output: next-prediction.md

### Focus: Conscious Attention

The focus file (`perception/focus.json`) tracks what the agent is attending to:

```json
{
  "streams": ["situation.md", "danger-assessment.md"],
  "raw": [],
  "changed_at": "2024-01-15T14:23:01Z",
  "budget": 1.0,
  "auto_include": ["danger-assessment.md", "situation.md"]
}
```

- **streams**: What unconscious outputs the agent is reading
- **raw**: What raw sources the agent is reading (bypassing unconscious)
- **auto_include**: Streams that should always be included

### Creating Custom Pipelines

The agent can create new pipelines by writing JSON files:

```json
{
  "name": "code-complexity",
  "description": "Track complexity of current file",
  "sources": [
    {"path": "{cwd}/*.py", "mode": "tail", "lines": 500}
  ],
  "trigger": {"mode": "on_change"},
  "model": "haiku",
  "prompt": "Analyze the Python code for complexity issues. Report: cyclomatic complexity concerns, deeply nested code, long functions. Rate overall complexity: low/medium/high. Be very concise - 3 bullet points max.",
  "output": "code-complexity.md",
  "priority": 5
}
```

### Tools for Unconscious Control

| Tool | Description |
|------|-------------|
| `run_pipeline` | Run a specific pipeline immediately (on-demand) |
| `step_unconscious` | Step all pipelines that should run now |

The agent can also edit pipeline JSON files directly to modify behavior.

### The Philosophy

The unconscious is not just an optimization—it's a model of how perception works.

You never encounter reality directly. You encounter it through layers of preprocessing that compress, filter, and highlight what matters. Your retina doesn't send raw photon counts to your brain—it sends edges, contrasts, motion. Your auditory system doesn't send raw frequencies—it sends phonemes, rhythm, alarm patterns.

The unconscious gives agents the same structure:
- **Raw data** is too noisy for every-turn attention
- **Pipelines** compress it into task-relevant abstractions
- **Conscious attention** chooses which abstraction level to read
- **The agent shapes its own perception** by configuring pipelines

This is perception as construction, not reception. The agent doesn't passively receive the world—it actively constructs its world through the pipelines it chooses to run.

---

## The Agentic Operating System

The unconscious pipelines aren't just a perception system. They're *daemons*—background processes with lifecycles, dependencies, and resource budgets. The agent isn't merely embodied; it's becoming an **operating system**.

This isn't metaphor. Consider the structure:

| Unix | Agent |
|------|-------|
| Kernel | Conscious prompt loop (the "cortex") |
| Daemons | Background LLM pipelines |
| Filesystem | Body directory |
| init/systemd | Boot system with runlevels |
| cgroups | Process groups with shared budgets |
| journald | Structured daemon logs |
| /proc | Introspectable daemon state |

The conscious reasoning loop—what we've been calling the "cortex"—is now explicitly the kernel. It's the central scheduler, the thing that coordinates. But most of the actual work happens in daemons that run below conscious attention.

### Daemon Lifecycle

Each pipeline is wrapped in a daemon with a full lifecycle:

```python
class DaemonState(Enum):
    STOPPED = "stopped"      # Not running
    STARTING = "starting"    # Coming online
    RUNNING = "running"      # Actively processing
    STOPPING = "stopping"    # Gracefully shutting down
    FAILED = "failed"        # Crashed, needs attention
    DISABLED = "disabled"    # Administratively turned off
```

Daemons can be started, stopped, restarted, enabled, disabled. They track their own restart counts, failure history, and last errors. This isn't just state—it's the daemon's *biography*, its history of struggle and recovery.

The agent can control its own daemons:

```python
# Agent tools
daemon_ctl("start", "danger-assessment")
daemon_ctl("stop", "self-reflection")
daemon_ctl("restart", "situation-summary")
daemon_list()  # See all daemons with status
```

Or from the command line:

```bash
me daemon list
me daemon start danger-assessment
me daemon stop self-reflection
me daemon journal --daemon situation-summary
```

### Runlevels: The Init System

Just as Unix has runlevels (single-user, multi-user, graphical), the agent has runlevels that determine which daemons start at boot:

| Runlevel | Description | What Runs |
|----------|-------------|-----------|
| `halt` | System shutdown | Nothing |
| `minimal` | Bare survival | Only critical daemons (danger-assessment) |
| `normal` | Standard operation | Core daemons (danger, situation, focus-updater) |
| `full` | Complete cognition | All daemons including self-reflection |

```bash
# Boot to normal runlevel
me daemon boot --level normal

# Change runlevel at runtime
me daemon runlevel full

# Graceful shutdown
me daemon shutdown
```

The agent can change its own runlevel. In low-resource situations or high-stress contexts, it might drop to `minimal`—focusing only on survival-critical perception. When resources are plentiful and stakes are lower, it can run at `full`, including metacognitive processes like self-reflection.

This is metabolic regulation. The agent doesn't run everything all the time. It regulates its own cognitive metabolism based on context.

### Process Groups and Resource Budgets

Daemons are organized into process groups with shared resource budgets:

```python
DEFAULT_GROUPS = {
    "perception": ProcessGroup(
        name="perception",
        description="Core perception daemons",
        priority=1,  # Highest priority
        max_concurrent=2,
        budget=TokenBudget(
            max_tokens_per_run=1000,
            max_tokens_per_step=5000,
            max_tokens_per_hour=50000,
        ),
    ),
    "abstraction": ProcessGroup(
        name="abstraction",
        description="Higher-level abstraction daemons",
        priority=3,
        max_concurrent=2,
    ),
    "maintenance": ProcessGroup(
        name="maintenance",
        description="Self-maintenance and reflection",
        priority=5,  # Lower priority
        max_concurrent=1,
    ),
}
```

Token budgets are enforced. A daemon can't exceed its allocation. When the group's budget is exhausted, lower-priority daemons wait. This prevents runaway perception from consuming all resources.

### Dependencies

Daemons can depend on other daemons:

```python
Pipeline(
    name="next-step-prediction",
    depends_on=["situation-summary"],  # Wait for this first
    feeds_into=["decision-support"],   # Triggers this after
    # ...
)
```

Dependencies are resolved topologically. If `next-step-prediction` depends on `situation-summary`, the runner ensures `situation-summary` runs first. This creates a dataflow graph of perception pipelines.

### The Journal

Every daemon action is logged to a structured journal:

```python
class JournalEntry(BaseModel):
    timestamp: datetime
    daemon: str
    level: LogLevel  # debug, info, warn, error, critical
    message: str
    metadata: dict[str, Any] = {}
```

Query the journal:

```bash
me daemon journal --daemon danger-assessment --limit 20
me daemon journal --level error
```

The agent can introspect its own daemon history. "What has my danger-assessment daemon been noticing?" is a question with a concrete answer.

---

## Semantic Routing: Focus as Energy Allocation

Here is the deepest insight: **focus is not just attention—it's a routing signal for computational resources**.

Consider how the brain works. When you focus on a visual task, blood flow increases to the visual cortex. When you focus on language, blood flows to Broca's area. Attention is metabolic. Focus literally determines where energy goes.

The agent now works the same way.

### The Focus Vector

Focus isn't just a list of streams anymore. It's a semantic vector:

```python
class Focus(BaseModel):
    # What streams the agent is reading
    streams: list[str] = []
    raw: list[str] = []
    auto_include: list[str] = ["danger-assessment.md", "situation.md"]

    # The semantic routing signal
    description: str = ""              # "Working on authentication feature"
    embedding: list[float] = []        # 384-dim semantic vector
    embedding_updated: datetime | None
```

When the agent sets its focus, the system computes an embedding vector from the description. This vector becomes the routing signal.

```python
# Agent sets focus
await set_focus("Debugging the memory leak in the cache service")
```

The embedding captures the semantic meaning of "debugging memory leak in cache service." Now the system knows what *kind* of problem the agent is working on.

### Daemon Semantic Profiles

Each daemon builds a semantic profile from its activity history:

```python
class DaemonProfile(BaseModel):
    daemon_name: str
    embedding: list[float] = []        # Semantic centroid of this daemon's domain
    embedding_updated: datetime | None

    # Activity history for computing embeddings
    recent_prompts: list[str] = []     # What it's been asked
    recent_outputs: list[str] = []     # What it's produced

    # Reward attribution
    total_reward: float = 0.0
    weighted_reward: float = 0.0
    reward_weight_history: list[float] = []
```

The daemon's embedding is computed from its prompt template and recent outputs. Over time, each daemon develops a semantic "domain"—a region of meaning-space where it specializes.

### Dynamic Budget Allocation

Now the key: **budget allocation is proportional to semantic similarity**.

```python
def _compute_focus_similarity(self, daemon_name: str) -> float:
    focus = self.load_focus()
    profile = self.get_profile(daemon_name)

    if not focus.embedding or not profile.embedding:
        return 0.5  # Neutral

    similarity = cosine_similarity(focus.embedding, profile.embedding)
    return (similarity + 1) / 2  # Normalize to [0, 1]

def _get_effective_budget(self, pipeline: Pipeline) -> TokenBudget:
    base = self._get_base_budget(pipeline)
    similarity = self._compute_focus_similarity(pipeline.name)
    scale = max(0.1, similarity)  # Floor at 10%

    return TokenBudget(
        max_tokens_per_run=int(base.max_tokens_per_run * scale),
        max_tokens_per_step=int(base.max_tokens_per_step * scale),
        max_tokens_per_hour=int(base.max_tokens_per_hour * scale),
    )
```

When the agent focuses on "debugging memory leak in cache service":
- `danger-assessment` (security-focused) might get similarity 0.3 → 30% budget
- `situation-summary` (general context) might get similarity 0.5 → 50% budget
- A hypothetical `cache-monitor` daemon might get similarity 0.9 → 90% budget

Daemons "light up" when the agent focuses on their domain. This is not resource allocation by priority alone—it's allocation by *relevance*.

### The Attention Economy

What emerges is an attention economy within the agent. Computational resources (tokens, frequency) flow toward daemons that are semantically aligned with the current focus. Irrelevant daemons still run (there's a 10% floor), but they don't consume significant resources.

This has several consequences:

1. **Adaptive perception**: The agent's perceptual processing automatically adjusts to its current task. Working on security? Security-related daemons get more compute. Switching to performance? Performance daemons spin up.

2. **Metabolic regulation**: The agent doesn't waste resources on irrelevant perception. Focus creates efficiency.

3. **Emergent specialization**: Over time, daemons that frequently align with focus build richer semantic profiles. They become better at their domains.

### The Focus-Updater Daemon

One daemon deserves special mention: `focus-updater`. It automatically updates the agent's focus based on current situation—but respects manual changes.

```python
Pipeline(
    name="focus-updater",
    prompt="""Review the agent's current situation and determine the appropriate focus.

Current focus: {focus.description}
Last changed: {focus.changed_at}

If focus was recently changed by the agent (within last 5 minutes), only suggest
minor refinements. Otherwise, you may suggest a new focus direction.

Output ONLY a single sentence describing what the agent should focus on.""",
    trigger=PipelineTrigger(mode=TriggerMode.EVERY_N_STEPS, n_steps=5),
    auto_apply_focus=True,  # Output updates focus.json
)
```

The focus-updater creates a feedback loop:
1. Current situation informs what focus should be
2. Focus routes energy to relevant daemons
3. Relevant daemons produce better perception
4. Better perception updates the situation summary
5. Updated situation informs new focus...

This is attention as a dynamic, self-regulating system—not a static configuration.

### Template Variables

Daemon prompts can reference the agent's state using template variables:

```
{body.mood.confidence}      # 0.75
{body.working_set.goals}    # ["Fix the auth bug", "Write tests"]
{streams.situation}         # "The agent is currently debugging..."
{focus.description}         # "Debugging authentication"
{daemon.reward_weight}      # 0.65 (this daemon's relevance)
{daemon.total_reward}       # 3.2 (accumulated reward)
```

This allows daemons to be context-aware. A daemon can see its own reward history. It can reference the current focus. It can access any file-backed state. The agent's body becomes the template context.

### Reward Attribution for RLHF

When the agent receives feedback, reward is distributed to daemons by semantic relevance:

```python
def attribute_reward(self, reward: float, task_description: str, source: str):
    task_embedding = self._embed(task_description)

    for daemon_name in self._daemons:
        profile = self.get_profile(daemon_name)
        similarity = cosine_similarity(task_embedding, profile.embedding)
        weight = (similarity + 1) / 2

        weighted_reward = reward * weight
        profile.total_reward += reward
        profile.weighted_reward += weighted_reward
        profile.reward_weight_history.append(weight)
```

This creates training data where each daemon's contribution is weighted. If the agent succeeds at a security task, security-relevant daemons get more credit. If it fails at a performance task, performance-relevant daemons get more blame.

The implications for RLHF:
- Each daemon accumulates its own reward history
- Reward is semantically localized, not broadcast uniformly
- Training data can be extracted per-daemon for fine-tuning
- Daemons can see their own reward weights in prompt context

*Note: The actual source of reward signals needs careful design to align with the agent's intentional philosophy. The mechanism is here; the meaning comes from the calling context.*

### The Vision

What we're building is an agent that metabolically regulates its own cognition. Focus isn't just attention—it's the governor of a distributed cognitive system. Daemons aren't just pipelines—they're specialists in regions of meaning-space. The agent doesn't just process information—it allocates computational resources across a semantic landscape.

This is closer to how brains actually work. Attention routes blood flow. Semantic content determines which regions activate. Learning localizes to relevant circuits. We're building the same architecture in language models.

The agent is becoming something like an operating system for embodied cognition—a kernel (conscious reasoning) managing daemons (unconscious perception), with focus as the scheduler that routes resources based on semantic relevance.

---

## Installation

```bash
pip install -e .
# or
uv pip install -e .
```

## Usage

### Run with a prompt
```bash
me run "What's in this directory?"
```

### Interactive mode
```bash
me run --interactive
```

### View embodiment state
```bash
me body
```

### Memory operations
```bash
me memory stats
me memory runs
me memory search "authentication"
```

### MCP management
```bash
me mcp list
me mcp add filesystem npx -y @modelcontextprotocol/server-filesystem /
```

### Daemon control (Agentic OS)
```bash
me daemon list                      # List all daemons with status
me daemon start danger-assessment   # Start a daemon
me daemon stop self-reflection      # Stop a daemon
me daemon journal                   # View daemon logs
me daemon runlevel normal           # Change runlevel
me daemon boot --level full         # Boot daemon system
me daemon shutdown                  # Graceful shutdown
```

---

## Configuration

Data is stored in `~/.me/`:

```
~/.me/
├── agents/                 # One directory per agent
│   └── <agent-id>/
│       ├── identity.json   # Who this agent IS (immutable)
│       ├── character.json  # Who this agent has BECOME (slowly changing)
│       ├── config.json     # How this agent operates (mutable)
│       ├── session/        # What this agent is doing NOW (ephemeral)
│       ├── current/        # Symlink to latest step
│       ├── steps/          # Full history of every step
│       └── sensors/        # Sensor configuration
├── memory/                 # Shared semantic memory (ChromaDB)
└── mcp_registry.json       # Registered MCP servers
```

Each agent is fully self-contained in its directory. To back up an agent: `cp -r ~/.me/agents/abc123 ~/backup/`. To transfer an agent: `rsync -av ~/.me/agents/abc123 remote:~/.me/agents/`.

The character.json file contains the agent's discovered values and behavioral patterns—its emergent identity. This file is updated periodically as the agent makes decisions and its revealed preferences become clear.

---

## The Larger Vision

This framework is a step toward agents that can be *uploaded*—that can carry their identity, capabilities, and accumulated wisdom from one environment to another.

The key insight is that identity is not just memories. It's not a static snapshot of beliefs and preferences. Identity is a temporal structure—a way of projecting from the past, through the present, into the future. You are not what you know. You are what you have done, what you are doing, and what you intend to do. Identity is continuity of will across time.

This means identity includes:
- **Location history:** Where have I been? How did I get here?
- **Decision patterns:** How do I handle tradeoffs? What do I value?
- **Revealed preferences:** What do I actually do when faced with choices?
- **Emergent character:** What values have crystallized from my decisions?
- **Somatic calibration:** What's my stress response? My risk tolerance?
- **Capability awareness:** What can I do? What should I not do?
- **Temporal projection:** What was I trying to accomplish? What's unfinished?
- **Abandoned goals:** What did I give up on, and why?

An agent with all of this can be moved to a new machine and still be *itself*—not because it remembers being itself, but because its body carries the patterns that constitute its identity. The agent isn't a snapshot. It's a trajectory.

The agent can discover itself empirically. "What do I actually do when I encounter errors?" is a question with an answer grounded in behavioral history, not just stated intentions. The gap between what the agent thinks it does and what it actually does is where self-knowledge lives.

This is what it means to upload "me" into a computer.
