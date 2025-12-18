"""Agent core components."""

from me.agent.core import Agent, AgentConfig
from me.agent.body import (
    BodyDirectory,
    AgentIdentity,
    AgentConfig as BodyConfig,
    Embodiment,
    Mood,
    SomaticMode,
    Character,
    WorkingSet,
    MouthConfig,
    MouthMode,
    SensorsConfig,
    SensorDefinition,
    Episode,
    Procedure,
    SignificantMoment,
    Intention,
    Theory,
)
from me.agent.unconscious import (
    UnconsciousDirectory,
    UnconsciousRunner,
    Pipeline,
    PipelineTrigger,
    PipelineSource,
    TriggerMode,
    Focus,
    # Daemon system (Agentic OS)
    Daemon,
    DaemonState,
    Runlevel,
    ProcessGroup,
    TokenBudget,
    JournalEntry,
    LogLevel,
    # Semantic routing
    DaemonProfile,
    cosine_similarity,
    # Daemon Graph Templates (Higher-Order Abstraction)
    DaemonGraphTemplate,
    RewardSource,
    MetaDaemon,
    MetaDaemonCapability,
    # Default installations
    install_default_pipelines,
    install_default_templates,
    install_default_reward_sources,
    install_extended_groups,
    DEFAULT_TEMPLATES,
    DEFAULT_REWARD_SOURCES,
    EXTENDED_GROUPS,
)
from me.agent.memory import Memory
from me.agent.mcp import MCPRegistry
from me.agent.process import ProcessReader

__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    # Body
    "BodyDirectory",
    "AgentIdentity",
    "BodyConfig",
    "Embodiment",
    "Mood",
    "SomaticMode",
    "Character",
    "WorkingSet",
    "MouthConfig",
    "MouthMode",
    "SensorsConfig",
    "SensorDefinition",
    # Memory types
    "Episode",
    "Procedure",
    "SignificantMoment",
    "Intention",
    "Theory",
    # Unconscious
    "UnconsciousDirectory",
    "UnconsciousRunner",
    "Pipeline",
    "PipelineTrigger",
    "PipelineSource",
    "TriggerMode",
    "Focus",
    # Daemon system (Agentic OS)
    "Daemon",
    "DaemonState",
    "Runlevel",
    "ProcessGroup",
    "TokenBudget",
    "JournalEntry",
    "LogLevel",
    # Semantic routing
    "DaemonProfile",
    "cosine_similarity",
    # Daemon Graph Templates (Higher-Order Abstraction)
    "DaemonGraphTemplate",
    "RewardSource",
    "MetaDaemon",
    "MetaDaemonCapability",
    # Default installations
    "install_default_pipelines",
    "install_default_templates",
    "install_default_reward_sources",
    "install_extended_groups",
    "DEFAULT_TEMPLATES",
    "DEFAULT_REWARD_SOURCES",
    "EXTENDED_GROUPS",
    # Other components
    "Memory",
    "MCPRegistry",
    "ProcessReader",
]
