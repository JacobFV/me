"""
Default skill templates for bootstrapping agents.

These are example skills that demonstrate the SKILL.md format.
They can be installed into an agent's body to give it initial capabilities.

Philosophy note:
Skills are meant to be learned and evolved by the agent through experience.
These templates are just starting points - the agent should adapt them,
create new skills, and let unused ones atrophy naturally.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from me.agent.skills import SkillManager


TEMPLATES_DIR = Path(__file__).parent / "templates"


def list_templates() -> list[str]:
    """List available skill templates."""
    if not TEMPLATES_DIR.exists():
        return []
    return [d.name for d in TEMPLATES_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]


def install_template(manager: "SkillManager", template_name: str) -> bool:
    """Install a skill template into an agent's body."""
    template_dir = TEMPLATES_DIR / template_name
    if not template_dir.exists() or not (template_dir / "SKILL.md").exists():
        return False

    skill = manager.install_skill(template_dir)
    return skill is not None


def install_all_templates(manager: "SkillManager") -> list[str]:
    """Install all skill templates into an agent's body."""
    installed = []
    for template_name in list_templates():
        if install_template(manager, template_name):
            installed.append(template_name)
    return installed
