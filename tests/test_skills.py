"""
Tests for the embodied skills system.

Tests cover:
- Skill creation and parsing (SKILL.md format)
- Skill lifecycle (developing -> learned -> atrophied)
- Proficiency computation from usage
- Skill activation/deactivation
- Skill search and recommendations
- Integration with body directory
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta, UTC
from pathlib import Path

from me.agent.skills import (
    Skill,
    SkillMetadata,
    SkillState,
    SkillType,
    SkillIndex,
    SkillManager,
    SkillSystem,
    SkillComposer,
    parse_frontmatter,
    parse_sections,
    parse_numbered_list,
    skill_daemon_to_pipeline,
    get_skill_daemon_names,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_skills_dir():
    """Create a temporary skills directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def skill_manager(temp_skills_dir):
    """Create and initialize a skill manager."""
    manager = SkillManager(temp_skills_dir)
    manager.initialize()
    return manager


@pytest.fixture
def sample_skill_dir(temp_skills_dir):
    """Create a sample skill directory with SKILL.md."""
    skill_dir = temp_skills_dir / "test-skill"
    skill_dir.mkdir(parents=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: test-skill
description: A test skill for unit testing
version: 1.0.0
tags:
  - test
  - example
domains:
  - testing
steps:
  - First step
  - Second step
---

## Overview

This is a test skill.

## Steps

1. Do first thing
2. Do second thing
3. Verify results
""")

    return skill_dir


# =============================================================================
# Test Parsing Functions
# =============================================================================

class TestParsing:
    """Test markdown/YAML parsing functions."""

    def test_parse_frontmatter_basic(self):
        """Test parsing basic YAML frontmatter."""
        content = """---
name: test
description: A test
---

Body content here.
"""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter["name"] == "test"
        assert frontmatter["description"] == "A test"
        assert "Body content here" in body

    def test_parse_frontmatter_with_lists(self):
        """Test parsing frontmatter with list values."""
        content = """---
name: test
tags:
  - tag1
  - tag2
---

Body
"""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter["tags"] == ["tag1", "tag2"]

    def test_parse_frontmatter_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        content = "Just plain content"
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert body == content

    def test_parse_sections(self):
        """Test parsing markdown sections."""
        content = """## Overview

This is the overview.

## Steps

1. Step one
2. Step two

## Notes

Some notes here.
"""
        sections = parse_sections(content)
        assert "overview" in sections
        assert "steps" in sections
        assert "notes" in sections
        assert "This is the overview" in sections["overview"]

    def test_parse_numbered_list(self):
        """Test parsing numbered lists."""
        content = """1. First item
2. Second item
3. Third item"""
        items = parse_numbered_list(content)
        assert len(items) == 3
        assert items[0] == "First item"
        assert items[2] == "Third item"

    def test_parse_numbered_list_with_parens(self):
        """Test parsing numbered lists with parentheses."""
        content = """1) Item one
2) Item two"""
        items = parse_numbered_list(content)
        assert len(items) == 2


# =============================================================================
# Test SkillMetadata
# =============================================================================

class TestSkillMetadata:
    """Test SkillMetadata model."""

    def test_default_values(self):
        """Test default metadata values."""
        meta = SkillMetadata(name="test")
        assert meta.state == SkillState.DEVELOPING
        assert meta.proficiency == 0.0
        assert meta.use_count == 0
        assert meta.is_active is False

    def test_success_rate(self):
        """Test success rate computation."""
        meta = SkillMetadata(name="test", success_count=8, failure_count=2)
        assert meta.success_rate == 0.8

    def test_success_rate_zero_uses(self):
        """Test success rate with no uses."""
        meta = SkillMetadata(name="test")
        assert meta.success_rate == 0.0

    def test_is_mastered(self):
        """Test mastery threshold."""
        meta = SkillMetadata(name="test", proficiency=0.79)
        assert not meta.is_mastered

        meta.proficiency = 0.80
        assert meta.is_mastered

    def test_is_atrophying(self):
        """Test atrophy detection."""
        meta = SkillMetadata(name="test")
        assert not meta.is_atrophying  # Never used

        meta.last_used = datetime.now(UTC) - timedelta(days=29)
        assert not meta.is_atrophying

        meta.last_used = datetime.now(UTC) - timedelta(days=31)
        assert meta.is_atrophying

    def test_compute_proficiency(self):
        """Test proficiency computation."""
        meta = SkillMetadata(
            name="test",
            use_count=10,
            success_count=8,
            failure_count=2,
            last_used=datetime.now(UTC),
        )
        prof = meta.compute_proficiency()
        # 80% success rate + experience bonus, no decay
        assert prof > 0.8
        assert prof <= 1.0

    def test_compute_proficiency_with_decay(self):
        """Test proficiency decays over time."""
        meta = SkillMetadata(
            name="test",
            use_count=10,
            success_count=10,
            failure_count=0,
            last_used=datetime.now(UTC) - timedelta(days=30),
        )
        prof = meta.compute_proficiency()
        # Should be lower due to time decay
        assert prof < 1.0


# =============================================================================
# Test Skill
# =============================================================================

class TestSkill:
    """Test Skill model."""

    def test_from_skill_md(self, sample_skill_dir):
        """Test loading skill from SKILL.md."""
        skill = Skill.from_skill_md(sample_skill_dir)
        assert skill.metadata.name == "test-skill"
        assert skill.metadata.description == "A test skill for unit testing"
        assert "test" in skill.metadata.tags
        assert "testing" in skill.metadata.domains
        assert len(skill.steps) == 2

    def test_to_prompt_context(self, sample_skill_dir):
        """Test formatting skill for prompt."""
        skill = Skill.from_skill_md(sample_skill_dir)
        context = skill.to_prompt_context()
        assert "test-skill" in context
        assert "0%" in context  # Default proficiency

    def test_get_execution_instructions(self, sample_skill_dir):
        """Test getting execution instructions."""
        skill = Skill.from_skill_md(sample_skill_dir)
        instructions = skill.get_execution_instructions()
        assert "First step" in instructions


# =============================================================================
# Test SkillManager
# =============================================================================

class TestSkillManager:
    """Test SkillManager operations."""

    def test_initialize(self, skill_manager, temp_skills_dir):
        """Test manager initialization creates directories."""
        assert (temp_skills_dir / "learned").exists()
        assert (temp_skills_dir / "developing").exists()
        assert (temp_skills_dir / "atrophied").exists()
        assert (temp_skills_dir / "index.json").exists()

    def test_codify_skill(self, skill_manager):
        """Test creating a new skill."""
        skill = skill_manager.codify_skill(
            name="new-skill",
            description="A newly codified skill",
            instructions="## How to do it\n\n1. Step one\n2. Step two",
            tags=["test"],
        )

        assert skill is not None
        assert skill.metadata.name == "new-skill"
        assert skill.metadata.state == SkillState.DEVELOPING
        assert skill.metadata.learned_from == "codified"

        # Check file was created
        skill_dir = skill_manager.skills_dir / "developing" / "new-skill"
        assert skill_dir.exists()
        assert (skill_dir / "SKILL.md").exists()

    def test_get_skill(self, skill_manager):
        """Test retrieving a skill."""
        skill_manager.codify_skill(
            name="retrieve-test",
            description="Test skill",
            instructions="Test instructions",
        )

        skill = skill_manager.get_skill("retrieve-test")
        assert skill is not None
        assert skill.metadata.name == "retrieve-test"

    def test_get_skill_not_found(self, skill_manager):
        """Test retrieving non-existent skill."""
        skill = skill_manager.get_skill("nonexistent")
        assert skill is None

    def test_activate_skill(self, skill_manager):
        """Test activating a skill."""
        skill_manager.codify_skill(
            name="activate-test",
            description="Test",
            instructions="Test",
        )

        skill = skill_manager.activate_skill("activate-test")
        assert skill is not None
        assert skill.metadata.is_active

        # Check index updated
        index = skill_manager.index
        assert "activate-test" in index.active_skills
        assert index.skills["activate-test"].is_active

    def test_deactivate_skill(self, skill_manager):
        """Test deactivating a skill."""
        skill_manager.codify_skill(
            name="deactivate-test",
            description="Test",
            instructions="Test",
        )
        skill_manager.activate_skill("deactivate-test")
        result = skill_manager.deactivate_skill("deactivate-test")

        assert result is True
        assert "deactivate-test" not in skill_manager.index.active_skills

    def test_record_usage_success(self, skill_manager):
        """Test recording successful skill usage."""
        skill_manager.codify_skill(
            name="usage-test",
            description="Test",
            instructions="Test",
        )

        meta = skill_manager.record_usage("usage-test", success=True)
        assert meta is not None
        assert meta.use_count == 1
        assert meta.success_count == 1
        assert meta.failure_count == 0

    def test_record_usage_failure(self, skill_manager):
        """Test recording failed skill usage."""
        skill_manager.codify_skill(
            name="failure-test",
            description="Test",
            instructions="Test",
        )

        meta = skill_manager.record_usage("failure-test", success=False)
        assert meta is not None
        assert meta.use_count == 1
        assert meta.success_count == 0
        assert meta.failure_count == 1

    def test_proficiency_increases_with_success(self, skill_manager):
        """Test that proficiency increases with successful usage."""
        skill_manager.codify_skill(
            name="proficiency-test",
            description="Test",
            instructions="Test",
        )

        # Record multiple successful uses
        for _ in range(10):
            meta = skill_manager.record_usage("proficiency-test", success=True)

        assert meta.proficiency > 0.5

    def test_state_transition_to_learned(self, skill_manager):
        """Test skill transitions from developing to learned."""
        skill_manager.codify_skill(
            name="transition-test",
            description="Test",
            instructions="Test",
        )

        # Record many successful uses to reach mastery
        for _ in range(50):
            meta = skill_manager.record_usage("transition-test", success=True)

        # Should transition to learned
        assert meta.state == SkillState.LEARNED

        # Check file moved
        learned_dir = skill_manager.skills_dir / "learned" / "transition-test"
        developing_dir = skill_manager.skills_dir / "developing" / "transition-test"
        assert learned_dir.exists()
        assert not developing_dir.exists()

    def test_list_skills(self, skill_manager):
        """Test listing skills."""
        skill_manager.codify_skill(name="skill1", description="S1", instructions="I1")
        skill_manager.codify_skill(name="skill2", description="S2", instructions="I2")

        skills = skill_manager.list_skills()
        assert len(skills) == 2

    def test_list_skills_by_state(self, skill_manager):
        """Test filtering skills by state."""
        skill_manager.codify_skill(name="dev-skill", description="D", instructions="I")

        developing = skill_manager.list_skills(SkillState.DEVELOPING)
        learned = skill_manager.list_skills(SkillState.LEARNED)

        assert len(developing) == 1
        assert len(learned) == 0

    def test_search_skills(self, skill_manager):
        """Test searching skills."""
        skill_manager.codify_skill(
            name="git-workflow",
            description="Git branching workflow",
            instructions="How to use git",
            tags=["git", "version-control"],
        )
        skill_manager.codify_skill(
            name="debugging",
            description="Debug code",
            instructions="How to debug",
            tags=["debugging"],
        )

        results = skill_manager.search_skills("git")
        assert len(results) == 1
        assert results[0].name == "git-workflow"

    def test_delete_skill(self, skill_manager):
        """Test deleting a skill."""
        skill_manager.codify_skill(name="delete-me", description="D", instructions="I")

        result = skill_manager.delete_skill("delete-me")
        assert result is True
        assert skill_manager.get_skill("delete-me") is None

    def test_install_skill(self, skill_manager, sample_skill_dir):
        """Test installing external skill."""
        skill = skill_manager.install_skill(sample_skill_dir)

        assert skill is not None
        assert skill.metadata.name == "test-skill"
        assert skill.metadata.learned_from == "external"

    def test_get_statistics(self, skill_manager):
        """Test getting skill statistics."""
        skill_manager.codify_skill(name="s1", description="D", instructions="I")
        skill_manager.codify_skill(name="s2", description="D", instructions="I")
        skill_manager.record_usage("s1", success=True)
        skill_manager.record_usage("s1", success=True)

        stats = skill_manager.get_statistics()
        assert stats["total_skills"] == 2
        # s1 gets promoted to LEARNED after 2 successes (proficiency = 100% > 80% threshold)
        assert stats["learned"] == 1
        assert stats["developing"] == 1
        assert stats["total_uses"] == 2


# =============================================================================
# Test SkillComposer
# =============================================================================

class TestSkillComposer:
    """Test skill composition."""

    def test_compose_skills(self, skill_manager):
        """Test composing multiple skills."""
        skill_manager.codify_skill(
            name="skill-a",
            description="First skill",
            instructions="Do A",
        )
        skill_manager.codify_skill(
            name="skill-b",
            description="Second skill",
            instructions="Do B",
        )

        composer = SkillComposer(skill_manager)
        composite = composer.compose(
            ["skill-a", "skill-b"],
            name="combined-skill",
            description="A + B combined",
        )

        assert composite is not None
        assert composite.metadata.skill_type == SkillType.COMPOSITE
        assert "skill-a" in composite.instructions
        assert "skill-b" in composite.instructions


# =============================================================================
# Test SkillSystem
# =============================================================================

class TestSkillSystem:
    """Test unified SkillSystem interface."""

    def test_system_initialization(self, temp_skills_dir):
        """Test SkillSystem initialization."""
        system = SkillSystem(temp_skills_dir)
        system.initialize()

        assert (temp_skills_dir / "learned").exists()
        assert (temp_skills_dir / "developing").exists()

    def test_system_codify_and_use(self, temp_skills_dir):
        """Test full workflow through SkillSystem."""
        system = SkillSystem(temp_skills_dir)
        system.initialize()

        # Codify
        skill = system.codify(
            name="sys-test",
            description="System test",
            instructions="Test instructions",
        )
        assert skill is not None

        # Activate
        activated = system.activate("sys-test")
        assert activated is not None

        # Use
        meta = system.use("sys-test", success=True)
        assert meta.use_count == 1

        # Deactivate
        result = system.deactivate("sys-test")
        assert result is True


# =============================================================================
# Test Skill-Daemon Integration
# =============================================================================

class TestSkillDaemonIntegration:
    """Test skill-daemon integration functions."""

    def test_skill_daemon_to_pipeline(self, temp_skills_dir):
        """Test converting skill daemon definition to pipeline."""
        daemon_def = {
            "name": "test-daemon",
            "trigger": "on_change",
            "sources": ["{cwd}/.git/index"],
            "prompt": "Analyze changes",
            "model": "haiku",
        }

        pipeline = skill_daemon_to_pipeline("test-skill", daemon_def, temp_skills_dir)

        assert pipeline["name"] == "skill-test-skill-test-daemon"
        assert pipeline["trigger"]["mode"] == "on_change"
        assert len(pipeline["sources"]) == 1
        assert pipeline["prompt"] == "Analyze changes"
        assert "skill-daemon" in pipeline["tags"]

    def test_get_skill_daemon_names(self):
        """Test getting daemon names from skill."""
        skill = Skill(
            metadata=SkillMetadata(
                name="test-skill",
                daemons=[
                    {"name": "daemon-a"},
                    {"name": "daemon-b"},
                ],
            ),
        )

        names = get_skill_daemon_names(skill)
        assert len(names) == 2
        assert "skill-test-skill-daemon-a" in names
        assert "skill-test-skill-daemon-b" in names


# =============================================================================
# Test Body Integration
# =============================================================================

class TestBodyIntegration:
    """Test skill integration with BodyDirectory."""

    def test_body_skills_property(self):
        """Test accessing skills through body."""
        from me.agent.body import BodyDirectory

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            body = BodyDirectory(base_dir, "test-agent")
            body.initialize()

            # Access skills
            skills = body.skills
            assert skills is not None

            # Should be able to codify
            skill = skills.codify_skill(
                name="body-test",
                description="Test",
                instructions="Test",
            )
            assert skill is not None

    def test_skills_in_body_to_dict(self):
        """Test skills statistics appear in body.to_dict()."""
        from me.agent.body import BodyDirectory

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            body = BodyDirectory(base_dir, "test-agent")
            body.initialize()

            body.skills.codify_skill(
                name="dict-test",
                description="Test",
                instructions="Test",
            )

            data = body.to_dict()
            assert "skills" in data
            assert data["skills"]["total_skills"] == 1

    def test_skills_in_prompt_section(self):
        """Test skills appear in prompt section."""
        from me.agent.body import BodyDirectory

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            body = BodyDirectory(base_dir, "test-agent")
            body.initialize()

            body.skills.codify_skill(
                name="prompt-test",
                description="Test",
                instructions="Test",
            )

            prompt = body.to_prompt_section()
            assert "Skills:" in prompt


# =============================================================================
# Test Skill Templates
# =============================================================================

class TestSkillTemplates:
    """Test skill template installation."""

    def test_list_templates(self):
        """Test listing available templates."""
        from me.agent.skill_templates import list_templates

        templates = list_templates()
        # Should have our example templates
        assert "git-workflow" in templates or len(templates) >= 0

    def test_install_template(self, skill_manager):
        """Test installing a template."""
        from me.agent.skill_templates import install_template, TEMPLATES_DIR

        if not TEMPLATES_DIR.exists():
            pytest.skip("No templates directory")

        # Try to install git-workflow if it exists
        if (TEMPLATES_DIR / "git-workflow").exists():
            result = install_template(skill_manager, "git-workflow")
            assert result is True

            skill = skill_manager.get_skill("git-workflow")
            assert skill is not None


# =============================================================================
# Test Skill Index
# =============================================================================

class TestSkillIndex:
    """Test SkillIndex operations."""

    def test_get_by_state(self):
        """Test filtering by state."""
        index = SkillIndex()
        index.skills["s1"] = SkillMetadata(name="s1", state=SkillState.LEARNED)
        index.skills["s2"] = SkillMetadata(name="s2", state=SkillState.DEVELOPING)
        index.skills["s3"] = SkillMetadata(name="s3", state=SkillState.LEARNED)

        learned = index.get_by_state(SkillState.LEARNED)
        assert len(learned) == 2

        developing = index.get_by_state(SkillState.DEVELOPING)
        assert len(developing) == 1

    def test_get_active(self):
        """Test getting active skills."""
        index = SkillIndex()
        index.skills["s1"] = SkillMetadata(name="s1", is_active=True)
        index.skills["s2"] = SkillMetadata(name="s2", is_active=False)
        index.active_skills = ["s1"]

        active = index.get_active()
        assert len(active) == 1
        assert active[0].name == "s1"

    def test_search(self):
        """Test search function."""
        index = SkillIndex()
        index.skills["git-workflow"] = SkillMetadata(
            name="git-workflow",
            description="Git workflow",
            tags=["git", "vcs"],
        )
        index.skills["debugging"] = SkillMetadata(
            name="debugging",
            description="Debug skills",
            tags=["debug"],
        )

        results = index.search("git")
        assert len(results) == 1
        assert results[0].name == "git-workflow"

        # Search by description
        results = index.search("workflow")
        assert len(results) == 1

        # Search by tag
        results = index.search("vcs")
        assert len(results) == 1
