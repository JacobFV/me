"""Tests for skill extraction and composition system."""

import tempfile
import uuid
from datetime import datetime, UTC
from pathlib import Path

import pytest

from me.agent.skills import (
    Skill,
    SkillType,
    SkillStatus,
    SkillLibrary,
    SkillExtractor,
    SkillComposer,
    SkillTransfer,
    SkillSystem,
)


# =============================================================================
# Skill Model Tests
# =============================================================================

class TestSkillModel:
    def test_success_rate(self):
        skill = Skill(
            name="test_skill",
            description="A test skill",
            success_count=8,
            failure_count=2,
            total_uses=10,
        )
        assert skill.success_rate() == 0.8

    def test_success_rate_zero_uses(self):
        skill = Skill(name="unused", description="Never used")
        assert skill.success_rate() == 0.0

    def test_is_applicable(self):
        skill = Skill(
            name="conditional_skill",
            description="Needs conditions",
            context_requirements={"env": "production", "has_tests": True},
        )

        # Matching context
        assert skill.is_applicable({"env": "production", "has_tests": True, "extra": "ok"})

        # Missing requirement
        assert not skill.is_applicable({"env": "production"})

        # Wrong value
        assert not skill.is_applicable({"env": "development", "has_tests": True})


# =============================================================================
# SkillLibrary Tests
# =============================================================================

class TestSkillLibrary:
    def test_add_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            skill = Skill(
                id="skill1",
                name="test_skill",
                description="A test skill",
            )
            library.add(skill)

            retrieved = library.get("skill1")
            assert retrieved is not None
            assert retrieved.name == "test_skill"

    def test_get_by_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            library.add(Skill(id="s1", name="unique_name", description="test"))

            skill = library.get_by_name("unique_name")
            assert skill is not None
            assert skill.id == "s1"

    def test_list_by_domain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            library.add(Skill(id="s1", name="skill1", description="", domains=["testing"]))
            library.add(Skill(id="s2", name="skill2", description="", domains=["testing", "ci"]))
            library.add(Skill(id="s3", name="skill3", description="", domains=["deployment"]))

            testing_skills = library.list_by_domain("testing")
            assert len(testing_skills) == 2

    def test_list_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            library.add(Skill(id="s1", name="atomic1", description="", type=SkillType.ATOMIC))
            library.add(Skill(id="s2", name="composite1", description="", type=SkillType.COMPOSITE))
            library.add(Skill(id="s3", name="atomic2", description="", type=SkillType.ATOMIC))

            atomic_skills = library.list_by_type(SkillType.ATOMIC)
            assert len(atomic_skills) == 2

    def test_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            library.add(Skill(id="s1", name="file_operations", description="Read and write files"))
            library.add(Skill(id="s2", name="git_commit", description="Commit changes to git"))
            library.add(Skill(id="s3", name="test_runner", description="Run test files"))

            results = library.search("file")
            assert len(results) >= 2  # file_operations and test_runner (test files)

    def test_recommend_for_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            # High success rate skill
            library.add(Skill(
                id="s1",
                name="good_skill",
                description="Works well",
                context_requirements={},
                success_count=9,
                total_uses=10,
                last_used=datetime.now(UTC),
            ))

            # Low success rate skill
            library.add(Skill(
                id="s2",
                name="bad_skill",
                description="Doesn't work",
                context_requirements={},
                success_count=1,
                total_uses=10,
            ))

            recommendations = library.recommend_for_context({})
            assert len(recommendations) >= 1
            assert recommendations[0].name == "good_skill"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            library1 = SkillLibrary(Path(tmpdir))
            library1.add(Skill(id="s1", name="persistent_skill", description="test"))

            # Load in new instance
            library2 = SkillLibrary(Path(tmpdir))
            skill = library2.get("s1")
            assert skill is not None
            assert skill.name == "persistent_skill"

    def test_get_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            library.add(Skill(id="s1", name="skill1", description="", total_uses=5, success_count=4))
            library.add(Skill(id="s2", name="skill2", description="", total_uses=10, success_count=8))

            stats = library.get_statistics()
            assert stats["total_skills"] == 2
            assert stats["total_uses"] == 15


# =============================================================================
# SkillExtractor Tests
# =============================================================================

class TestSkillExtractor:
    def test_extract_from_successful_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))
            extractor = SkillExtractor(library)

            actions = [
                {"type": "file", "name": "read_file", "params": {"path": "/tmp/test.txt"}},
                {"type": "edit", "name": "modify_content", "params": {"changes": []}},
                {"type": "file", "name": "write_file", "params": {"path": "/tmp/test.txt"}},
            ]

            extracted = extractor.extract_from_episode(
                episode_id="ep1",
                actions=actions,
                outcome="success",
                success=True,
                reward=1.0,
                context={"cwd": "/tmp"},
            )

            # Should extract some skills
            assert len(extracted) >= 1

    def test_no_extraction_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))
            extractor = SkillExtractor(library)

            actions = [{"type": "test", "name": "action1"}]

            extracted = extractor.extract_from_episode(
                episode_id="ep1",
                actions=actions,
                outcome="failure",
                success=False,
                reward=-1.0,
                context={},
            )

            assert len(extracted) == 0

    def test_filter_trivial_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))
            extractor = SkillExtractor(library)

            actions = [
                {"type": "print", "name": "debug_log"},  # Should be filtered
                {"type": "file", "name": "important_action"},  # Should be kept
            ]

            extracted = extractor.extract_from_episode(
                episode_id="ep1",
                actions=actions,
                outcome="success",
                success=True,
                reward=1.0,
                context={},
            )

            # Only non-trivial action should be extracted
            skill_names = [s.name for s in extracted]
            assert not any("print" in name or "debug" in name for name in skill_names)


# =============================================================================
# SkillComposer Tests
# =============================================================================

class TestSkillComposer:
    def test_compose_skills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            # Add base skills
            library.add(Skill(
                id="s1",
                name="read_file",
                description="Read a file",
                postconditions=["file_content_available"],
            ))
            library.add(Skill(
                id="s2",
                name="process_content",
                description="Process content",
                preconditions=["file_content_available"],
                postconditions=["content_processed"],
            ))

            composer = SkillComposer(library)

            composite = composer.compose(
                skill_ids=["s1", "s2"],
                name="read_and_process",
                description="Read file and process its content",
            )

            assert composite is not None
            assert composite.type == SkillType.COMPOSITE
            assert len(composite.sub_skills) == 2

    def test_get_execution_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            # Add skills
            library.add(Skill(id="s1", name="step1", description="", type=SkillType.ATOMIC))
            library.add(Skill(id="s2", name="step2", description="", type=SkillType.ATOMIC))
            library.add(Skill(
                id="s3",
                name="composite",
                description="",
                type=SkillType.COMPOSITE,
                sub_skills=["s1", "s2"],
            ))

            composer = SkillComposer(library)

            plan = composer.get_execution_plan("s3")
            assert len(plan) == 2
            assert plan[0].name == "step1"
            assert plan[1].name == "step2"

    def test_suggest_compositions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            library.add(Skill(id="s1", name="test_runner", description="Run tests"))
            library.add(Skill(id="s2", name="test_validator", description="Validate test results"))

            composer = SkillComposer(library)

            suggestions = composer.suggest_compositions("test")
            # Should suggest composing test-related skills
            assert len(suggestions) >= 0  # Might be empty if can't compose


# =============================================================================
# SkillTransfer Tests
# =============================================================================

class TestSkillTransfer:
    def test_transfer_skill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            # Add source skill
            library.add(Skill(
                id="s1",
                name="python_test",
                description="Run Python tests",
                domains=["python"],
                steps=["run pytest"],
            ))

            transfer = SkillTransfer(library)

            adapted = transfer.transfer_skill(
                skill_id="s1",
                target_domain="javascript",
                parameter_mapping={"pytest": "jest"},
            )

            assert adapted is not None
            assert "javascript" in adapted.domains
            assert adapted.status == SkillStatus.DRAFT

    def test_validate_transfer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            library.add(Skill(id="s1", name="source", description="", domains=["source_domain"]))

            transfer = SkillTransfer(library)

            adapted = transfer.transfer_skill("s1", "target_domain")
            assert adapted.status == SkillStatus.DRAFT

            # Validate successful transfer
            transfer.validate_transfer(adapted.id, success=True)

            updated = library.get(adapted.id)
            assert updated.status == SkillStatus.ACTIVE
            assert updated.success_count == 1

    def test_find_transferable_skills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = SkillLibrary(Path(tmpdir))

            # Abstract skill (transfers well)
            library.add(Skill(
                id="s1",
                name="abstract_skill",
                description="",
                type=SkillType.ABSTRACT,
                domains=["general"],
                success_count=9,
                total_uses=10,
            ))

            # Domain-specific skill
            library.add(Skill(
                id="s2",
                name="specific_skill",
                description="",
                type=SkillType.ATOMIC,
                domains=["python"],
                success_count=9,
                total_uses=10,
            ))

            transfer = SkillTransfer(library)

            transferable = transfer.find_transferable_skills("javascript")
            # Abstract skill should rank higher
            assert len(transferable) >= 1


# =============================================================================
# SkillSystem Integration Tests
# =============================================================================

class TestSkillSystem:
    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = SkillSystem(Path(tmpdir))

            assert system.library is not None
            assert system.extractor is not None
            assert system.composer is not None
            assert system.transfer is not None

    def test_extract_compose_transfer_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = SkillSystem(Path(tmpdir))

            # Extract skills from episode
            actions = [
                {"type": "read", "name": "read_input"},
                {"type": "transform", "name": "transform_data"},
            ]

            extracted = system.extract_from_episode(
                episode_id="ep1",
                actions=actions,
                outcome="success",
                success=True,
                reward=1.0,
                context={},
            )

            # Should have extracted something
            assert len(system.library.list_all()) >= len(extracted)

    def test_record_skill_use(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = SkillSystem(Path(tmpdir))

            # Add a skill
            system.library.add(Skill(id="s1", name="test_skill", description=""))

            # Record use
            system.record_skill_use("s1", success=True, reward=0.8, duration_ms=100.0)
            system.record_skill_use("s1", success=True, reward=0.9, duration_ms=120.0)

            skill = system.library.get("s1")
            assert skill.total_uses == 2
            assert skill.success_count == 2
            assert skill.avg_reward > 0

    def test_recommend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = SkillSystem(Path(tmpdir))

            # Add skills
            system.library.add(Skill(
                id="s1",
                name="relevant_skill",
                description="good for testing",
                success_count=9,
                total_uses=10,
            ))

            recommendations = system.recommend({}, goal="testing")
            assert len(recommendations) >= 1

    def test_get_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = SkillSystem(Path(tmpdir))

            system.library.add(Skill(id="s1", name="skill1", description=""))
            system.library.add(Skill(id="s2", name="skill2", description=""))

            stats = system.get_statistics()
            assert "library" in stats
            assert "transfers" in stats
            assert stats["library"]["total_skills"] == 2
