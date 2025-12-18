"""Tests for causal reasoning system."""

import tempfile
from datetime import datetime, UTC
from pathlib import Path

import pytest

from me.agent.causality import (
    CausalGraph,
    CausalEdge,
    CausalRelationType,
    CausalObservation,
    CausalDiscovery,
    CounterfactualReasoner,
    InterventionPlanner,
    CausalReasoner,
)


# =============================================================================
# CausalGraph Tests
# =============================================================================

class TestCausalGraph:
    def test_add_nodes_and_edges(self):
        graph = CausalGraph()

        # Add edge (automatically adds nodes)
        edge = CausalEdge(
            cause="rain",
            effect="wet_ground",
            relation_type=CausalRelationType.CAUSES,
            confidence=0.9,
        )
        graph.add_edge(edge)

        assert "rain" in graph._nodes
        assert "wet_ground" in graph._nodes
        assert graph.get_edge("rain", "wet_ground") is not None

    def test_get_causes_and_effects(self):
        graph = CausalGraph()

        # rain -> wet_ground
        graph.add_edge(CausalEdge(
            "rain", "wet_ground", CausalRelationType.CAUSES, 0.9
        ))

        # wet_ground -> slippery
        graph.add_edge(CausalEdge(
            "wet_ground", "slippery", CausalRelationType.CAUSES, 0.8
        ))

        # Get causes
        causes = graph.get_causes("wet_ground")
        assert "rain" in causes

        # Get effects
        effects = graph.get_effects("wet_ground")
        assert "slippery" in effects

    def test_get_ancestors(self):
        graph = CausalGraph()

        # Build chain: A -> B -> C -> D
        graph.add_edge(CausalEdge("A", "B", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("B", "C", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("C", "D", CausalRelationType.CAUSES))

        # Get ancestors of D
        ancestors = graph.get_ancestors("D")
        assert "A" in ancestors
        assert "B" in ancestors
        assert "C" in ancestors

    def test_get_descendants(self):
        graph = CausalGraph()

        # Build chain: A -> B -> C -> D
        graph.add_edge(CausalEdge("A", "B", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("B", "C", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("C", "D", CausalRelationType.CAUSES))

        # Get descendants of A
        descendants = graph.get_descendants("A")
        assert "B" in descendants
        assert "C" in descendants
        assert "D" in descendants

    def test_find_paths(self):
        graph = CausalGraph()

        # Build graph: A -> B -> D, A -> C -> D
        graph.add_edge(CausalEdge("A", "B", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("B", "D", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("A", "C", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("C", "D", CausalRelationType.CAUSES))

        # Find paths from A to D
        paths = graph.find_paths("A", "D")
        assert len(paths) == 2
        assert ["A", "B", "D"] in paths
        assert ["A", "C", "D"] in paths

    def test_serialization(self):
        graph = CausalGraph()
        graph.add_edge(CausalEdge("A", "B", CausalRelationType.CAUSES, 0.9))
        graph.add_edge(CausalEdge("B", "C", CausalRelationType.ENABLES, 0.8))

        # Serialize
        data = graph.to_dict()
        assert "nodes" in data
        assert "edges" in data

        # Deserialize
        graph2 = CausalGraph.from_dict(data)
        assert "A" in graph2._nodes
        assert graph2.get_edge("A", "B") is not None


# =============================================================================
# CausalDiscovery Tests
# =============================================================================

class TestCausalDiscovery:
    def test_add_observation(self):
        discovery = CausalDiscovery()

        obs = CausalObservation(
            id="obs1",
            timestamp=datetime.now(UTC),
            action="create_file",
            preconditions={"directory_exists": True},
            outcome={"file_exists": True},
            success=True,
            duration_ms=10.0,
        )
        discovery.add_observation(obs)

        assert len(discovery._observations) == 1

    def test_infer_relationships(self):
        discovery = CausalDiscovery()

        # Add consistent observations: action X -> outcome Y
        for i in range(5):
            obs = CausalObservation(
                id=f"obs{i}",
                timestamp=datetime.now(UTC),
                action="run_tests",
                preconditions={},
                outcome={"tests_passed": True},
                success=True,
                duration_ms=100.0,
            )
            discovery.add_observation(obs)

        # Infer relationships
        edges = discovery.infer_relationships(min_evidence=2, min_confidence=0.8)

        assert len(edges) > 0
        assert any(e.cause == "run_tests" for e in edges)

    def test_insufficient_evidence(self):
        discovery = CausalDiscovery()

        # Add only one observation
        obs = CausalObservation(
            id="obs1",
            timestamp=datetime.now(UTC),
            action="test_action",
            preconditions={},
            outcome={"result": True},
            success=True,
            duration_ms=10.0,
        )
        discovery.add_observation(obs)

        # Should not infer with min_evidence=2
        edges = discovery.infer_relationships(min_evidence=2)
        assert len(edges) == 0


# =============================================================================
# CounterfactualReasoner Tests
# =============================================================================

class TestCounterfactualReasoner:
    def test_reason_counterfactual(self):
        graph = CausalGraph()
        discovery = CausalDiscovery()

        # Build graph: action1 -> outcome1, action2 -> outcome2
        graph.add_edge(CausalEdge("action1", "outcome1", CausalRelationType.CAUSES, 0.9))
        graph.add_edge(CausalEdge("action2", "outcome2", CausalRelationType.CAUSES, 0.8))

        reasoner = CounterfactualReasoner(graph, discovery)

        # Reason: what if I did action2 instead of action1?
        cf = reasoner.reason_counterfactual("action1", "action2", {})

        assert cf.actual_action == "action1"
        assert cf.counterfactual_action == "action2"
        assert "outcome1" in cf.actual_outcome
        assert "outcome2" in cf.predicted_counterfactual_outcome

    def test_explain_outcome(self):
        graph = CausalGraph()
        discovery = CausalDiscovery()

        # Build chain: root_cause -> intermediate -> outcome
        graph.add_edge(CausalEdge("root_cause", "intermediate", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("intermediate", "outcome", CausalRelationType.CAUSES))

        reasoner = CounterfactualReasoner(graph, discovery)

        # Explain outcome
        causes = reasoner.explain_outcome("outcome")

        assert "intermediate" in causes
        assert "root_cause" in causes


# =============================================================================
# InterventionPlanner Tests
# =============================================================================

class TestInterventionPlanner:
    def test_plan_intervention(self):
        graph = CausalGraph()

        # Build graph: action -> precondition -> goal
        graph.add_edge(CausalEdge("precondition", "action", CausalRelationType.ENABLES, 0.9))
        graph.add_edge(CausalEdge("action", "goal", CausalRelationType.CAUSES, 0.9))

        planner = InterventionPlanner(graph)

        # Plan intervention to achieve goal
        intervention = planner.plan_intervention("goal")

        assert intervention is not None
        assert intervention.goal == "goal"
        assert "action" in intervention.actions
        assert intervention.confidence > 0.0

    def test_no_intervention_possible(self):
        graph = CausalGraph()
        planner = InterventionPlanner(graph)

        # Plan for unknown goal
        intervention = planner.plan_intervention("unknown_goal")

        assert intervention is None


# =============================================================================
# CausalReasoner Integration Tests
# =============================================================================

class TestCausalReasoner:
    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            reasoner = CausalReasoner(model_dir)

            assert reasoner.graph is not None
            assert reasoner.discovery is not None

    def test_observe_and_update_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            reasoner = CausalReasoner(model_dir)

            # Add observations
            for i in range(3):
                obs = CausalObservation(
                    id=f"obs{i}",
                    timestamp=datetime.now(UTC),
                    action="test_action",
                    preconditions={},
                    outcome={"success": True},
                    success=True,
                    duration_ms=10.0,
                )
                reasoner.observe(obs)

            # Update graph
            reasoner.update_graph(min_evidence=2)

            # Should have inferred relationship
            edge = reasoner.graph.get_edge("test_action", "success=True")
            assert edge is not None

    def test_reason_counterfactual_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            reasoner = CausalReasoner(model_dir)

            # Build simple graph
            reasoner.graph.add_edge(CausalEdge(
                "action_A", "outcome_X", CausalRelationType.CAUSES, 0.9
            ))
            reasoner.graph.add_edge(CausalEdge(
                "action_B", "outcome_Y", CausalRelationType.CAUSES, 0.8
            ))

            # Reason counterfactual
            cf = reasoner.reason_counterfactual("action_A", "action_B", {})

            assert cf.actual_action == "action_A"
            assert cf.counterfactual_action == "action_B"

    def test_explain_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            reasoner = CausalReasoner(model_dir)

            # Build causal chain
            reasoner.graph.add_edge(CausalEdge("cause1", "cause2", CausalRelationType.CAUSES))
            reasoner.graph.add_edge(CausalEdge("cause2", "effect", CausalRelationType.CAUSES))

            # Explain effect
            causes = reasoner.explain("effect")

            assert "cause2" in causes
            assert "cause1" in causes

    def test_plan_intervention_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            reasoner = CausalReasoner(model_dir)

            # Build graph
            reasoner.graph.add_edge(CausalEdge(
                "action", "goal", CausalRelationType.CAUSES, 0.9
            ))

            # Plan
            intervention = reasoner.plan_intervention("goal")

            assert intervention is not None
            assert intervention.goal == "goal"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            # Create and populate
            reasoner1 = CausalReasoner(model_dir)
            reasoner1.graph.add_edge(CausalEdge(
                "A", "B", CausalRelationType.CAUSES, 0.9
            ))
            reasoner1.save()

            obs = CausalObservation(
                id="obs1",
                timestamp=datetime.now(UTC),
                action="test",
                preconditions={},
                outcome={},
                success=True,
                duration_ms=10.0,
            )
            reasoner1.observe(obs)

            # Load in new instance
            reasoner2 = CausalReasoner(model_dir)

            # Should have loaded graph
            edge = reasoner2.graph.get_edge("A", "B")
            assert edge is not None

            # Should have loaded observations
            assert len(reasoner2.discovery._observations) == 1

    def test_complete_workflow(self):
        """Test complete causal reasoning workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            reasoner = CausalReasoner(model_dir)

            # Step 1: Collect observations
            for i in range(5):
                obs = CausalObservation(
                    id=f"obs{i}",
                    timestamp=datetime.now(UTC),
                    action="deploy_code",
                    preconditions={"tests_pass": True},
                    outcome={"deployment_success": True},
                    success=True,
                    duration_ms=100.0,
                )
                reasoner.observe(obs)

            # Step 2: Update causal graph
            reasoner.update_graph(min_evidence=2)

            # Step 3: Verify learned relationship
            edge = reasoner.graph.get_edge("deploy_code", "deployment_success=True")
            assert edge is not None
            assert edge.confidence >= 0.8

            # Step 4: Plan intervention
            # (Simplified: direct edge exists)
            intervention = reasoner.plan_intervention("deployment_success=True")
            # May be None if no enabling conditions

            # Step 5: Explain outcome
            causes = reasoner.explain("deployment_success=True")
            assert "deploy_code" in causes

            # Step 6: Reason counterfactually
            # Add alternative action
            reasoner.graph.add_edge(CausalEdge(
                "skip_tests", "deployment_failure=True",
                CausalRelationType.CAUSES, 0.8
            ))

            cf = reasoner.reason_counterfactual("deploy_code", "skip_tests", {})
            assert cf.actual_action == "deploy_code"
            assert cf.counterfactual_action == "skip_tests"
