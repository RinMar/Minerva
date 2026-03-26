"""
Tests for the memory pipeline.
Used to ensure that facts are correctly ingested, stored, retrieved, and deleted
by the orchestrator and database layers.
"""
import unittest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.memory.context_store import delete_fact
from src.memory.graph_manager import add_triplets, expand_nodes
from src.memory.db import get_session, Base
from src.memory.db import EntityNode, GraphEdge
from src.memory.orchestrator import MemoryOrchestrator
from src.models.embeddings import embedding_model
import src.memory.db as db_module

from sqlalchemy.pool import StaticPool

# 1. First patch the database engine so we don't wipe the user's real memory.db!
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


class TestMemoryPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db_module._engine = test_engine
        db_module.SessionLocal = TestSessionLocal
        # Create tables in our fresh in-memory database
        Base.metadata.create_all(test_engine)

    def setUp(self):
        # Clear tables before each test
        with get_session() as session:
            session.query(EntityNode).delete()
            session.query(GraphEdge).delete()
            session.commit()

        self.user_name = "test_user"

    def test_add_triplets(self):
        """Test graph_manager.add_triplets successfully creates EntityNodes and GraphEdges."""
        triplets = [{"head": "User", "type": "loves", "tail": "pasta"}]
        add_triplets(triplets, "hobbies", ["food"], "The user loves pasta.", self.user_name, embedding_model)
        with get_session() as session:
            entities = session.query(EntityNode).filter(
                EntityNode.user_name == self.user_name
            ).all()
            self.assertEqual(len(entities), 2)
            names = [e.name for e in entities]
            self.assertIn("User", names)
            self.assertIn("pasta", names)

            edges = session.query(GraphEdge).all()
            self.assertEqual(len(edges), 1)
            self.assertEqual(edges[0].relation, "loves")

    def test_update_entity_text(self):
        """Test that adding a new triplet for the same entity appends to its summary."""
        triplets1 = [{"head": "User", "type": "loves", "tail": "pasta"}]
        add_triplets(triplets1, "hobbies", ["food"], "User loves pasta.", self.user_name, embedding_model)

        triplets2 = [{"head": "User", "type": "plays", "tail": "tennis"}]
        add_triplets(triplets2, "hobbies", ["sports"], "User plays tennis.", self.user_name, embedding_model)

        with get_session() as session:
            entities = session.query(EntityNode).filter(EntityNode.name == "User").all()
            self.assertEqual(len(entities), 1)
            self.assertIn("User loves pasta.", entities[0].text)
            self.assertIn("User plays tennis.", entities[0].text)

    def test_delete_fact(self):
        """Test fact deletion functionality."""
        triplets = [{"head": "OldIdea", "type": "is", "tail": "bad"}]
        add_triplets(triplets, "misc", ["test"], "To be deleted.", self.user_name, embedding_model)

        # Verify it went in
        with get_session() as session:
            count = session.query(EntityNode).filter(EntityNode.name == "OldIdea").count()
            self.assertEqual(count, 1)
            edges = session.query(GraphEdge).count()
            self.assertEqual(edges, 1)

        # Delete it via store
        delete_fact("To be deleted.", embedding_model, self.user_name)

        # Verify entity text is gone, and since it was the only text, the entity itself should drop
        with get_session() as session:
            count = session.query(EntityNode).filter(EntityNode.name == "OldIdea").count()
            self.assertEqual(count, 0)
            edges = session.query(GraphEdge).count()
            self.assertEqual(edges, 0)

    def test_update_fact_removes_old_edges(self):
        """Test that updating a fact removes old edges correctly without deleting the entity."""
        triplets_old = [{"head": "User", "type": "lives in", "tail": "Paris"}]
        add_triplets(triplets_old, "location", ["city"], "User lives in Paris.", self.user_name, embedding_model)

        # Add another unrelated fact to same entity to keep it alive
        triplets_keep = [{"head": "User", "type": "likes", "tail": "coffee"}]
        add_triplets(triplets_keep, "preference", ["food"], "User likes coffee.", self.user_name, embedding_model)

        with get_session() as session:
            self.assertEqual(session.query(GraphEdge).count(), 2)
            user_node = session.query(EntityNode).filter_by(name="User").first()
            self.assertIn("lives in Paris", user_node.text)

        # "Update" by deleting old fact text
        delete_fact("User lives in Paris", embedding_model, self.user_name)

        with get_session() as session:
            # The edge for Paris should be gone, coffee remains (1 edge left)
            self.assertEqual(session.query(GraphEdge).count(), 1)
            self.assertEqual(session.query(GraphEdge).first().relation, "likes")

            # The User node should still exist because coffee kept it alive
            user_node = session.query(EntityNode).filter_by(name="User").first()
            self.assertIsNotNone(user_node)
            self.assertNotIn("lives in Paris", user_node.text)
            self.assertIn("likes coffee", user_node.text)

    def test_graph_manager__add_and_expand_nodes(self):
        """Test that the graph edges are created and traversed successfully."""
        triplets = [
            {"head": "NodeA", "type": "rel1", "tail": "NodeB"},
            {"head": "NodeA", "type": "rel2", "tail": "NodeC"}
        ]
        add_triplets(triplets, "test", ["test"], "A to B and C", self.user_name, embedding_model)

        with get_session() as session:
            node_A = session.query(EntityNode).filter_by(name="NodeA").first()
            node_B = session.query(EntityNode).filter_by(name="NodeB").first()
            node_C = session.query(EntityNode).filter_by(name="NodeC").first()

            expanded = expand_nodes([node_A.id], self.user_name, depth=1)
            self.assertIn(node_A.id, expanded)
            self.assertIn(node_B.id, expanded)
            self.assertIn(node_C.id, expanded)

            # Ensure bidirectional expansion
            expanded_reverse = expand_nodes([node_B.id], self.user_name, depth=1)
            self.assertIn(node_A.id, expanded_reverse)

    @patch("src.memory.orchestrator.extract_triplets")
    def test_orchestrator_fact_graph(self, mock_extract):
        """
        Test the orchestrator pipeline directly inserting extracted triplets into the graph.
        """
        mock_extract.return_value = [{"head": "User", "type": "works at", "tail": "Microsoft"}]

        orchestrator = MemoryOrchestrator(
            llm=MagicMock(),
            emb_model=embedding_model,
            user_name=self.user_name
        )

        # Run a storage pipeline for a single mock new fact
        orchestrator._run_store_pipeline([{
            "action": "store",
            "topic": "career",
            "tags": ["job"],
            "fact": "User is a developer at Microsoft."
        }])

        # Verify entities and edges exist
        with get_session() as session:
            user_node = session.query(EntityNode).filter_by(name="User").first()
            ms_node = session.query(EntityNode).filter_by(name="Microsoft").first()

            self.assertIsNotNone(user_node)
            self.assertIsNotNone(ms_node)

            edges = session.query(GraphEdge).filter_by(source_id=user_node.id).all()
            targets = [e.target_id for e in edges]
            self.assertIn(ms_node.id, targets)
            self.assertEqual(edges[0].relation, "works at")

    def test_orchestrator_handles_string_triplets(self):
        """Reproduces the 'str has no attribute get' bug — string triplets must be normalized."""
        orchestrator = MemoryOrchestrator(
            llm=MagicMock(),
            emb_model=embedding_model,
            user_name=self.user_name
        )

        # Run a storage pipeline with string-formatted triplets
        orchestrator._run_store_pipeline([{
            "action": "store",
            "topic": "intro",
            "tags": ["test"],
            "fact": "Tom is a computer scientist creating me",
            "triplets": ["Tom - is a - Computer Scientist", "Tom - creating - Me"]
        }])

        # Verify entities and edges exist
        with get_session() as session:
            count = session.query(EntityNode).filter_by(user_name=self.user_name).count()
            self.assertGreater(count, 0)

            # Check for specific edge
            edge = session.query(GraphEdge).filter_by(relation="creating").first()
            self.assertIsNotNone(edge)
            self.assertEqual(edge.source_name, "Tom")
            self.assertEqual(edge.target_name, "Me")


if __name__ == "__main__":
    unittest.main()
