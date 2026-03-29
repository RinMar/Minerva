"""
Tests for the memory pipeline.
Used to ensure that facts are correctly ingested, stored, retrieved, and deleted
by the orchestrator and database layers.
"""
import unittest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.memory.store as memory_store
import src.memory.retrieve as memory_retrieve

from src.memory.db import get_session, Base
from src.memory.db import EntityNode, GraphEdge
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

        self.user_id = 1
        self.user_name = "test_user"

        # Patch embedding model globally for this test class
        self.patcher_emb = patch('src.models.embeddings.get_embedding_model')

        self.mock_emb_func = self.patcher_emb.start()

        self.mock_emb = MagicMock()
        # Return highly distinct vectors, using hash of the text so cosine similarity works properly
        import hashlib
        self.mock_emb.encode.side_effect = lambda text, **kwargs: MagicMock(
            tolist=lambda: [(int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000) / 1000.0] * 384
        )
        self.mock_emb_func.return_value = self.mock_emb

    def tearDown(self):
        self.patcher_emb.stop()

    def test_store_triplets(self):
        """Test store_triplets successfully creates EntityNodes and GraphEdges."""
        triplets = [{"head": "User", "type": "loves", "tail": "pasta"}]
        memory_store.store_triplets(triplets, "hobbies", ["food"], "The user loves pasta.", self.user_id, self.mock_emb)
        with get_session() as session:
            entities = session.query(EntityNode).filter(
                EntityNode.user_id == self.user_id
            ).all()
            self.assertEqual(len(entities), 2)
            names = [e.name for e in entities]
            self.assertIn("User", names)
            self.assertIn("pasta", names)

            edges = session.query(GraphEdge).all()
            self.assertEqual(len(edges), 1)
            self.assertEqual(edges[0].relation, "loves")

    def test_add_entity(self):
        """Test explicit add_entity successfully creates/modifies EntityNodes."""
        memory_store.add_entity("User", "profile", "The user loves pasta.", ["food"], self.user_id, self.mock_emb)
        with get_session() as session:
            entities = session.query(EntityNode).filter(EntityNode.name == "User").all()
            self.assertEqual(len(entities), 1)
            self.assertIn("The user loves pasta.", entities[0].text)

    def test_update_entity_text(self):
        """Test that adding a new triplet for the same entity appends to its summary."""
        triplets1 = [{"head": "User", "type": "loves", "tail": "pasta"}]
        memory_store.store_triplets(triplets1, "hobbies", ["food"], "User loves pasta.", self.user_id, self.mock_emb)

        triplets2 = [{"head": "User", "type": "plays", "tail": "tennis"}]
        memory_store.store_triplets(triplets2, "hobbies", ["sports"], "User plays tennis.", self.user_id, self.mock_emb)

        with get_session() as session:
            entities = session.query(EntityNode).filter(EntityNode.name == "User").all()
            self.assertEqual(len(entities), 1)
            self.assertIn("User loves pasta.", entities[0].text)
            self.assertIn("User plays tennis.", entities[0].text)

    def test_expand_nodes(self):
        """Test that the graph edges are created and traversed successfully."""
        triplets = [
            {"head": "NodeA", "type": "rel1", "tail": "NodeB"},
            {"head": "NodeA", "type": "rel2", "tail": "NodeC"}
        ]
        memory_store.store_triplets(triplets, "test", ["test"], "A to B and C", self.user_id, self.mock_emb)

        with get_session() as session:
            node_A = session.query(EntityNode).filter_by(name="NodeA").first()
            node_B = session.query(EntityNode).filter_by(name="NodeB").first()
            node_C = session.query(EntityNode).filter_by(name="NodeC").first()

            expanded = memory_retrieve.expand_nodes([node_A.id], self.user_id, depth=1)
            self.assertIn(node_A.id, expanded)
            self.assertIn(node_B.id, expanded)
            self.assertIn(node_C.id, expanded)

            # Ensure bidirectional expansion
            expanded_reverse = memory_retrieve.expand_nodes([node_B.id], self.user_id, depth=1)
            self.assertIn(node_A.id, expanded_reverse)

    def test_delete_edge(self):
        """Test explicit delete_edge."""
        triplets = [{"head": "NodeA", "type": "rel1", "tail": "NodeB"}]
        memory_store.store_triplets(triplets, "test", ["test"], "A to B", self.user_id, self.mock_emb)
        with get_session() as session:
            self.assertEqual(session.query(GraphEdge).count(), 1)
        memory_store.delete_edge("NodeA", "NodeB", "rel1", self.user_id)
        with get_session() as session:
            self.assertEqual(session.query(GraphEdge).count(), 0)
            self.assertEqual(session.query(EntityNode).filter_by(user_id=self.user_id).count(), 2)

    def test_update_edge(self):
        """Test explicit update_edge."""
        triplets = [{"head": "NodeA", "type": "rel1", "tail": "NodeB"}]
        memory_store.store_triplets(triplets, "test", ["test"], "A to B", self.user_id, self.mock_emb)
        memory_store.update_edge("NodeA", "NodeB", "rel1", "rel2", self.mock_emb, self.user_id)
        with get_session() as session:
            edge = session.query(GraphEdge).first()
            self.assertEqual(edge.relation, "rel2")

    def test_delete_entity(self):
        """Test explicit delete_entity and cascading edges."""
        triplets = [{"head": "NodeA", "type": "rel1", "tail": "NodeB"}]
        memory_store.store_triplets(triplets, "test", ["test"], "A to B", self.user_id, self.mock_emb)
        memory_store.delete_entity("NodeA", self.user_id)
        with get_session() as session:
            self.assertEqual(session.query(EntityNode).filter_by(name="NodeA").count(), 0)
            self.assertEqual(session.query(GraphEdge).count(), 0)

    def test_update_entity(self):
        """Test explicit update_entity renaming and text replacement."""
        triplets = [{"head": "OldName", "type": "rel1", "tail": "NodeB"}]
        memory_store.store_triplets(triplets, "test", ["test"], "OldName is related to NodeB",
                                   self.user_id, self.mock_emb)
        memory_store.update_entity("OldName", "NewName", None, self.mock_emb, self.user_id)
        with get_session() as session:
            ent = session.query(EntityNode).filter_by(name="NewName").first()
            self.assertIsNotNone(ent)
            self.assertEqual(session.query(EntityNode).filter_by(name="OldName").count(), 0)
            self.assertIn("NewName", ent.text)

            edges = session.query(GraphEdge).all()
            self.assertGreater(len(edges), 0)
            self.assertEqual(edges[0].source_name, "NewName")


if __name__ == "__main__":
    unittest.main()
