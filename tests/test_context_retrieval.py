import unittest
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.memory.db as db_module
from src.memory.db import Base, EntityNode, GraphEdge, EmbeddingIndex, get_session
from src.memory.context_store import retrieve_context


# Mock embedding model
class MockEmbeddingModel:
    def encode(self, text):
        # Return a simple mock embedding based on text length or something deterministic
        return [float(len(text))] * 384


# Use an in-memory database for testing
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


class TestContextRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db_module._engine = test_engine
        db_module.SessionLocal = TestSessionLocal
        Base.metadata.create_all(test_engine)
        cls.user_id = 1
        cls.user_name = "test_user"

    def setUp(self):
        with get_session() as session:
            session.query(EntityNode).delete()
            session.query(GraphEdge).delete()
            session.query(EmbeddingIndex).delete()
            session.commit()

    def test_retrieve_context_basic(self):
        # 1. Setup data
        with get_session() as session:
            # Create an entity
            e1 = EntityNode(id="e1", user_id=self.user_id, name="Python", text="Python is a programming language.")
            session.add(e1)
            # Create an embedding for the entity
            emb1 = EmbeddingIndex(
                user_id=self.user_id,
                collection="entity",
                source_id="e1",
                text_content="Python",
                embedding_json=json.dumps([1.0] * 384)
            )
            session.add(emb1)
            session.commit()

        # 2. Call retrieve_context
        query_emb = [1.0] * 384
        result = retrieve_context(query_emb, self.user_id, k=1, top_n=1)

        # 3. Assertions
        self.assertIn("Relevant entities:", result)
        self.assertIn("[Python]: Python is a programming language.", result)

    def test_retrieve_context_with_edges(self):
        with get_session() as session:
            # Entities
            e1 = EntityNode(id="e1", user_id=self.user_id, name="Alice", text="Alice lives in Paris.")
            e2 = EntityNode(id="e2", user_id=self.user_id, name="Paris", text="Paris is the capital of France.")
            session.add_all([e1, e2])

            # Edge
            edge = GraphEdge(id=1, user_id=self.user_id, source_id="e1", target_id="e2",
                             source_name="Alice", target_name="Paris", relation="lives in")
            session.add(edge)

            # Embeddings
            emb_edge = EmbeddingIndex(
                user_id=self.user_id,
                collection="edge",
                source_id="1",
                text_content="Alice lives in Paris",
                embedding_json=json.dumps([1.0] * 384)
            )
            session.add(emb_edge)
            session.commit()

        query_emb = [1.0] * 384
        result = retrieve_context(query_emb, self.user_id, k=5, top_n=5)

        self.assertIn("Relations:", result)
        self.assertIn("- Alice -- lives in --> Paris", result)
        # Should also include entities connected by the edge
        self.assertIn("[Alice]: Alice lives in Paris.", result)
        self.assertIn("[Paris]: Paris is the capital of France.", result)
