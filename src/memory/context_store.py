"""
Context store — storage and retrieval via SQLite + SQLAlchemy.

STORAGE:
  - add_fact: write to SQLite (dedup by fact_id)
  - update_fact: delete old, insert new
  - delete_fact: natively remove fact (cascades to graph edges)

RETRIEVAL:
  - retrieve_context:
      fetch all user embeddings → compute similarity → rerank seeds → graph expand → rerank final → format
"""

import json
from src.utils import cosine_similarity
from src.memory.hash_utils import fact_id
from src.memory import graph_manager
from src.memory.db import get_session
from src.memory.schema import EntityNode, EmbeddingIndex, GraphEdge


def ensure_user_profile(user_name: str, emb_model):
    """
    Ensure the base profile Entity exists for the user.
    """
    user_fact_text = f"The user's name is {user_name}."
    eid = fact_id("user")

    with get_session() as session:
        existing = session.query(EntityNode).filter_by(id=eid, user_name=user_name).first()
        if not existing:
            emb = emb_model.encode(user_fact_text).tolist()
            new_entity = EntityNode(
                id=eid,
                user_name=user_name,
                name="user",
                topic="profile",
                text=user_fact_text,
                tags_json=json.dumps(["identity", "profile"])
            )
            session.add(new_entity)

            new_emb = EmbeddingIndex(
                user_name=user_name,
                collection="entity",
                source_id=eid,
                text_content=user_fact_text,
                embedding_json=json.dumps(emb)
            )
            session.add(new_emb)
            session.commit()


def _find_closest_entity(session, text: str, emb_model, user_name: str, threshold: float = 0.85) -> str:
    """Find the most semantically similar EntityNode in the DB for a given text."""
    query_emb = emb_model.encode(text).tolist()
    all_entities = session.query(EmbeddingIndex.source_id, EmbeddingIndex.embedding_json).filter_by(
        user_name=user_name, collection="entity"
    ).all()

    best_id = None
    best_score = -1
    for eid, emb_json in all_entities:
        emb = json.loads(emb_json)
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_id = eid

    if best_score >= threshold:
        return best_id
    return None


def delete_fact(fact_text: str, emb_model, user_name: str, memory_base_dir="local_store"):
    """
    Delete an entity from the database entirely, using semantic similarity to find it.
    """
    with get_session() as session:
        target_eid = _find_closest_entity(session, fact_text, emb_model, user_name)
        if target_eid:
            deleted = session.query(EntityNode).filter_by(id=target_eid).delete()
            session.query(EmbeddingIndex).filter_by(source_id=target_eid, collection="entity").delete()
            session.commit()
            return deleted > 0
    return False


def _rerank(cross_encoder, query_text: str, candidates: list,
            top_n: int) -> list:
    """Score candidates with the cross-encoder and sort."""
    if not candidates:
        return []
    pairs = [(query_text, text) for _, text in candidates]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [item for item, score in scored][:top_n]


def retrieve_context(query_emb, user_name: str, cross_encoder=None,
                     query_text: str = "", k: int = 10, top_n: int = 5,
                     memory_base_dir="local_store") -> str:
    with get_session() as session:
        # Load all embeddings (entities and edges) for the user
        all_embs = session.query(EmbeddingIndex).filter_by(
            user_name=user_name
        ).all()

        if not all_embs:
            return ""

        # Step 1: Embedding search
        scored = []
        for row in all_embs:
            emb = json.loads(row.embedding_json)
            score = cosine_similarity(query_emb, emb)
            scored.append((row, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_k_rows = [row for row, _ in scored[:k]]

        if not top_k_rows:
            return ""

        # Step 2: Extract seed Entity IDs from the top-k result
        seed_entity_ids = []
        for row in top_k_rows:
            if row.collection == "entity":
                seed_entity_ids.append(row.source_id)
            elif row.collection == "edge":
                # Find the target edge and get its source/target entities
                edge = session.query(GraphEdge).filter_by(id=int(row.source_id)).first()
                if edge:
                    seed_entity_ids.extend([edge.source_id, edge.target_id])

        seed_entity_ids = list(set(seed_entity_ids))

        if not seed_entity_ids:
            return ""

        # Step 3: Fact-to-Fact Graph expansion
        expanded_ids = graph_manager.expand_nodes(
            seed_entity_ids, user_name, depth=1, max_nodes=20
        )

        final_ids = list(set(seed_entity_ids).union(expanded_ids))

        # Step 4: Load texts for the expanded candidates
        entity_rows = session.query(EntityNode.id, EntityNode.text).filter(
            EntityNode.user_name == user_name,
            EntityNode.id.in_(final_ids)
        ).all()

        candidates = [(r.id, r.text) for r in entity_rows]

        if cross_encoder and query_text:
            candidates = _rerank(
                cross_encoder, query_text, candidates,
                top_n=top_n
            )
        else:
            candidates = candidates[:top_n]

        # Step 5: Format
        if not candidates:
            return ""

        lines = ["Relevant known facts:\n"]
        for eid, text in candidates:
            lines.append(f"- {text}")
        return "\n".join(lines)
